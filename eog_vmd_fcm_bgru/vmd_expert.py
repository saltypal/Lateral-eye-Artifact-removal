"""Inspectable VMD diagnostics and a train-only soft FCM ocular expert."""
from dataclasses import dataclass
import time
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
from vmdpy import VMD
from .signal_features import mode_descriptor


def decompose(values, modes=5, alpha=1000, fs=200, relative_tolerance=False, tolerance=1e-7):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or values.size < 32:
        raise ValueError("VMD needs a finite one-dimensional continuous segment")
    padded = np.pad(values, (0, values.size % 2), mode="edge")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("VMD convergence tolerance must be finite and positive")
    amplitude_scale = float(np.sqrt(np.mean(padded ** 2))) if relative_tolerance else 1.0
    if amplitude_scale <= np.finfo(float).tiny:
        raise ValueError("RMS-normalized VMD requires a nonzero signal")
    start = time.perf_counter()
    vectors, _, frequencies = VMD(padded / amplitude_scale, alpha, 0, modes, 0, 1, tolerance)
    elapsed = time.perf_counter() - start
    if not np.isfinite(vectors).all() or frequencies.size == 0:
        raise RuntimeError("Nonfinite VMD solution")
    vectors = vectors[:, :values.size] * amplitude_scale
    centers = frequencies[-1] * fs
    order = np.argsort(centers)
    vectors, centers = vectors[order], centers[order]
    residual = values - vectors.sum(axis=0)
    energy = np.mean(vectors ** 2, axis=-1)
    fraction = energy / max(energy.sum(), np.finfo(float).tiny)
    frequencies_psd, power = signal.welch(vectors, fs=fs, nperseg=min(512, values.size), nfft=512, axis=-1)
    probability = power / np.maximum(power.sum(axis=-1, keepdims=True), np.finfo(float).tiny)
    centroid = (probability * frequencies_psd).sum(axis=-1)
    bandwidth = np.sqrt((probability * (frequencies_psd - centroid[:, None]) ** 2).sum(axis=-1))
    overlaps = [float(np.minimum(probability[i], probability[i + 1]).sum()) for i in range(modes - 1)]
    diagnostics = {"centers_hz": centers.tolist(), "welch_centroids_hz": centroid.tolist(),
                   "bandwidth_hz": bandwidth.tolist(), "energy_fraction": fraction.tolist(),
                   "adjacent_psd_overlap": overlaps, "nearest_center_hz": float(np.diff(centers).min()),
                   "close_center_pairs_lt_1hz": int((np.diff(centers) < 1).sum()),
                   "iterations": int(len(frequencies)), "hit_iteration_limit": bool(len(frequencies) >= 499),
                   "residual_ratio": float(np.linalg.norm(residual) / max(np.linalg.norm(values), np.finfo(float).tiny)),
                   "runtime_s": elapsed,
                   "rms_normalized_stopping": bool(relative_tolerance), "stopping_tolerance": tolerance,
                   "solver_amplitude_scale": amplitude_scale,
                   "vmdpy_iterative_array_lower_bound_bytes": int(500 * (2 * padded.size) * (modes + 1) * 16)}
    import platform
    diagnostics["measured_process_high_water_rss_kib_linux"] = None
    if platform.system() == "Linux":
        import resource
        diagnostics["measured_process_high_water_rss_kib_linux"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return vectors.astype(np.float32), residual.astype(np.float32), diagnostics


def descriptors(vectors):
    result = np.stack([mode_descriptor(mode) for mode in vectors])
    # Existing descriptor's energy is made relative to mode set energy.
    result[:, 2] /= max(float(result[:, 2].sum()), np.finfo(np.float32).tiny)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class ModeExpert:
    modes: int
    alpha: float
    scaler: StandardScaler
    centers: np.ndarray
    ocular_cluster: int
    relative_tolerance: bool = False
    tolerance: float = 1e-7

    @classmethod
    def fit(cls, training_channels, training_eog, modes=5, alpha=1000, seed=42, relative_tolerance=False, tolerance=1e-7):
        features, evidence = [], []
        for values, references in zip(training_channels, training_eog):
            vectors, _, _ = decompose(values, modes, alpha, relative_tolerance=relative_tolerance, tolerance=tolerance)
            features.extend(descriptors(vectors))
            for vector in vectors:
                correlations = [abs(np.corrcoef(vector, ref)[0, 1]) for ref in references
                                if np.std(vector) > 0 and np.std(ref) > 0]
                evidence.append(max(correlations, default=0.0))
        features = np.asarray(features)
        scaler = StandardScaler().fit(features)
        centers, membership, _, _, _, _, _ = fuzz.cluster.cmeans(scaler.transform(features).T, c=2, m=2,
                                                               error=1e-5, maxiter=300, seed=seed)
        weighted_evidence = (membership * np.asarray(evidence)[None]).sum(axis=-1) / np.maximum(membership.sum(axis=-1), 1e-12)
        return cls(modes, alpha, scaler, centers, int(weighted_evidence.argmax()), relative_tolerance, tolerance)

    def artifact(self, eeg, strength=1.0, soft=True):
        artifacts, all_diagnostics = [], []
        for values in eeg:
            vectors, _, diagnostics = decompose(values, self.modes, self.alpha,
                                                relative_tolerance=self.relative_tolerance, tolerance=self.tolerance)
            membership, _, _, _, _, _ = fuzz.cluster.cmeans_predict(self.scaler.transform(descriptors(vectors)).T,
                                                                   self.centers, m=2, error=1e-5, maxiter=300, seed=0)
            weights = membership[self.ocular_cluster]
            if not soft:
                weights = (weights >= 0.5).astype(float)
            estimate = (vectors * weights[:, None]).sum(axis=0)
            # Local amplitude heuristic; this is NOT a calibrated probability.
            scale = np.median(np.abs(estimate - np.median(estimate))) / 0.67449
            if scale > 0:
                envelope = np.sqrt(signal.convolve(estimate ** 2, np.ones(41) / 41, mode="same"))
                gate = np.clip((envelope / scale - 1.5) / 2.0, 0, 1)
            else:
                gate = np.zeros_like(estimate)
            artifacts.append(strength * estimate * gate if soft else strength * estimate)
            all_diagnostics.append(diagnostics)
        return np.stack(artifacts), all_diagnostics
