"""Calibration-excluded EOG ridge, rank-aware ICA, ICA-source VMD and ASR."""
from dataclasses import dataclass
import numpy as np
import mne
from scipy import signal
from .vmd_expert import decompose


def ridge_residual(eeg, eog, calibration_samples, penalty=1):
    reference = eog[:, :calibration_samples]
    mean = reference.mean(axis=-1, keepdims=True)
    centered = reference - mean
    covariance = centered @ centered.T
    regularization = penalty * np.trace(covariance) / max(len(reference), 1)
    weights = np.linalg.solve(covariance + regularization * np.eye(len(reference)),
                              centered @ (eeg[:, :calibration_samples] - eeg[:, :calibration_samples].mean(axis=-1, keepdims=True)).T).T
    return weights @ (eog - mean)


@dataclass
class ICAExpert:
    decomposition: object
    names: list
    ocular_scores: np.ndarray
    source_to_scalp: np.ndarray
    rank: int

    @classmethod
    def fit(cls, calibration_eeg, calibration_eog, method="picard", seed=42):
        if calibration_eeg.shape[-1] < 2000 or len(calibration_eeg) < 2:
            raise ValueError("ICA unavailable: need >=10 seconds and two valid electrodes")
        centered = calibration_eeg - calibration_eeg.mean(axis=-1, keepdims=True)
        rank = int(np.linalg.matrix_rank(centered))
        if rank < 2:
            raise ValueError("ICA unavailable: deficient rank")
        names = [f"EEG{i}" for i in range(len(calibration_eeg))]
        raw = mne.io.RawArray(calibration_eeg * 1e-6, mne.create_info(names, 200, "eeg"), verbose=False)
        fitting = raw.copy().filter(l_freq=1, h_freq=None, method="iir", verbose=False)
        fit_params = {"extended": True, "ortho": False} if method == "picard" else {"extended": True}
        decomposition = mne.preprocessing.ICA(n_components=rank, method=method, fit_params=fit_params,
                                              random_state=seed, max_iter=1000, verbose=False)
        decomposition.fit(fitting, verbose=False)
        if decomposition.n_iter_ >= 1000:
            raise RuntimeError("ICA did not converge")
        sources = decomposition.get_sources(raw).get_data()
        scores = np.asarray([max((abs(np.corrcoef(source, reference)[0, 1]) for reference in calibration_eog
                                  if np.std(reference) > 0), default=0) for source in sources])
        # Fitted back-projection is input-only and uses no paired clean target.
        source_to_scalp = np.linalg.lstsq((sources - sources.mean(axis=-1, keepdims=True)).T, centered.T, rcond=None)[0].T
        return cls(decomposition, names, scores, source_to_scalp, rank)

    def residual(self, eeg, threshold=0.3, source_vmd=False, modes=5, alpha=1000):
        raw = mne.io.RawArray(eeg * 1e-6, mne.create_info(self.names, 200, "eeg"), verbose=False)
        candidates = np.flatnonzero(self.ocular_scores >= threshold)
        if candidates.size == 0:
            return np.zeros_like(eeg)
        if not source_vmd:
            reconstructed = self.decomposition.apply(raw.copy(), exclude=candidates.tolist(), verbose=False).get_data() * 1e6
            return eeg - reconstructed
        sources = self.decomposition.get_sources(raw).get_data()
        removed = np.zeros_like(sources)
        for index in candidates:
            vectors, _, diagnostics = decompose(sources[index], modes, alpha)
            centers = np.asarray(diagnostics["centers_hz"])
            ocular = vectors[centers <= 7.5].sum(axis=0)
            scale = np.median(np.abs(ocular - np.median(ocular))) / 0.67449
            if scale > 0:
                envelope = np.sqrt(signal.convolve(ocular ** 2, np.ones(41) / 41, mode="same"))
                gate = np.clip((envelope / scale - 1.5) / 2, 0, 1)
                removed[index] = gate * ocular
        return self.source_to_scalp @ removed


def asr_correction(eeg, calibration_samples, cutoff=20):
    from mne_denoise.asr import ASR
    calibration = eeg[:, :calibration_samples]
    if calibration.shape[-1] < 2000 or np.linalg.matrix_rank(calibration - calibration.mean(axis=-1, keepdims=True)) < 2:
        raise ValueError("ASR unavailable: insufficient calibration or rank")
    estimator = ASR(sfreq=200, cutoff=cutoff, calibration="auto", random_state=42)
    estimator.fit(calibration)
    corrected = np.asarray(estimator.transform(eeg))
    if corrected.shape != eeg.shape or not np.isfinite(corrected).all():
        raise ValueError("ASR shape/finite contract failed")
    return corrected
