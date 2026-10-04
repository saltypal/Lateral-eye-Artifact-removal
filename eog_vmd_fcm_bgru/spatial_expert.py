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
    def fit(cls, calibration_eeg, calibration_eog, method="picard", seed=42, calibration_highpass=None):
        if calibration_eeg.shape[-1] < 2000 or len(calibration_eeg) < 2:
            raise ValueError("ICA unavailable: need >=10 seconds and two valid electrodes")
        centered = calibration_eeg - calibration_eeg.mean(axis=-1, keepdims=True)
        rank = int(np.linalg.matrix_rank(centered))
        if rank < 2:
            raise ValueError("ICA unavailable: deficient rank")
        names = [f"EEG{i}" for i in range(len(calibration_eeg))]
        raw = mne.io.RawArray(calibration_eeg * 1e-6, mne.create_info(names, 200, "eeg"), verbose=False)
        if calibration_highpass is None:
            fitting = raw.copy().filter(l_freq=1, h_freq=None, method="iir", verbose=False)
        else:
            if calibration_highpass.shape != calibration_eeg.shape or not np.isfinite(calibration_highpass).all():
                raise ValueError("Independent-trial highpass data must align with calibration")
            fitting = mne.io.RawArray(calibration_highpass * 1e-6, mne.create_info(names, 200, "eeg"), verbose=False)
        fit_params = {"extended": True, "ortho": False} if method == "picard" else {"extended": True}
        decomposition = mne.preprocessing.ICA(n_components=rank, method=method, fit_params=fit_params,
                                              random_state=seed, max_iter=1000, verbose=False)
        decomposition.fit(fitting, verbose=False)
        if decomposition.n_iter_ >= 1000:
            raise RuntimeError(f"ICA did not converge: method={method}, rank={rank}, channels={len(names)}, samples={calibration_eeg.shape[-1]}, iterations={decomposition.n_iter_}")
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


def fit_asr(eeg, calibration_samples, cutoff=20):
    from mne_denoise.asr import ASR
    calibration = eeg[:, :calibration_samples]
    if calibration.shape[-1] < 2000 or np.linalg.matrix_rank(calibration - calibration.mean(axis=-1, keepdims=True)) < 2:
        raise ValueError("ASR unavailable: insufficient calibration or rank")
    estimator = ASR(sfreq=200, cutoff=cutoff, calibration="auto", random_state=42)
    estimator.fit(calibration)
    return estimator


def asr_correction(eeg, calibration_samples, cutoff=20):
    estimator = fit_asr(eeg, calibration_samples, cutoff)
    corrected = np.asarray(estimator.transform(eeg))
    if corrected.shape != eeg.shape or not np.isfinite(corrected).all():
        raise ValueError("ASR shape/finite contract failed")
    return corrected


def armbr_correction(eeg, names, calibration_samples):
    """Author ARMBR blink baseline, requiring verified frontopolar names."""
    blink_channels = [index for index, name in enumerate(names) if name.strip().upper() in {"FP1", "FP2", "FPZ"}]
    if not blink_channels:
        raise ValueError("ARMBR unavailable: verified frontopolar channel names required")
    from ARMBR import run_armbr
    # The pinned author package uses MNE filtering, which requires float64.
    _, threshold, mask, _, _, projection = run_armbr(np.asarray(eeg[:, :calibration_samples].T, dtype=np.float64), blink_channels, [], 200, -1)
    projection = np.asarray(projection)
    if projection.shape != (len(eeg), len(eeg)) or not np.isfinite(projection).all() or not np.any(mask):
        raise ValueError("ARMBR unavailable: calibration found no valid blink projection")
    corrected = (eeg.T @ projection).T
    if corrected.shape != eeg.shape or not np.isfinite(corrected).all():
        raise ValueError("ARMBR finite/shape contract failed")
    return corrected, {"author_auto_threshold": float(threshold), "calibration_blink_samples": int(np.sum(mask)),
                       "scope": "blink-specific comparator; horizontal-eye efficacy not assumed"}


def session_spatial_calibration(trials, configuration):
    """Fit static transforms on independent, wholly unscored OSF trials.

    Only ICA/regression covariance and source fitting pool independent samples.
    Each high-pass copy is filtered before pooling. ASR/ARMBR never see fake
    trial joins; they calibrate on individual continuous trials or are N/A.
    No artifact/trial labels are consulted by this input-only calibration.
    """
    from .dataset_io import reference_matrix
    if not trials:
        raise ValueError("Empty OSF calibration")
    names = trials[0]["names"]
    if any(trial["names"] != names for trial in trials):
        raise ValueError("Calibration trials have inconsistent valid electrodes")
    references = [reference_matrix(trial)[0] for trial in trials]
    if any(value.shape[0] != references[0].shape[0] for value in references):
        raise ValueError("Calibration EOG references do not align")
    pooled = np.concatenate([trial["eeg"] for trial in trials], axis=1)
    pooled_references = np.concatenate(references, axis=1)
    state = {"errors": {}, "trial_ids": [trial["trial"] for trial in trials], "names": names,
             "ica": None, "asr": None, "armbr_projection": None, "ridge": None}
    if len(pooled_references):
        mean = pooled_references.mean(axis=-1, keepdims=True)
        centered = pooled_references - mean
        covariance = centered @ centered.T
        penalty = float(configuration["ridge_eog"])
        regularization = penalty * np.trace(covariance) / len(centered)
        weights = np.linalg.solve(covariance + regularization * np.eye(len(centered)),
                                  centered @ (pooled - pooled.mean(axis=-1, keepdims=True)).T).T
        state["ridge"] = (weights, mean)
    if "ica" in configuration and len(pooled_references):
        highpass = signal.butter(4, 1, fs=200, btype="highpass", output="sos")
        fitting = np.concatenate([signal.sosfiltfilt(highpass, trial["eeg"], axis=-1) for trial in trials], axis=1)
        methods = [configuration["ica"].split(":")[0]]
        if configuration.get("ica_fallback") and configuration["ica_fallback"] not in methods:
            methods.append(configuration["ica_fallback"])
        state["ica_attempts"] = []
        for method in methods:
            print("OSF input-only ICA calibration", method, "channels", len(names), "samples", pooled.shape[-1], flush=True)
            try:
                state["ica"] = ICAExpert.fit(pooled, pooled_references, method, calibration_highpass=fitting)
                state["ica_method"] = method
                state["errors"].pop("ica", None)
                state["ica_attempts"].append({"method": method, "converged": True,
                                              "iterations": state["ica"].decomposition.n_iter_})
                break
            except Exception as error:
                state["errors"]["ica"] = repr(error)
                state["ica_attempts"].append({"method": method, "converged": False, "error": repr(error)})
    if "asr" in configuration:
        continuous = next((trial["eeg"] for trial in trials if trial["eeg"].shape[-1] >= 2000), None)
        if continuous is None:
            state["errors"]["asr"] = "No independent calibration trial has the required continuous 10 seconds"
        else:
            try:
                state["asr"] = fit_asr(continuous, 2000, int(configuration["asr"]))
            except Exception as error:
                state["errors"]["asr"] = repr(error)
    # The author method calibrates each genuine trial separately. Selection
    # uses its own input-only blink detector, never the ground-truth labels.
    if any(name.upper() in {"FP1", "FP2", "FPZ"} for name in names):
        for trial in trials:
            try:
                from ARMBR import run_armbr
                indices = [index for index, name in enumerate(names) if name.upper() in {"FP1", "FP2", "FPZ"}]
                _, alpha, mask, _, _, projection = run_armbr(np.asarray(trial["eeg"].T, dtype=np.float64), indices, [], 200, -1)
                projection = np.asarray(projection)
                if projection.shape == (len(names), len(names)) and np.isfinite(projection).all() and np.any(mask):
                    state["armbr_projection"] = projection
                    state["armbr_metadata"] = {"calibration_trial": trial["trial"], "author_auto_threshold": float(alpha),
                                               "calibration_blink_samples": int(np.sum(mask))}
                    break
            except Exception as error:
                state["errors"]["armbr"] = repr(error)
    if state["armbr_projection"] is None:
        state["errors"].setdefault("armbr", "No input-only valid blink projection in the declared calibration trials")
    return state
