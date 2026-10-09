"""Paired recovery and native-EOG association are distinct metric contracts."""
import numpy as np
from scipy.signal import welch
from .reference import correlations


def paired(cleaned, target, fs=200):
    cleaned, target = np.asarray(cleaned, float), np.asarray(target, float)
    if cleaned.shape != target.shape or not np.isfinite(cleaned).all() or not np.isfinite(target).all():
        raise ValueError("Invalid paired prediction")
    energy = float(np.sum(target ** 2))
    error = float(np.sum((cleaned - target) ** 2))
    if energy <= 1e-24:
        snr, status = None, "zero_target_energy"
    elif error == 0:
        snr, status = None, "perfect_reconstruction"
    else:
        snr, status = float(10 * np.log10(energy / error)), "finite"
    cc = []
    for y, s in zip(np.atleast_2d(cleaned), np.atleast_2d(target)):
        if np.std(y) > 1e-12 and np.std(s) > 1e-12:
            cc.append(float(np.corrcoef(y, s)[0, 1]))
    cov_y, cov_s = np.atleast_2d(np.cov(cleaned)), np.atleast_2d(np.cov(target))
    covariance = float(np.linalg.norm(cov_y - cov_s) / max(np.linalg.norm(cov_s), 1e-24))
    frequencies, py = welch(cleaned, fs=fs, window="hann", nperseg=min(512, cleaned.shape[-1]), noverlap=min(256, cleaned.shape[-1] // 2), nfft=512, axis=-1)
    _, ps = welch(target, fs=fs, window="hann", nperseg=min(512, target.shape[-1]), noverlap=min(256, target.shape[-1] // 2), nfft=512, axis=-1)
    bands = {}
    for name, lo, hi in (("delta", .5, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30), ("upper", 30, 40)):
        mask = (frequencies >= lo) & (frequencies < hi)
        ratio = (py[..., mask].sum(-1) + 1e-24) / (ps[..., mask].sum(-1) + 1e-24)
        bands[name + "_db"] = float(np.mean(np.abs(10 * np.log10(ratio))))
    return {"snr_db": snr, "snr_status": status, "rmse": float(np.sqrt(error / target.size)),
            "relative_error": float(np.sqrt(error / energy)) if energy > 1e-24 else None,
            "pearson": float(np.mean(cc)) if cc else None, "covariance": covariance, **bands}


def ocular(eeg, references):
    values = correlations(eeg, references)
    return {"heog_abs": float(np.abs(values[:, 0]).mean()), "veog_abs": float(np.abs(values[:, 1]).mean()),
            "per_channel_signed": values.tolist(), "interpretation": "native ocular-association proxy, not paired reconstruction"}


def preservation_pass(metrics, limits):
    return (metrics["relative_error"] is not None and metrics["relative_error"] <= limits["clean_change"]
            and metrics["alpha_db"] <= limits["alpha_db"] and metrics["beta_db"] <= limits["beta_db"]
            and metrics["covariance"] <= limits["covariance"])
