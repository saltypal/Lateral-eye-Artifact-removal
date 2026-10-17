"""Paper-grounded paired metrics and explicitly labelled preservation proxies."""
import numpy as np
from scipy.signal import welch


def correlations(values, references):
    """Signed Pearson correlation for every EEG/component and H/V reference."""
    values = np.asarray(values, dtype=float)
    references = np.asarray(references, dtype=float)
    values = values - values.mean(axis=-1, keepdims=True)
    references = references - references.mean(axis=-1, keepdims=True)
    numerator = values @ references.T
    denominator = np.linalg.norm(values, axis=-1)[:, None] * np.linalg.norm(references, axis=-1)[None]
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def paired(prediction, target):
    """C×T inputs: per-channel metrics plus a separate pooled-energy SNR.

    Constant-target Pearson is undefined and retained as NaN. Exact recovery is
    +inf SNR, not an arbitrary claimed measured finite noise floor.
    """
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    if prediction.shape != target.shape or target.ndim != 2:
        raise ValueError("Paired EEG must have matching C×T axes")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Paired values must be finite")
    error = target - prediction
    mse = np.mean(error ** 2, axis=-1)
    power = np.mean(target ** 2, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 10 * np.log10(power / mse)
        rrmse = np.sqrt(mse / power)
        pooled_snr = float(10 * np.log10(np.sum(target ** 2) / np.sum(error ** 2)))
    first = prediction - prediction.mean(axis=-1, keepdims=True)
    second = target - target.mean(axis=-1, keepdims=True)
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    cc = np.divide(np.sum(first * second, axis=-1), denominator,
                   out=np.full(len(target), np.nan), where=denominator > 0)
    return {"mse": mse, "rmse": np.sqrt(mse), "rrmse_t": rrmse,
            "pearson_cc": cc, "snr_out_db": snr, "pooled_window_snr_db": pooled_snr}


def preservation(cleaned, original, fs=200):
    """Modification relative to the provided reference; native rest is a proxy."""
    cleaned, original = np.asarray(cleaned, float), np.asarray(original, float)
    power = np.sum(original ** 2)
    relative = np.sqrt(np.sum((cleaned - original) ** 2) / max(power, np.finfo(float).tiny))
    frequency, before = welch(original, fs=fs, nperseg=min(512, original.shape[-1]), axis=-1)
    _, after = welch(cleaned, fs=fs, nperseg=min(512, original.shape[-1]), axis=-1)
    errors = {}
    for band, lower, upper in (("alpha", 8, 13), ("beta", 13, 30)):
        keep = (frequency >= lower) & (frequency < upper)
        p0, p1 = before[:, keep].sum(axis=-1), after[:, keep].sum(axis=-1)
        valid = (p0 > 0) & (p1 > 0)
        errors[band + "_error_db"] = float(np.mean(np.abs(10 * np.log10(p1[valid] / p0[valid])))) if valid.any() else float("nan")
    centered0 = original - original.mean(axis=-1, keepdims=True)
    centered1 = cleaned - cleaned.mean(axis=-1, keepdims=True)
    cov0 = centered0 @ centered0.T / original.shape[-1]
    cov1 = centered1 @ centered1.T / original.shape[-1]
    return {"relative_change": float(relative), "modification_rmse": float(np.sqrt(np.mean((cleaned-original)**2))),
            "covariance_relative_error": float(np.linalg.norm(cov1-cov0) / max(np.linalg.norm(cov0), 1e-30)), **errors}
