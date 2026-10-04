"""Explicit paired metrics and labeled OSF proxies. Never invent clean OSF."""
import numpy as np
from scipy import signal, stats

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
PSD_CONTRACT = {"fs": 200, "window": "hann", "nperseg": 512, "noverlap": 256, "nfft": 512, "detrend": "constant", "scaling": "density"}


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def welch(values):
    settings = PSD_CONTRACT.copy()
    settings["nperseg"] = min(settings["nperseg"], values.shape[-1])
    settings["noverlap"] = min(settings["noverlap"], settings["nperseg"] // 2)
    return signal.welch(values, axis=-1, **settings)


def band_power(values):
    frequency, power = welch(values)
    bin_width = frequency[1] - frequency[0]
    return {name: power[..., (frequency >= low) & (frequency < high)].sum(axis=-1) * bin_width
            for name, (low, high) in BANDS.items()}


def relative_norm(error, reference):
    denominator = np.linalg.norm(reference)
    return float(np.linalg.norm(error) / denominator) if denominator > 0 else float("nan")


def paired_metrics(estimate, target):
    if estimate.shape != target.shape:
        raise ValueError("No implicit trim is permitted in evaluation")
    error = estimate - target
    _, predicted_psd = welch(estimate)
    _, target_psd = welch(target)
    predicted_bands, target_bands = band_power(estimate), band_power(target)
    noise_power, target_energy = float(np.sum(error ** 2)), float(np.sum(target ** 2))
    result = {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
              "pearson": float(np.nanmean([correlation(a, b) for a, b in zip(estimate, target)])),
              "spearman": float(np.nanmean([stats.spearmanr(a, b).statistic for a, b in zip(estimate, target)])),
              "nrmse_std": float(np.sqrt(np.mean(error ** 2)) / np.std(target)) if np.std(target) > 0 else float("nan"),
              "rrmse": relative_norm(error, target),
              "snr_db": float(10 * np.log10(target_energy / noise_power)) if target_energy > 0 and noise_power > 0 else float("nan"),
              "spectral_rrmse": relative_norm(predicted_psd - target_psd, target_psd),
              "covariance_error": relative_norm(np.atleast_2d(np.cov(estimate)) - np.atleast_2d(np.cov(target)), np.atleast_2d(np.cov(target)))}
    for name in BANDS:
        valid = (target_bands[name] > 0) & (predicted_bands[name] > 0)
        result[f"{name}_error_db"] = float(np.mean(np.abs(10 * np.log10(predicted_bands[name][valid] / target_bands[name][valid])))) if valid.any() else float("nan")
    return result


def modification_metrics(estimate, raw):
    change = estimate - raw
    tolerance = 1e-6 + 1e-3 * np.sqrt(np.mean(raw ** 2, axis=-1, keepdims=True))
    return {"change_rms": float(np.sqrt(np.mean(change ** 2))),
            "relative_change": relative_norm(change, raw), "modified_fraction": float(np.mean(np.abs(change) > tolerance))}


def ocular_proxies(raw, cleaned, eog, labels):
    result = modification_metrics(cleaned, raw)
    result["energy_retained_ratio"] = relative_norm(cleaned, raw) ** 2
    result["raw_preservation_correlation"] = float(np.nanmean([correlation(a, b) for a, b in zip(raw, cleaned)]))
    for name, reference in eog.items():
        before = np.asarray([abs(correlation(channel, reference)) for channel in raw])
        after = np.asarray([abs(correlation(channel, reference)) for channel in cleaned])
        result[f"{name.lower()}_abs_corr_before"] = float(np.nanmean(before))
        result[f"{name.lower()}_abs_corr_after"] = float(np.nanmean(after))
        result[f"{name.lower()}_corr_reduction"] = float(np.nanmean(before - after))
    for name, codes in {"blink": [5], "horizontal": [1, 2], "vertical": [3, 4], "rest": [6]}.items():
        selected = np.isin(labels, codes) if labels is not None else np.zeros(raw.shape[-1], dtype=bool)
        result[f"{name}_samples"] = int(selected.sum())
        result[f"{name}_rms_ratio"] = relative_norm(cleaned[:, selected], raw[:, selected]) if selected.any() else float("nan")
        result[f"{name}_relative_change"] = relative_norm(cleaned[:, selected] - raw[:, selected], raw[:, selected]) if selected.any() else float("nan")
    # Spectral stability needs contiguous rest intervals; concatenating gaps
    # would create artificial edges and invalidate a frequency comparison.
    rest = labels == 6 if labels is not None else np.zeros(raw.shape[-1], dtype=bool)
    changes = np.diff(np.pad(rest.astype(int), (1, 1)))
    blocks = [(begin, end) for begin, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
              if end - begin >= PSD_CONTRACT["nperseg"]]
    result["rest_spectral_contiguous_samples"] = sum(end - begin for begin, end in blocks)
    rest_measurements = [(end - begin, paired_metrics(cleaned[:, begin:end], raw[:, begin:end])) for begin, end in blocks]
    for name in ["spectral_rrmse", "alpha_error_db", "beta_error_db", "covariance_error"]:
        valid = [(weight, scores[name]) for weight, scores in rest_measurements if np.isfinite(scores[name])]
        result["rest_raw_" + name] = float(np.average([score for _, score in valid], weights=[weight for weight, _ in valid])) if valid else float("nan")
    return result
