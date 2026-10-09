"""Robust EOG-association metrics with explicit degeneracy and cross-fit rules."""
from __future__ import annotations

import numpy as np


def _as_rows(values, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 1:
        result = result[None]
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite [rows,samples]")
    return result


def signed_correlation(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Correlation inputs must be aligned finite vectors")
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > np.finfo(float).tiny else float("nan")


def lagged_hv_correlation(eeg, references, max_lag: int = 20) -> dict:
    eeg, references = _as_rows(eeg, "eeg"), _as_rows(references, "references")
    if references.shape[0] != 2 or eeg.shape[-1] != references.shape[-1] or max_lag < 0:
        raise ValueError("Expected aligned EEG and exactly HEOG/VEOG references")
    lags = range(-max_lag, max_lag + 1)
    result = {}
    for name, reference in zip(("heog", "veog"), references):
        values = []
        for lag in lags:
            left, right = (eeg[:, lag:], reference[:-lag]) if lag > 0 else (eeg[:, :lag], reference[-lag:]) if lag < 0 else (eeg, reference)
            values.append([signed_correlation(channel, right) for channel in left])
        matrix = np.asarray(values)
        magnitude = np.nanmean(np.abs(matrix), axis=1)
        if np.isfinite(magnitude).any():
            index = int(np.nanargmax(magnitude))
            result[name] = {"lag_samples": int(list(lags)[index]), "mean_abs_correlation": float(magnitude[index]),
                            "signed_channel_correlations": matrix[index].tolist()}
        else:
            result[name] = {"lag_samples": None, "mean_abs_correlation": float("nan"), "signed_channel_correlations": [float("nan")] * len(eeg)}
    return result


def _design(references, lags):
    references = _as_rows(references, "references")
    maximum = max(abs(int(lag)) for lag in lags)
    if references.shape[-1] <= 2 * maximum + len(references):
        raise ValueError("Reference interval too short for requested lags")
    target = references[:, maximum:references.shape[-1] - maximum]
    columns = []
    for lag in lags:
        columns.extend(references[:, maximum + lag:references.shape[-1] - maximum + lag])
    return np.stack(columns, axis=1), maximum


def cross_fitted_reference_r2(eeg_fit, references_fit, eeg_eval, references_eval, *, lags=(-4, -2, 0, 2, 4), ridge=1e-3) -> dict:
    """Fit EOG->EEG projections on one interval and score explained power on another."""
    eeg_fit, eeg_eval = _as_rows(eeg_fit, "eeg_fit"), _as_rows(eeg_eval, "eeg_eval")
    design_fit, trim_fit = _design(references_fit, lags)
    design_eval, trim_eval = _design(references_eval, lags)
    if eeg_fit.shape[0] != eeg_eval.shape[0] or ridge <= 0:
        raise ValueError("Channel count/ridge invalid")
    y_fit = eeg_fit[:, trim_fit:eeg_fit.shape[-1] - trim_fit].T
    y_eval = eeg_eval[:, trim_eval:eeg_eval.shape[-1] - trim_eval].T
    mean = y_fit.mean(axis=0, keepdims=True)
    gram = design_fit.T @ design_fit + ridge * np.eye(design_fit.shape[1])
    coefficients = np.linalg.solve(gram, design_fit.T @ (y_fit - mean))
    predicted = design_eval @ coefficients + mean
    residual = y_eval - predicted
    total = np.sum((y_eval - y_eval.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1 - np.sum(residual ** 2, axis=0) / total
    r2[total <= np.finfo(float).tiny] = np.nan
    return {"r2_by_channel": r2.tolist(), "mean_r2": float(np.nanmean(r2)) if np.isfinite(r2).any() else float("nan"),
            "lags": list(lags), "ridge": ridge}


def grouped_scores(rows, group_key: str = "participant_id") -> list[dict]:
    """Macro-average per group; missing groups stay explicit rather than becoming one pseudo-subject."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        value = row.get(group_key)
        key = str(value) if value is not None else "__unverified__:" + str(row.get("record_id", "unknown"))
        groups.setdefault(key, []).append(row)
    result = []
    for key, items in sorted(groups.items()):
        numeric = {name: [float(item[name]) for item in items if name in item and np.isfinite(item[name])]
                   for name in {name for item in items for name in item}}
        result.append({"group": key, "n": len(items), **{name: float(np.mean(values)) for name, values in numeric.items() if values}})
    return result
