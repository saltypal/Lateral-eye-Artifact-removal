"""Joint signed EOG regression with one explicit finite-segment lag convention."""
import numpy as np


def lag_matrix(values, lags=(0,)):
    """Rows are lag-major; positive lag is R(t-lag), with leading unavailable samples."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or not lags:
        raise ValueError("Finite [reference,time] values and lags required")
    length = values.shape[-1]
    rows = []
    valid = np.ones(length, bool)
    for lag in lags:
        if int(lag) != lag or abs(lag) >= length:
            raise ValueError("Invalid finite-segment lag")
        lag = int(lag)
        shifted = np.zeros_like(values)
        if lag > 0:
            shifted[:, lag:] = values[:, :-lag]
            valid[:lag] = False
        elif lag < 0:
            shifted[:, :lag] = values[:, -lag:]
            valid[lag:] = False
        else:
            shifted[:] = values
        rows.extend(shifted)
    return np.asarray(rows), valid


def project(values, references, lags=(0,), penalty=0.01):
    """Window-adaptive ridge; no clean target, no cross-window mode coefficients."""
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    design, valid = lag_matrix(references, lags)
    if values.shape[-1] != design.shape[-1] or valid.sum() < 2 * design.shape[0]:
        raise ValueError("Reference alignment/effective sample failure")
    mean = design[:, valid].mean(axis=1, keepdims=True)
    scale = design[:, valid].std(axis=1, keepdims=True)
    design = (design - mean) / np.maximum(scale, 1e-12)
    design[scale[:, 0] <= 1e-12] = 0
    centered = values - values[:, valid].mean(axis=1, keepdims=True)
    covariance = design[:, valid] @ design[:, valid].T / valid.sum()
    cross = design[:, valid] @ centered[:, valid].T / valid.sum()
    coefficients = np.linalg.solve(covariance + penalty * np.eye(len(design)), cross)
    artifact = coefficients.T @ design
    artifact[:, ~valid] = 0
    return artifact, valid


def correlations(values, references, max_lag=20):
    """Return signed peak association [channel,reference]; each channel stays distinct."""
    values = np.atleast_2d(values)
    references = np.atleast_2d(references)
    best = np.zeros((len(values), len(references)))
    for lag in range(-max_lag, max_lag + 1):
        design, valid = lag_matrix(references, (lag,))
        x = values[:, valid] - values[:, valid].mean(axis=-1, keepdims=True)
        r = design[:, valid] - design[:, valid].mean(axis=-1, keepdims=True)
        denominator = np.linalg.norm(x, axis=1)[:, None] * np.linalg.norm(r, axis=1)[None]
        current = np.divide(x @ r.T, denominator, out=np.zeros_like(best), where=denominator > 1e-12)
        best = np.where(np.abs(current) > np.abs(best), current, best)
    return best


def signed_context(eeg, names, regions, hemispheres):
    summaries, available = [], []
    for side in (0, 1, 2):
        indices = [i for i in range(len(names)) if regions[i] == 0 and hemispheres[i] == side]
        available.append(bool(indices))
        summaries.append(eeg[indices].mean(axis=0) if indices else np.zeros(eeg.shape[-1]))
    left, right, midline = summaries
    common = np.mean([s for s, ok in zip(summaries, available) if ok], axis=0) if any(available) else np.zeros(eeg.shape[-1])
    lateral = right - left if available[0] and available[1] else np.zeros_like(common)
    return {"common": common, "lateral": lateral, "midline": midline, "available": available}
