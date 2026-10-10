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


def project(values, references, lags=(0,), penalty=0.01, *, reference_baseline=None):
    """Window-adaptive ridge, optionally anchored to an unscored EOG baseline.

    Coefficients always fit centered scoring data without a clean target.
    The default artifact has zero scoring-window mean, preserving the old
    algorithm. An explicit calibration baseline instead retains reference
    excursions relative to that baseline. This experimental option must be
    evaluated for EEG preservation before replacing the default.
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    design, valid = lag_matrix(references, lags)
    if values.shape[-1] != design.shape[-1] or valid.sum() < 2 * design.shape[0]:
        raise ValueError("Reference alignment/effective sample failure")
    mean = design[:, valid].mean(axis=1, keepdims=True)
    scale = design[:, valid].std(axis=1, keepdims=True)
    nonconstant=scale[:,0]>0
    safe_scale=np.where(nonconstant[:,None],scale,1.)
    design = (design - mean) / safe_scale
    design[~nonconstant] = 0
    centered = values - values[:, valid].mean(axis=1, keepdims=True)
    covariance = design[:, valid] @ design[:, valid].T / valid.sum()
    cross = design[:, valid] @ centered[:, valid].T / valid.sum()
    coefficients = np.linalg.solve(covariance + penalty * np.eye(len(design)), cross)
    if reference_baseline is not None:
        baseline=np.asarray(reference_baseline,dtype=float)
        if baseline.shape!=(len(design),) or not np.isfinite(baseline).all():
            raise ValueError("Calibration reference baseline must match lag-major reference rows")
        design += (mean-baseline[:,None])/safe_scale
        design[~nonconstant]=0
    artifact = coefficients.T @ design
    artifact[:, ~valid] = 0
    return artifact, valid


def calibration_reference_baseline(references, boundaries=(), lags=(0,)):
    """Unscored lag-major EOG means; unavailable samples never cross joins."""
    references=np.asarray(references,dtype=float)
    design,valid=lag_matrix(references,lags)
    for boundary in boundaries:
        if int(boundary)!=boundary or not 0<int(boundary)<references.shape[-1]:
            raise ValueError("Invalid calibration trial boundary")
        boundary=int(boundary)
        for lag in lags:
            if lag>0:
                valid[boundary:boundary+lag]=False
            elif lag<0:
                valid[boundary+lag:boundary]=False
    if valid.sum()<2*len(design):
        raise ValueError("Insufficient calibration samples for an EOG baseline")
    return design[:,valid].mean(axis=-1)


def correlations(values, references, max_lag=20):
    """Return signed peak association [channel,reference]; each channel stays distinct."""
    values = np.atleast_2d(np.asarray(values,dtype=float))
    references = np.atleast_2d(np.asarray(references,dtype=float))
    best = np.zeros((len(values), len(references)))
    for lag in range(-max_lag, max_lag + 1):
        design, valid = lag_matrix(references, (lag,))
        x = values[:, valid] - values[:, valid].mean(axis=-1, keepdims=True)
        r = design[:, valid] - design[:, valid].mean(axis=-1, keepdims=True)
        xnorm=np.linalg.norm(x,axis=1,keepdims=True)
        rnorm=np.linalg.norm(r,axis=1,keepdims=True)
        x=np.divide(x,xnorm,out=np.zeros_like(x),where=xnorm>0)
        r=np.divide(r,rnorm,out=np.zeros_like(r),where=rnorm>0)
        current = np.clip(x @ r.T,-1.,1.)
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
