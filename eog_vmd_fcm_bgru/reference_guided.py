"""EOG-assisted mode selection and raw-frontal support for posterior ICA.

Correlations and cluster memberships are evidence, not artifact probabilities.
Every correction estimates a residual against the original aligned EEG.
"""
from dataclasses import dataclass
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from .channel_regions import region_ids
from .vmd_expert import descriptors


def aligned_correlations(vectors, references):
    """Signed Pearson correlations [mode, reference], zero for constant rows."""
    vectors = np.asarray(vectors, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    if vectors.ndim != 2 or references.ndim != 2 or vectors.shape[1] != references.shape[1]:
        raise ValueError("Modes and references must have aligned [rows, samples] shapes")
    if not np.isfinite(vectors).all() or not np.isfinite(references).all():
        raise ValueError("Mode/reference values must be finite")
    centered = vectors - vectors.mean(axis=1, keepdims=True)
    eye_centered = references - references.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(centered, axis=1)[:, None] * np.linalg.norm(eye_centered, axis=1)[None]
    result = np.zeros((len(vectors), len(references)), dtype=np.float64)
    np.divide(centered @ eye_centered.T, denominator, out=result, where=denominator > 0)
    return np.clip(result, -1, 1)


def reference_projection(vectors, references, penalty=0.01):
    """Project only centered activity onto normalized EOG; preserve the mean."""
    vectors = np.asarray(vectors, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    aligned_correlations(vectors, references)  # Validate alignment and finiteness.
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("Reference ridge penalty must be positive and finite")
    reference = references - references.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(reference ** 2, axis=1))
    keep = rms > np.finfo(float).tiny
    if not keep.any():
        return np.zeros_like(vectors)
    reference = reference[keep] / rms[keep, None]
    covariance = reference @ reference.T
    regularization = penalty * np.trace(covariance) / len(reference)
    centered = vectors - vectors.mean(axis=1, keepdims=True)
    weights = np.linalg.solve(covariance + regularization * np.eye(len(reference)), reference @ centered.T).T
    return weights @ reference


def lagged_reference_matrix(references, lags=(0,)):
    """Return zero-padded signed HEOG/VEOG lag regressors.

    A lag is expressed in samples and means ``reference[t - lag]``.  This is
    deliberately a finite-window construction: it never wraps samples from
    the end of an epoch to its start.  The caller chooses lags during
    development and freezes them before validation/test use.
    """
    references = np.asarray(references, dtype=np.float64)
    if references.ndim != 2 or not np.isfinite(references).all():
        raise ValueError("References must have finite [reference, samples] shape")
    if not lags:
        raise ValueError("At least one EOG lag is required")
    original_lags = tuple(lags)
    lags = tuple(int(value) for value in original_lags)
    if any(value != original for value, original in zip(lags, original_lags)):
        raise ValueError("EOG lags must be integers")
    rows = []
    samples = references.shape[1]
    for lag in lags:
        shifted = np.zeros_like(references)
        if abs(lag) < samples:
            if lag >= 0:
                shifted[:, lag:] = references[:, :samples - lag]
            else:
                shifted[:, :samples + lag] = references[:, -lag:]
        rows.append(shifted)
    return np.concatenate(rows, axis=0)


def signed_lagged_reference_projection(vectors, references, lags=(0,), penalty=0.01):
    """Window-adaptive joint ridge projection onto signed, lagged EOG.

    This is an EOG-assisted *teacher* operation.  Coefficients are estimated
    independently in each correction window because VMD mode indices are not
    stable identities across windows.  Only the recipe (lags, penalty and
    global mode-gating rule) is transferable.  Means in the EEG modes are
    preserved, and zero/constant EOG regressors are ignored.

    Returns ``(projected, details)`` where ``projected`` has the shape of
    ``vectors`` and details are safe to write to a run manifest.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Modes must have finite [mode, samples] shape")
    regressors = lagged_reference_matrix(references, lags)
    if regressors.shape[1] != vectors.shape[1]:
        raise ValueError("Modes and references must have the same sample count")
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("Reference ridge penalty must be positive and finite")

    centered = regressors - regressors.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered ** 2, axis=1))
    keep = rms > np.finfo(float).tiny
    if not keep.any():
        return np.zeros_like(vectors), {
            "lags": list(map(int, lags)), "usable_regressors": 0,
            "coefficient_shape": [int(vectors.shape[0]), 0],
            "window_adaptive": True,
        }
    design = centered[keep] / rms[keep, None]
    covariance = design @ design.T
    regularization = penalty * np.trace(covariance) / len(design)
    mode_centered = vectors - vectors.mean(axis=1, keepdims=True)
    coefficients = np.linalg.solve(
        covariance + regularization * np.eye(len(design)), design @ mode_centered.T
    ).T
    return coefficients @ design, {
        "lags": list(map(int, lags)), "usable_regressors": int(keep.sum()),
        "coefficient_shape": [int(value) for value in coefficients.shape],
        "window_adaptive": True,
    }


def soft_correlation_gate(correlations, threshold):
    """Zero below threshold, linear attenuation up to absolute correlation one."""
    if not np.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError("Correlation threshold must be in [0, 1)")
    if correlations.ndim != 2 or correlations.shape[1] == 0:
        raise ValueError("At least one aligned ocular reference is required")
    evidence = np.max(np.abs(correlations), axis=1)
    return np.clip((evidence - threshold) / (1 - threshold), 0, 1)


@dataclass
class ReferenceModeSelector:
    scaler: StandardScaler
    centers: np.ndarray
    cluster_reference_evidence: np.ndarray
    reference_names: tuple
    training_modes: int

    @staticmethod
    def feature_rows(vectors, references):
        correlations = aligned_correlations(vectors, references)
        return np.concatenate([descriptors(vectors), np.abs(correlations)], axis=1), correlations

    @classmethod
    def fit(cls, training_modes, training_references, clusters=2, seed=42,
            reference_names=("HEOG", "VEOG")):
        if len(training_modes) != len(training_references) or not training_modes:
            raise ValueError("Training mode sets and ocular references must align")
        if clusters not in (2, 3):
            raise ValueError("This declared search supports two or three clusters")
        rows, evidence = [], []
        for vectors, references in zip(training_modes, training_references):
            if len(references) != len(reference_names):
                raise ValueError("Ocular reference names and rows do not align")
            features, correlations = cls.feature_rows(vectors, references)
            rows.extend(features)
            evidence.extend(np.abs(correlations))
        rows, evidence = np.asarray(rows), np.asarray(evidence)
        if len(rows) < clusters:
            raise ValueError("Insufficient training modes for clustering")
        scaler = StandardScaler().fit(rows)
        centers, membership, _, _, _, iterations, _ = fuzz.cluster.cmeans(
            scaler.transform(rows).T, c=clusters, m=2, error=1e-5, maxiter=300, seed=seed)
        if iterations >= 299 or not np.isfinite(centers).all():
            raise RuntimeError("Training fuzzy clustering did not converge")
        # Keep HEOG/VEOG evidence separate; a cluster is not forced to be ocular.
        cluster_evidence = membership @ evidence / membership.sum(axis=1, keepdims=True)
        return cls(scaler, centers, cluster_evidence, tuple(reference_names), len(rows))

    def evidence(self, vectors, references):
        if len(references) != len(self.reference_names):
            raise ValueError("Inference ocular-reference ordering differs from training")
        features, correlations = self.feature_rows(vectors, references)
        membership, _, _, _, iterations, _ = fuzz.cluster.cmeans_predict(
            self.scaler.transform(features).T, self.centers, m=2, error=1e-5, maxiter=300, seed=42)
        if iterations >= 299 or not np.isfinite(membership).all():
            raise RuntimeError("Fuzzy mode assignment did not converge")
        cluster_evidence = membership.T @ self.cluster_reference_evidence
        return {"correlations": correlations, "membership": membership.T,
                "cluster_weight": np.max(cluster_evidence, axis=1)}


def selected_mode_residual(vectors, references, threshold, selector=None,
                           projection=False, penalty=0.01):
    correlations = aligned_correlations(vectors, references)
    weights = soft_correlation_gate(correlations, threshold)
    if selector is not None:
        weights = weights * selector.evidence(vectors, references)["cluster_weight"]
    removed_vectors = reference_projection(vectors, references, penalty) if projection else vectors
    return (weights[:, None] * removed_vectors).sum(axis=0), {
        "signed_reference_correlations": correlations.tolist(), "mode_weights": weights.tolist(),
        "selection": "correlation" if selector is None else "fcm_and_correlation",
        "removed_activity": "EOG_projection" if projection else "whole_selected_mode"}


def frontal_support_indices(names, limit=4):
    """Name-based support, prioritizing bilateral frontopolar/lateral electrodes."""
    normalized = [name.strip().upper() for name in names]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate electrode names make routing ambiguous")
    if limit < 0 or int(limit) != limit:
        raise ValueError("Support count must be a nonnegative integer")
    frontal = set(np.flatnonzero(region_ids(names) == 0).tolist())
    priority = ["FP1", "FP2", "F7", "F8", "AF7", "AF8", "AF3", "AF4", "F3", "F4", "FPZ", "FZ"]
    lookup = {name: index for index, name in enumerate(normalized)}
    ordered = [lookup[name] for name in priority if name in lookup and lookup[name] in frontal]
    ordered.extend(sorted(frontal.difference(ordered), key=lambda index: normalized[index]))
    return ordered[:limit]


def posterior_ica_indices(names, support_count=4):
    targets = np.flatnonzero(region_ids(names) == 1).tolist()
    if not targets:
        raise ValueError("Verified posterior electrodes are required")
    support = frontal_support_indices(names, support_count)
    return targets + support, targets, support


def routed_ica_residual(eeg, fit_indices, target_indices, expert, threshold=0.8, strength=0.25):
    """ICA sees raw support rows, but modifies only the declared target rows."""
    if len(fit_indices) != len(set(fit_indices)) or not set(target_indices).issubset(fit_indices):
        raise ValueError("ICA fitting rows must be unique and include every target row")
    if any(index < 0 or index >= len(eeg) for index in fit_indices) or not 0 <= strength <= 1:
        raise ValueError("Invalid ICA channel routing or strength")
    estimated = expert.residual(eeg[fit_indices], threshold=threshold)
    if estimated.shape != eeg[fit_indices].shape or not np.isfinite(estimated).all():
        raise ValueError("ICA residual violates its finite/alignment contract")
    positions = {index: position for position, index in enumerate(fit_indices)}
    residual = np.zeros_like(eeg)
    for index in target_indices:
        residual[index] = strength * estimated[positions[index]]
    return residual
