"""Calibration-only CCA direction selection; an explicitly labelled adaptation."""
from dataclasses import dataclass
import json

import numpy as np
from scipy.linalg import eigh

from .io import atomic_json
from .nonfrontal_challengers import balanced_gain, finite_matrix, rest_whitened_projection
from .nonfrontal_ridge import FrozenRidgeModel, fit_frozen_ridge
from .posterior import boundary_valid
from .reference import lag_matrix

METHODS = ("ridge_direct", "reference_rank2_control", "cca_reference_rank1",
           "cca_reference_rank2", "cca_eeg_rank1", "cca_eeg_rank2")


def covariance_roots(covariance, regularization):
    """Symmetric square roots in standardized coordinates, including rank floors."""
    covariance = np.asarray(covariance, dtype=float)
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or not np.isfinite(covariance).all() or regularization < 0):
        raise ValueError("Invalid CCA covariance")
    values, vectors = eigh((covariance + covariance.T) / 2)
    floor = max(float(np.max(values)) * 1e-10, 1e-12)
    roots = np.sqrt(np.maximum(values + regularization, floor))
    root = (vectors * roots) @ vectors.T
    inverse = (vectors / roots) @ vectors.T
    return root, inverse


def reference_projection(root, inverse, directions, selected):
    """Full reference basis gives identity; truncation is the substantive change."""
    selected = np.asarray(selected, dtype=int)
    basis = directions[:, selected]
    return root @ basis @ basis.T @ inverse


@dataclass
class CCAExperts:
    ids: np.ndarray
    field_model: object
    eeg_mean: np.ndarray
    eeg_scale: np.ndarray
    eeg_directions: np.ndarray
    reference_root: np.ndarray
    reference_inverse: np.ndarray
    reference_directions: np.ndarray
    correlations: np.ndarray
    singular_values: np.ndarray
    admitted: np.ndarray
    eeg_backprojections: dict
    rest_projection: np.ndarray
    failures: dict


def fit_experts(calibration, policy):
    if (not np.isfinite(policy["cca_regularization"]) or policy["cca_regularization"] < 0
            or not np.isfinite(policy["cca_r2_threshold"]) or not 0 <= policy["cca_r2_threshold"] <= 1):
        raise ValueError("Invalid CCA regularization/admission policy")
    eeg = finite_matrix(calibration["eeg"], "calibration EEG")
    refs = finite_matrix(calibration["references"], "calibration EOG")
    mask, types = np.asarray(calibration["mask"], dtype=bool), np.asarray(calibration["trial_types"])
    if (mask.shape != (len(eeg),) or refs.shape != (2, eeg.shape[-1])
            or types.shape != (eeg.shape[-1],) or not mask.any()):
        raise ValueError("CCA calibration axes differ")
    ids = np.flatnonzero(mask)
    boundaries, lags = tuple(calibration["boundaries"]), tuple(policy["lags"])
    field = fit_frozen_ridge(eeg[ids], refs, boundaries, lags, policy["penalty"])
    design, valid = field._standardized_design(refs)
    valid &= boundary_valid(eeg.shape[-1], boundaries, lags)
    rest = valid & (types == 1)
    if rest.sum() < max(128, 2 * len(ids)):
        raise ValueError("Insufficient resting CCA calibration after lag/trial exclusions")
    mean = eeg[ids][:, valid].mean(-1, keepdims=True)
    scale = eeg[ids][:, valid].std(-1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.)
    x = (eeg[ids][:, valid] - mean) / scale
    r = design[:, valid]
    _, wx = covariance_roots(x @ x.T / x.shape[-1], policy["cca_regularization"])
    rr, wr = covariance_roots(r @ r.T / r.shape[-1], policy["cca_regularization"])
    u, singular, vt = np.linalg.svd(wx @ (x @ r.T / x.shape[-1]) @ wr, full_matrices=True)
    eeg_directions, reference_directions = u.T @ wx, vt.T
    sx, sr = eeg_directions[:len(singular)] @ x, reference_directions[:, :len(singular)].T @ wr @ r
    correlations = []
    for left, right in zip(sx, sr):
        left, right = left - left.mean(), right - right.mean()
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        correlations.append(float(np.dot(left, right) / denominator) if denominator > 0 else 0.)
    correlations = np.asarray(correlations)
    admitted = np.flatnonzero(correlations ** 2 >= policy["cca_r2_threshold"])
    failures, backprojections = {}, {}
    for rank in (1, 2):
        selected = admitted[:rank]
        if not len(selected):
            failures["cca_reference_rank" + str(rank)] = "No admitted calibration canonical directions; scored pass-through"
            failures["cca_eeg_rank" + str(rank)] = failures["cca_reference_rank" + str(rank)]
            continue
        sources = eeg_directions[selected] @ x
        covariance = sources @ sources.T / sources.shape[-1]
        cross = (eeg[ids][:, valid] - mean) @ sources.T / sources.shape[-1]
        backprojections[rank] = np.linalg.solve(covariance + policy["penalty"] * np.eye(len(selected)), cross.T).T
    rest_projection, _, _ = rest_whitened_projection(field.artifact(refs)[:, valid], eeg[ids][:, rest], 2)
    # A structural degeneration contract, independent of scoring targets.
    full = reference_projection(rr, wr, reference_directions, np.arange(len(reference_directions)))
    if not np.allclose(full, np.eye(len(full)), rtol=1e-10, atol=1e-10):
        raise RuntimeError("Full-reference CCA projection is not identity")
    return CCAExperts(ids, field, mean, scale, eeg_directions, rr, wr, reference_directions,
                      correlations, singular, admitted, backprojections, rest_projection, failures)


def estimate_fields(eeg, references, regions, mask, fitted, policy):
    eeg, references = finite_matrix(eeg, "EEG"), finite_matrix(references, "EOG")
    regions, mask = np.asarray(regions), np.asarray(mask, dtype=bool)
    if (regions.shape != (len(eeg),) or mask.shape != regions.shape
            or references.shape != (2, eeg.shape[-1])
            or not np.array_equal(np.flatnonzero(mask), fitted.ids)):
        raise ValueError("CCA scoring montage differs")
    local = eeg[fitted.ids]
    design, valid = fitted.field_model._standardized_design(references)
    field = fitted.field_model.artifact(references)
    candidates = {"ridge_direct": field, "reference_rank2_control": fitted.rest_projection @ field}
    for rank in (1, 2):
        selected = fitted.admitted[:rank]
        if rank not in fitted.eeg_backprojections:
            continue
        projection = reference_projection(fitted.reference_root, fitted.reference_inverse,
                                          fitted.reference_directions, selected)
        reference_field = fitted.field_model.coefficients.T @ projection @ design
        reference_field[:, ~valid] = 0.
        candidates["cca_reference_rank" + str(rank)] = reference_field
        eeg_sources = fitted.eeg_directions[selected] @ ((local - fitted.eeg_mean) / fitted.eeg_scale)
        eeg_field = fitted.eeg_backprojections[rank] @ eeg_sources
        eeg_field[:, ~valid] = 0.
        candidates["cca_eeg_rank" + str(rank)] = eeg_field
    artifacts, details = {}, {}
    for method in METHODS:
        artifacts[method] = np.zeros_like(eeg)
        details[method] = {"regions": {}, "calibration_canonical_cc": fitted.correlations.tolist(),
                           "calibration_regularized_singular_values": fitted.singular_values.tolist(),
                           "calibration_admitted_indices": fitted.admitted.tolist(),
                           "cca_r2_threshold": policy["cca_r2_threshold"], "paired_target_access": False}
        if method not in candidates:
            continue
        candidate = candidates[method]
        if not np.isfinite(candidate).all():
            raise RuntimeError("CCA artifact is nonfinite")
        for region in (1, 2):
            rows = np.flatnonzero(regions[fitted.ids] == region)
            if not len(rows):
                continue
            _, association = balanced_gain(local[rows][:, valid], field[rows][:, valid], policy["gain_cap"])
            gain, _ = balanced_gain(local[rows][:, valid], candidate[rows][:, valid], policy["gain_cap"])
            applied_gain = 1. if method.startswith("cca_eeg") else gain
            triggered = bool(association >= policy["association"])
            if triggered:
                artifacts[method][fitted.ids[rows]] = applied_gain * candidate[rows]
            details[method]["regions"][str(region)] = {"association": association,
                "triggered": triggered, "gain": applied_gain, "source_side": "EEG" if method.startswith("cca_eeg") else "reference"}
    return artifacts, details, dict(fitted.failures)


def save_experts(directory, fitted):
    directory.mkdir(parents=True, exist_ok=True)
    model = fitted.field_model
    arrays = {key: getattr(fitted, key) for key in ("ids", "eeg_mean", "eeg_scale", "eeg_directions",
        "reference_root", "reference_inverse", "reference_directions", "correlations", "singular_values", "admitted", "rest_projection")}
    arrays.update(field_mean=model.reference_mean, field_scale=model.reference_scale,
                  field_active=model.active_columns, field_coefficients=model.coefficients)
    for rank, projection in fitted.eeg_backprojections.items():
        arrays["eeg_backprojection" + str(rank)] = projection
    np.savez_compressed(directory / "fitted_arrays.npz", **arrays)
    atomic_json(directory / "fit_status.json", {"field": model.metadata, "failures": fitted.failures,
        "ranks": list(fitted.eeg_backprojections), "method": "regularized_calibration_CCA_direction_adaptation",
        "author_implementation_parity": False, "target_fit": False})


def load_experts(directory):
    arrays = dict(np.load(directory / "fitted_arrays.npz", allow_pickle=False))
    status = json.loads((directory / "fit_status.json").read_text())
    metadata = status["field"]
    field = FrozenRidgeModel(tuple(metadata["lags"]), metadata["penalty"],
        arrays["field_mean"], arrays["field_scale"], arrays["field_active"], arrays["field_coefficients"],
        metadata["calibration_valid_samples"], metadata["calibration_channel_count"], metadata["predictor_count"])
    return CCAExperts(arrays["ids"], field,
        *(arrays[key] for key in ("eeg_mean", "eeg_scale", "eeg_directions", "reference_root", "reference_inverse",
            "reference_directions", "correlations", "singular_values", "admitted")),
        {int(rank): arrays["eeg_backprojection" + str(rank)] for rank in status["ranks"]},
        arrays["rest_projection"], status["failures"])
