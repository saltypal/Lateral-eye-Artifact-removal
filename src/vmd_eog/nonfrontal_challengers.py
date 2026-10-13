"""Target-free component correction for exploratory posterior/central experiments.

These are declared paper-inspired adaptations, not author reproductions. Fit on
unscored calibration, estimate against original EEG, and compose disjoint output
regions once. Numerical execution belongs exclusively to the Kaggle runner.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from sklearn.covariance import LedoitWolf

from .nonfrontal_ridge import fit_frozen_ridge
from .posterior import boundary_valid, fit_ica
from .reference import lag_matrix


METHODS = ("ica_rejection", "ica_component_ridge", "wavelet_ica",
           "selective_wavelet_ica", "reference_rank1", "reference_rank2")


def finite_matrix(values, label):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected finite nonempty [channel,time] " + label)
    return values


def balanced_gain(observation, field, cap=8.):
    """One shared region gain; no target or generation amplitude argument."""
    observation = finite_matrix(observation, "observation")
    field = finite_matrix(field, "field")
    if observation.shape != field.shape or not np.isfinite(cap) or cap <= 0:
        raise ValueError("Gain axes or bound differ")
    x = observation - observation.mean(-1, keepdims=True)
    f = field - field.mean(-1, keepdims=True)
    denominator = float(np.sum(f * f))
    input_energy = float(np.sum(x * x))
    if denominator <= 0 or input_energy <= 0:
        return 0., 0.
    cross = float(np.sum(x * f))
    gain = float(np.clip(cross / denominator, 0., cap))
    association = float(np.clip(cross / np.sqrt(denominator * input_energy), -1., 1.))
    return gain, association


def rest_whitened_projection(field, rest, rank):
    """Truncate a reference field in rest-whitened coordinates, not ordinary PCA."""
    field, rest = finite_matrix(field, "field"), finite_matrix(rest, "rest")
    if field.shape[0] != rest.shape[0] or not isinstance(rank, int) or rank < 1:
        raise ValueError("Rank projection axes/rank differ")
    if rest.shape[-1] < max(128, 2 * len(rest)):
        raise ValueError("Insufficient resting calibration")
    covariance = LedoitWolf().fit(rest.T).covariance_
    eigenvalues, vectors = eigh(covariance)
    floor = max(float(eigenvalues.max()) * 1e-8, 1e-20)
    roots = np.sqrt(np.maximum(eigenvalues, floor))
    whitening = (vectors / roots[None, :]).T
    inverse = vectors * roots[None, :]
    centered = field - field.mean(-1, keepdims=True)
    white = whitening @ centered
    spectrum, basis = eigh(white @ white.T / white.shape[-1])
    order = np.argsort(spectrum)[::-1]
    retained = min(rank, len(field))
    selected = basis[:, order[:retained]]
    projection = inverse @ selected @ selected.T @ whitening
    return projection, spectrum[order], retained


def large_wavelet_contribution(values, threshold, *, selective=False,
                               wavelet="sym4", max_level=6, dilation=50, taper=10):
    """Extract large coefficients; small coefficients remain in the cleaned IC.

    Threshold is fitted from calibration rest, not from a scoring clean target.
    Symmetric wavelet extension is local to one scoring window/trial. Temporal
    selection is an explicit adaptation, not the paper's original detector.
    """
    import pywt
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 1 or not np.isfinite(values).all() or len(values) < 16
            or not np.isfinite(threshold) or threshold <= 0
            or max_level < 1 or dilation < 0 or taper < 0):
        raise ValueError("Invalid wavelet vector/threshold/settings")
    basis = pywt.Wavelet(wavelet)
    level = min(max_level, pywt.dwt_max_level(len(values), basis.dec_len))
    if level < 1:
        raise ValueError("Window too short for wavelet decomposition")
    coefficients = pywt.wavedec(values, basis, mode="symmetric", level=level)
    large = [np.sign(part) * np.maximum(np.abs(part) - threshold, 0.)
             for part in coefficients]
    artifact = pywt.waverec(large, basis, mode="symmetric")[:len(values)]
    rebuilt = pywt.waverec(coefficients, basis, mode="symmetric")[:len(values)]
    # Symlet filter coefficients have finite precision. A pointwise relative
    # test becomes an absolute-zero test at silent samples; use the declared
    # 1e-10 relative tolerance on the vector reconstruction instead.
    reconstruction_error = float(np.linalg.norm(rebuilt - values))
    scale = float(np.linalg.norm(values))
    relative_error = reconstruction_error / scale if scale > 0 else reconstruction_error
    if not np.isfinite(relative_error) or relative_error > 1e-10:
        raise RuntimeError("Wavelet reconstruction contract failed")
    support = np.ones(len(values), dtype=np.float64)
    if selective:
        support = (np.abs(values) > threshold).astype(np.float64)
        if dilation:
            support = maximum_filter1d(support, 2 * dilation + 1, mode="constant", cval=0.)
        if taper:
            support = uniform_filter1d(support, 2 * taper + 1, mode="constant", cval=0.)
        artifact *= support
    return artifact, {"level": level, "threshold": float(threshold),
                      "support_fraction": float(np.mean(support)),
                      "reconstruction_relative_error": relative_error,
                      "reconstruction_max_abs_error": float(np.max(np.abs(rebuilt - values)))}


def replace_regions(original, frozen, artifact, regions, mask, allowed_regions):
    """Replace only requested nonfrontal outputs, relative to original input."""
    original = finite_matrix(original, "EEG")
    frozen, artifact = np.asarray(frozen), np.asarray(artifact)
    regions, mask = np.asarray(regions), np.asarray(mask, dtype=bool)
    if (frozen.shape != original.shape or artifact.shape != original.shape
            or regions.shape != (len(original),) or mask.shape != regions.shape
            or not np.isfinite(frozen).all() or not np.isfinite(artifact).all()
            or not set(allowed_regions).issubset({1, 2}) or not allowed_regions):
        raise ValueError("Invalid regional composition contract")
    ids = np.flatnonzero(mask & np.isin(regions, allowed_regions))
    result = frozen.copy()
    result[ids] = original[ids] - artifact[ids]
    outside = np.setdiff1d(np.arange(len(original)), ids)
    if not np.array_equal(result[outside], frozen[outside]):
        raise RuntimeError("Regional composition changed frozen rows")
    return result


@dataclass
class NonfrontalExperts:
    ids: np.ndarray
    field_model: object
    ica: object
    component_model: object
    rest_center: np.ndarray
    rest_scale: np.ndarray
    projections: dict
    failures: dict


def fit_experts(calibration, policy):
    """Fit only EEG/EOG calibration, labels, boundaries and verified masks."""
    eeg = finite_matrix(calibration["eeg"], "calibration EEG")
    refs = finite_matrix(calibration["references"], "calibration EOG")
    mask = np.asarray(calibration["mask"], dtype=bool)
    types = np.asarray(calibration["trial_types"])
    if mask.shape != (len(eeg),) or refs.shape != (2, eeg.shape[-1]) or types.shape != (eeg.shape[-1],):
        raise ValueError("Calibration masks/reference/types mismatch")
    ids = np.flatnonzero(mask)
    rest = types == 1
    if not len(ids) or rest.sum() < max(128, 2 * len(ids)):
        raise ValueError("Insufficient verified montage/rest calibration")
    boundaries = tuple(int(x) for x in calibration["boundaries"])
    lags = tuple(policy["lags"])
    _, valid = lag_matrix(refs, lags)
    valid &= boundary_valid(eeg.shape[-1], boundaries, lags)
    rest &= valid
    if rest.sum() < max(128, 2 * len(ids)):
        raise ValueError("Insufficient rest after trial/lag exclusions")
    model = fit_frozen_ridge(eeg[ids], refs, boundaries, lags, policy["penalty"])
    failures, projections = {}, {}
    fields = model.artifact(refs)
    for rank in (1, 2):
        try:
            projections[rank] = rest_whitened_projection(fields[:, valid], eeg[ids][:, rest], rank)
        except (ValueError, np.linalg.LinAlgError) as error:
            failures["reference_rank" + str(rank)] = str(error)
    expert = component_model = center = scale = None
    try:
        expert = fit_ica(eeg[ids], refs, threshold=policy["association"],
                         seed=42, boundaries=boundaries)
        sources = expert.unmixing @ (eeg[ids] - expert.mean[:, None])
        center = np.median(sources[:, rest], axis=-1)
        scale = np.median(np.abs(sources[:, rest] - center[:, None]), axis=-1) / .6744897501960817
        component_model = fit_frozen_ridge(sources, refs, boundaries, lags, policy["penalty"])
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        for method in METHODS[:4]:
            failures[method] = type(error).__name__ + ": " + str(error)
        expert = None
    return NonfrontalExperts(ids, model, expert, component_model, center, scale, projections, failures)


def estimate_fields(eeg, references, regions, mask, fitted, policy):
    """Return original-input artifact fields without accepting paired targets."""
    eeg, references = finite_matrix(eeg, "EEG"), finite_matrix(references, "EOG")
    regions, mask = np.asarray(regions), np.asarray(mask, dtype=bool)
    if (regions.shape != (len(eeg),) or mask.shape != regions.shape
            or references.shape != (2, eeg.shape[-1])
            or not np.array_equal(np.flatnonzero(mask), fitted.ids)):
        raise ValueError("Scoring montage/reference differs from calibration")
    local = eeg[fitted.ids]
    field = fitted.field_model.artifact(references)
    _, valid = lag_matrix(references, fitted.field_model.lags)
    candidates = {}
    failures = dict(fitted.failures)
    decisions = {}
    for rank, (projection, _, _) in fitted.projections.items():
        candidates["reference_rank" + str(rank)] = projection @ field
    if fitted.ica is not None:
        expert = fitted.ica
        sources = expert.unmixing @ (local - expert.mean[:, None])
        selected = expert.selected
        candidates["ica_rejection"] = expert.mixing[:, selected] @ sources[selected]
        component_field = fitted.component_model.artifact(references)
        candidates["ica_component_ridge"] = expert.mixing[:, selected] @ component_field[selected]
        for method in ("wavelet_ica", "selective_wavelet_ica"):
            artifact_sources = np.zeros_like(sources)
            wavelet_details = []
            try:
                for component in selected:
                    threshold = policy["mad_multiplier"] * fitted.rest_scale[component]
                    if threshold <= 0:
                        raise ValueError("Nonpositive calibrated source scale")
                    artifact_sources[component], detail = large_wavelet_contribution(
                        sources[component] - fitted.rest_center[component], threshold,
                        selective=method.startswith("selective"),
                        wavelet=policy["wavelet"], max_level=policy["wavelet_level"],
                        dilation=policy["dilation"], taper=policy["taper"])
                    wavelet_details.append({"component": int(component), **detail})
                candidates[method] = expert.mixing @ artifact_sources
                decisions[method] = {"wavelet": wavelet_details}
            except (ValueError, RuntimeError) as error:
                failures[method] = type(error).__name__ + ": " + str(error)
    artifacts = {method: np.zeros_like(eeg) for method in METHODS}
    for method in METHODS:
        decisions.setdefault(method, {})
        decisions[method]["regions"] = {}
        if method not in candidates:
            failures.setdefault(method, "Unavailable fitted mechanism; scored pass-through")
            continue
        candidate = candidates[method]
        if candidate.shape != local.shape or not np.isfinite(candidate).all():
            raise RuntimeError("Candidate artifact shape/finiteness failure")
        for region in (1, 2):
            rows = np.flatnonzero(regions[fitted.ids] == region)
            if not len(rows):
                continue
            _, association = balanced_gain(local[rows][:, valid], field[rows][:, valid], policy["gain_cap"])
            gain, _ = balanced_gain(local[rows][:, valid], candidate[rows][:, valid], policy["gain_cap"])
            triggered = bool(association >= policy["association"])
            # Rejection/wavelet estimators already use scoring source amplitude;
            # multiplying them by another fitted gain changes their mechanism.
            applied_gain = gain if method in ("ica_component_ridge", "reference_rank1", "reference_rank2") else 1.
            if triggered:
                artifacts[method][fitted.ids[rows]] = applied_gain * candidate[rows]
            decisions[method]["regions"][str(region)] = {
                "association": association, "triggered": triggered,
                "gain": float(applied_gain), "paired_target_access": False}
    return artifacts, decisions, failures
