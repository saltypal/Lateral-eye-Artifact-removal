"""Inspectable frontal artifact estimators; targets never enter inference.

These are independent paper-inspired adaptations, not author-code parity.
VMD modes remain adaptive vectors rather than fixed physiological bands.
"""
from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.spatial.distance import cdist

from .contracts import canonical_hash
from .data import _region
from .frontal import decompose
from .reference import correlations, project
from .sobi import approximate_entropy, fit_sobi


@dataclass(frozen=True)
class FixedRecipe:
    K: int
    alpha: float = 2000
    fs: int = 200
    tolerance: float = 1e-6
    max_iterations: int = 2000
    initialization: str = "uniform"
    normalization: str = "input_RMS"

    def identity(self):
        return canonical_hash(asdict(self))


def frontal_indices(eeg, channel_names, regions, mask=None):
    """Reject guessed/inconsistent region assignments before any VMD call."""
    eeg = np.asarray(eeg, dtype=float)
    names = np.asarray(channel_names)
    regions = np.asarray(regions)
    valid = np.ones(len(eeg), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if eeg.ndim != 2 or names.shape != (len(eeg),) or regions.shape != names.shape or valid.shape != names.shape:
        raise ValueError("Channel, metadata and mask axes differ")
    if not np.isfinite(eeg).all() or len(set(names.tolist())) != len(names):
        raise ValueError("Nonfinite input or duplicate channel names")
    inferred = np.asarray([_region(str(name)) for name in names])
    if np.any((regions == 0) & (inferred != 0)):
        raise ValueError("Frontal region labels require verified electrode names")
    return np.flatnonzero(valid & (regions == 0) & (inferred == 0))


def decompose_frontal(eeg, channel_names, regions, recipe, config, mask=None):
    """Return only verified frontal vectors, plus explicit failure diagnostics."""
    if (recipe.K, recipe.alpha) not in ((3, 2000), (4, 2000)) or recipe.fs != config["fs"]:
        raise ValueError("Decomposition differs from fixed recipe policy")
    indices = frontal_indices(eeg, channel_names, regions, mask)
    if not len(indices):
        raise ValueError("No verified available frontal electrodes")
    solver = {**config, "classical_grid": {**config["classical_grid"],
        "tolerance": recipe.tolerance, "max_iterations": recipe.max_iterations}}
    result = {}
    for index in indices:
        try:
            modes, residual, info = decompose(eeg[index], recipe.K, recipe.alpha, solver)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            modes = np.zeros((recipe.K, eeg.shape[-1]), dtype=np.float32)
            residual = np.asarray(eeg[index], dtype=np.float32).copy()
            info = {"failure": str(error), "hit_iteration_limit": False}
        result[int(index)] = (modes, residual, info)
    return result


def fuzzy_entropy(values, embedding=1, radius_fraction=.2, exponent=2, block=128):
    """Independent centered-template fuzzy entropy with bounded memory.

    Uses input-SD normalization and exp(-distance**exponent/radius).
    The author repository omits FuzzyEntropy.m; this declared convention is
    an adaptation. No claim of exact author entropy parity is made.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or len(values) <= embedding + 2:
        raise ValueError("Finite sufficiently long entropy source required")
    if embedding < 1 or int(embedding) != embedding or radius_fraction <= 0 or exponent <= 0 or block < 1:
        raise ValueError("Invalid fuzzy entropy parameters")
    scale = float(values.std())
    if scale == 0:
        return 0.
    standardized = (values - values.mean()) / scale
    similarities = []
    for dimension in (embedding, embedding + 1):
        vectors = np.lib.stride_tricks.sliding_window_view(standardized, dimension).copy()
        vectors -= vectors.mean(axis=-1, keepdims=True)
        total = 0.
        for start in range(0, len(vectors), block):
            distances = cdist(vectors[start:start + block], vectors, metric="chebyshev")
            membership = np.exp(-(distances ** exponent) / radius_fraction)
            local = np.arange(len(membership))
            membership[local, start + local] = 0.
            total += float(membership.sum())
        similarities.append(total / (len(vectors) * (len(vectors) - 1)))
    return float(np.log(max(similarities[0], np.finfo(float).tiny)) -
                 np.log(max(similarities[1], np.finfo(float).tiny)))


def reference_intervals(references, calibration_references, fs=200, threshold=3., padding_s=.1):
    """Calibration-standardized HEOG/VEOG event support with tapered edges.

    Fixed 3-SD detection and 100-ms support expansion are project adaptations.
    Whole trials/windows are passed separately, so dilation cannot cross joins.
    """
    references = np.asarray(references, dtype=float)
    calibration = np.asarray(calibration_references, dtype=float)
    if references.ndim != 2 or references.shape[0] != 2 or calibration.ndim != 2 or calibration.shape[0] != 2:
        raise ValueError("Named HEOG/VEOG axes must be [2,time]")
    if calibration.shape[-1] < 2 or not np.isfinite(calibration).all() or not np.isfinite(references).all():
        raise ValueError("Finite unscored reference calibration required")
    mean, scale = calibration.mean(-1, keepdims=True), calibration.std(-1, keepdims=True)
    excursions = np.divide(np.abs(references - mean), scale,
                           out=np.zeros_like(references), where=scale > 0)
    active = excursions >= threshold
    radius = int(round(padding_s * fs))
    expanded = maximum_filter1d(active.astype(float), size=2 * radius + 1, axis=-1, mode="constant")
    if radius:
        # Convolution only softens the expanded boundary, without creating
        # support outside its declared interval.
        kernel = np.hanning(2 * radius + 3)[1:-1]
        kernel /= kernel.sum()
        expanded *= np.stack([np.convolve(row, kernel, mode="same") for row in expanded])
    return expanded


def component_evidence(components, references, max_lag=20):
    signed = correlations(components, references, max_lag)
    return {"signed_association": signed,
            "heog_association": np.abs(signed[:, 0]),
            "veog_association": np.abs(signed[:, 1])}


def estimate_components(components, references, spec, calibration_references,
                        mixing=None, entropy=None, fs=200):
    """Select and repair modes/sources; sum one reconstructed artifact vector."""
    evidence = component_evidence(components, references)
    association = np.max(np.abs(evidence["signed_association"]), axis=1)
    selected = association >= spec["component_threshold"]
    kind = spec["kind"]
    if kind == "apen_whole":
        if entropy is None:
            entropy = np.asarray([approximate_entropy(source) for source in components])
        selected &= np.asarray(entropy) < .4
    if kind in ("whole", "apen_whole"):
        contributions = components * selected[:, None]
    else:
        contributions = project(components, references, spec["lags"], spec["penalty"])[0]
        contributions *= selected[:, None]
        if kind == "interval_projected":
            intervals = reference_intervals(references, calibration_references, fs)
            channel_supported = evidence["signed_association"]
            use_h = np.abs(channel_supported[:, 0]) >= spec["component_threshold"]
            use_v = np.abs(channel_supported[:, 1]) >= spec["component_threshold"]
            support = np.maximum(use_h[:, None] * intervals[0], use_v[:, None] * intervals[1])
            contributions *= support
        elif kind != "projected":
            raise ValueError("Unknown component correction kind")
    reconstructed = contributions if mixing is None else mixing @ contributions
    artifact = reconstructed.sum(axis=0) * spec["strength"]
    return artifact, {"selected": np.flatnonzero(selected).tolist(),
                      "heog_association": evidence["heog_association"].tolist(),
                      "veog_association": evidence["veog_association"].tolist()}


def prepare_sobi(modes, bank):
    delays = ((1, 2, 4, 8, 16, 32), (1, 5, 10, 20, 40, 80))[bank]
    expert = fit_sobi(modes, lags=delays)
    if not expert.diagnostics["converged"]:
        raise RuntimeError("SOBI reached its declared sweep cap")
    return expert


def author_selector_adaptation(sources, reference):
    """2022 author selector structure: fuzzy entropy + ordinary |CC| > .3.

    Caller must declare supplied reference; true simulated artifacts belong
    only in a separately labelled oracle experiment, never realistic inference.
    """
    entropy = np.asarray([fuzzy_entropy(source) for source in sources])
    association = np.abs(correlations(sources, np.atleast_2d(reference), 0)[:, 0])
    selected = entropy + association > .3
    return selected, {"fuzzy_entropy": entropy.tolist(), "absolute_cc": association.tolist(),
                      "selection_threshold": .3, "exact_author_parity": False}


class FrontalCorrector:
    """Whole-montage research interface; nonfrontal rows remain exactly original.

    Inputs are EEG [C,T], named HEOG/VEOG [2,T], unscored reference calibration
    and verified channel metadata. No paired clean target is accepted.
    A fitted SVM is required only for the separately labelled SVM adaptation.
    """
    def __init__(self, spec, config, detector=None):
        self.spec = dict(spec)
        self.config = config
        self.detector = detector
        self.last_diagnostics = []

    def denoise(self, eeg, references, calibration_references, channel_names, regions,
                mask=None, mode_bundles=None, calibration_eeg=None):
        original = np.asarray(eeg, dtype=float)
        references = np.asarray(references, dtype=float)
        indices = frontal_indices(original, channel_names, regions, mask)
        if references.shape != (2, original.shape[-1]) or not np.isfinite(references).all():
            raise ValueError("Aligned finite named HEOG/VEOG references required")
        family = self.spec["family"]
        if family == "svm_sobi_apen_fixed_adaptation" and self.detector is None:
            raise ValueError("SVM adaptation needs its source-excluded fitted detector")
        if family == "svm_sobi_apen_fixed_adaptation":
            calibration_eeg = np.asarray(calibration_eeg, dtype=float)
            if (calibration_eeg.ndim != 2 or calibration_eeg.shape[0] != len(original) or
                    not np.isfinite(calibration_eeg).all() or
                    np.any(calibration_eeg[indices].std(axis=-1) <= 0)):
                raise ValueError("SVM adaptation needs unscored same-unit EEG calibration")
        artifact = np.zeros_like(original)
        self.last_diagnostics = []
        if family == "identity" or not len(indices):
            return original.copy(), artifact, self.last_diagnostics
        bundles = None
        if family != "direct":
            recipe = FixedRecipe(self.spec["K"], self.spec["alpha"], self.config["fs"])
            bundles = (decompose_frontal(original, channel_names, regions, recipe, self.config, mask)
                       if mode_bundles is None else mode_bundles)
            if set(bundles) != set(indices.tolist()):
                raise ValueError("Mode cache must contain precisely the verified frontal rows")
            for index, (modes, residual, info) in bundles.items():
                if (np.asarray(modes).shape != (recipe.K, original.shape[-1]) or
                        np.asarray(residual).shape != (original.shape[-1],) or
                        not np.isfinite(modes).all() or not np.isfinite(residual).all() or
                        not np.allclose(np.asarray(modes).sum(0) + residual, original[index],
                                        rtol=1e-5, atol=1e-11)):
                    raise ValueError("Mode cache shape, finiteness or reconstruction differs")
        for index in indices:
            association = float(np.max(np.abs(correlations(original[index], references, 20))))
            if family == "svm_sobi_apen_fixed_adaptation":
                from .frontal_replication import detector_features
                trigger = bool(self.detector.predict(detector_features(original[index], self.config["fs"],
                    calibration_eeg[index].std())[None])[0])
            else:
                trigger = association >= self.spec["trigger"]
            detail = {"channel": str(channel_names[index]), "triggered": trigger,
                      "scope": "verified_frontal_only", "inference_target_access": False}
            try:
                if trigger and family == "direct":
                    artifact[index] = self.spec["strength"] * project(original[index], references,
                        self.spec["lags"], self.spec["penalty"])[0][0]
                elif trigger:
                    modes, residual, info = bundles[int(index)]
                    detail.update(info)
                    if info.get("failure") or info.get("hit_iteration_limit"):
                        raise RuntimeError(info.get("failure") or "VMD iteration limit")
                    components, mixing = modes, None
                    if "sobi" in family or family == "author2022_selector_adaptation":
                        expert = prepare_sobi(modes, self.spec.get("sobi_bank", 0))
                        components, mixing = expert.sources, expert.mixing
                    if family == "author2022_selector_adaptation":
                        selected_h, evidence_h = author_selector_adaptation(components, references[0])
                        selected_v, evidence_v = author_selector_adaptation(components, references[1])
                        selected = selected_h | selected_v
                        artifact[index] = self.spec["strength"] * (mixing[:, selected] @ components[selected]).sum(0)
                        detail.update(HEOG=evidence_h, VEOG=evidence_v)
                    else:
                        artifact[index], selection = estimate_components(components, references,
                            self.spec, calibration_references, mixing=mixing, fs=self.config["fs"])
                        detail.update(selection)
                if not np.isfinite(artifact[index]).all():
                    raise FloatingPointError("Nonfinite frontal artifact estimate")
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                artifact[index] = 0
                detail.update(failure=str(error), status="scored_passthrough")
            self.last_diagnostics.append(detail)
        return original - artifact, artifact, self.last_diagnostics
