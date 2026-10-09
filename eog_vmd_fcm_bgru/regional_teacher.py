"""Regional VMD plus frozen-spatial teacher primitives.

This module is intentionally EOG-assisted.  It creates teacher residuals for
analysis/distillation, never a runtime dependency of the EEG-only student.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .channel_regions import region_ids, fuse_residuals
from .posterior_experts import ArtifactEstimate
from .reference_guided import (
    ReferenceModeSelector,
    aligned_correlations,
    signed_lagged_reference_projection,
    soft_correlation_gate,
)
from .vmd_rolling import rolling_decompose


def _raw(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] < 32 or not np.isfinite(raw).all():
        raise ValueError("Raw EEG must be finite [channels, samples] with at least 32 samples")
    return raw


def _references(references: np.ndarray, samples: int) -> np.ndarray:
    references = np.asarray(references, dtype=np.float64)
    if references.shape != (2, samples) or not np.isfinite(references).all():
        raise ValueError("Teacher references must be ordered [HEOG, VEOG] and align with EEG")
    return references


@dataclass(frozen=True)
class FrontalVMDConfig:
    modes: int = 5
    alpha: float = 1000.0
    tolerance: float = 1e-6
    max_iterations: int = 2000
    correlation_threshold: float = 0.4
    projection_penalty: float = 0.01
    lags: tuple[int, ...] = (0,)
    strength: float = 1.0
    selector_name: str | None = None

    def __post_init__(self) -> None:
        if int(self.modes) != self.modes or self.modes < 2:
            raise ValueError("VMD K must be an integer >= 2")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("VMD alpha must be positive")
        if not 0 <= self.correlation_threshold < 1 or not 0 <= self.strength <= 1:
            raise ValueError("Invalid frontal gate threshold or strength")
        if not self.lags:
            raise ValueError("At least one lag is required")


@dataclass
class FrontalContext:
    """Signed bilateral frontal observations for posterior methods/diagnostics."""

    common: np.ndarray
    lateral: np.ndarray
    left_count: int
    right_count: int
    frontal_indices: np.ndarray
    availability: dict[str, bool]


def _is_left(name: str) -> bool:
    name = name.strip().upper()
    digits = "".join(char for char in name if char.isdigit())
    return bool(digits) and int(digits[-1]) % 2 == 1


def frontal_context(raw: np.ndarray, names: list[str], valid_mask: np.ndarray | None = None) -> FrontalContext:
    raw = _raw(raw)
    if len(names) != raw.shape[0]:
        raise ValueError("Names must align with raw EEG")
    valid = np.ones(raw.shape[0], dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != (raw.shape[0],):
        raise ValueError("Valid mask must align with raw EEG")
    frontal = np.flatnonzero((region_ids(names) == 0) & valid)
    left = np.asarray([index for index in frontal if _is_left(names[index])], dtype=int)
    right = np.asarray([index for index in frontal if index not in set(left)], dtype=int)
    zeros = np.zeros(raw.shape[1], dtype=np.float64)
    left_signal = raw[left].mean(axis=0) if len(left) else zeros
    right_signal = raw[right].mean(axis=0) if len(right) else zeros
    if len(left) and len(right):
        common = (left_signal + right_signal) / 2
        lateral = right_signal - left_signal
    elif len(frontal):
        common = raw[frontal].mean(axis=0)
        lateral = zeros
    else:
        common, lateral = zeros, zeros
    return FrontalContext(
        common=common,
        lateral=lateral,
        left_count=int(len(left)),
        right_count=int(len(right)),
        frontal_indices=frontal,
        availability={"frontal": bool(len(frontal)), "bilateral": bool(len(left) and len(right))},
    )


def frontal_vmd_estimate(
    raw: np.ndarray,
    references: np.ndarray,
    names: list[str],
    config: FrontalVMDConfig,
    *,
    valid_mask: np.ndarray | None = None,
    selector: ReferenceModeSelector | None = None,
) -> ArtifactEstimate:
    """Estimate a frontal-only artifact residual using canonical RMS VMD.

    VMD and reference ridge coefficients are recomputed in every window.  The
    global config and optional selector are pre-fitted/frozen; VMD mode indices
    are never treated as persistent sources across windows.
    """
    raw = _raw(raw)
    references = _references(references, raw.shape[1])
    if len(names) != raw.shape[0]:
        raise ValueError("Names must align with raw EEG")
    valid = np.ones(raw.shape[0], dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != (raw.shape[0],):
        raise ValueError("Valid mask must align with raw EEG")
    frontal = np.flatnonzero((region_ids(names) == 0) & valid)
    artifact = np.zeros_like(raw)
    confidence = np.zeros(raw.shape[0])
    details: list[dict[str, Any]] = []
    for index in frontal:
        modes, _, diagnostic = rolling_decompose(
            raw[index], config.modes, config.alpha, tolerance=config.tolerance,
            max_iterations=config.max_iterations,
        )
        row = {"channel_index": int(index), "channel_name": names[index], **diagnostic}
        if diagnostic["hit_iteration_limit"]:
            row["accepted"] = False
            row["reason"] = "VMD iteration limit; channel identity passthrough"
            details.append(row)
            continue
        correlations = aligned_correlations(modes, references)
        weights = soft_correlation_gate(correlations, config.correlation_threshold)
        if selector is not None:
            weights = weights * selector.evidence(modes, references)["cluster_weight"]
        projected, projection_details = signed_lagged_reference_projection(
            modes, references, config.lags, config.projection_penalty,
        )
        candidate = config.strength * np.sum(weights[:, None] * projected, axis=0)
        artifact[index] = candidate
        confidence[index] = float(np.clip(np.max(weights, initial=0.0), 0.0, 1.0))
        row.update({
            "accepted": True,
            "signed_heog_veog_correlations": correlations.tolist(),
            "mode_weights": weights.tolist(),
            "projection": projection_details,
        })
        details.append(row)
    output_mask = np.zeros(raw.shape[0], dtype=bool)
    output_mask[frontal] = True
    return ArtifactEstimate(
        artifact=artifact,
        output_mask=output_mask,
        confidence=confidence,
        method="frontal_vmd_eog_projection",
        diagnostics={
            "config": {
                "modes": config.modes, "alpha": config.alpha, "tolerance": config.tolerance,
                "max_iterations": config.max_iterations, "correlation_threshold": config.correlation_threshold,
                "projection_penalty": config.projection_penalty, "lags": list(config.lags),
                "strength": config.strength, "window_adaptive_projection": True,
            },
            "channels": details,
            "context": frontal_context(raw, names, valid).__dict__,
        },
    )


def direct_eog_ridge_estimate(
    raw: np.ndarray,
    references: np.ndarray,
    output_indices: np.ndarray | list[int],
    *,
    lags: tuple[int, ...] = (0,),
    penalty: float = 0.01,
    strength: float = 1.0,
) -> ArtifactEstimate:
    """Matched window-adaptive EOG regression baseline with no VMD."""
    from .reference_guided import signed_lagged_reference_projection
    raw = _raw(raw)
    references = _references(references, raw.shape[1])
    output_indices = np.asarray(output_indices, dtype=int)
    if np.any(output_indices < 0) or np.any(output_indices >= raw.shape[0]):
        raise ValueError("Regression output indices are invalid")
    projected, details = signed_lagged_reference_projection(raw[output_indices], references, lags, penalty)
    artifact = np.zeros_like(raw)
    artifact[output_indices] = strength * projected
    mask = np.zeros(raw.shape[0], dtype=bool)
    mask[output_indices] = True
    confidence = np.zeros(raw.shape[0])
    confidence[output_indices] = np.clip(
        np.sqrt(np.mean(artifact[output_indices] ** 2, axis=1)) /
        (np.sqrt(np.mean(raw[output_indices] ** 2, axis=1)) + np.finfo(float).tiny), 0, 1,
    )
    return ArtifactEstimate(artifact, mask, confidence, "window_adaptive_eog_ridge", {
        "projection": details, "lags": list(lags), "penalty": penalty, "strength": strength,
        "window_adaptive": True,
    })


def compose_disjoint_teacher(raw: np.ndarray, *estimates: ArtifactEstimate) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply disjoint regional expert residuals exactly once.

    Overlapping routes are a configuration error.  The legacy convex blend is
    preserved separately as an ablation; it is not treated as a bug.
    """
    raw = _raw(raw)
    total = np.zeros_like(raw)
    assigned = np.zeros(raw.shape[0], dtype=bool)
    diagnostics = {"methods": [], "unrouted_channels": None}
    for estimate in estimates:
        if estimate.artifact.shape != raw.shape:
            raise ValueError("Every teacher estimate must align with the same raw EEG")
        overlap = assigned & estimate.output_mask
        if overlap.any():
            raise ValueError("Disjoint teacher routes overlap; use explicit legacy convex blending for that ablation")
        total += estimate.artifact
        assigned |= estimate.output_mask
        diagnostics["methods"].append({"method": estimate.method, "output_channels": np.flatnonzero(estimate.output_mask).tolist()})
    diagnostics["unrouted_channels"] = np.flatnonzero(~assigned).tolist()
    return raw - total, total, diagnostics


def legacy_convex_blend(raw, vmd_artifact, spatial_artifact, names, frontal=0.8, posterior=0.2, shared=0.5):
    """Existing valid one-subtraction convex-blend baseline."""
    return fuse_residuals(raw, vmd_artifact, spatial_artifact, names, frontal, posterior, shared)

