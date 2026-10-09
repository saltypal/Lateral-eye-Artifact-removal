"""Frozen posterior spatial experts for the regional EOG teacher.

The public interface deliberately separates ``fit`` from ``estimate``.  A
spatial transform is fitted on declared, unscored calibration data and then
used unchanged on scoring windows.  It is therefore different from the
window-adaptive, EOG-assisted frontal VMD projection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .spatial_expert import ICAExpert


def _finite_matrix(values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError(f"{name} must have shape [channels, samples]")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _covariance(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    return centered @ centered.T / max(values.shape[1] - 1, 1)


def delay_embed(values: np.ndarray, order: int = 0) -> np.ndarray:
    """Symmetrically delay-embed a continuous [channel, sample] matrix.

    Order zero is a spatial GEVD-MWF.  A positive order makes the method
    offline/non-causal because it uses future samples.  Edge values are
    replicated rather than wrapped, so no event is moved across an epoch
    boundary.
    """
    values = _finite_matrix(values, "EEG")
    if int(order) != order or order < 0:
        raise ValueError("Delay order must be a nonnegative integer")
    order = int(order)
    if order == 0:
        return values.copy()
    padded = np.pad(values, ((0, 0), (order, order)), mode="edge")
    return np.concatenate(
        [padded[:, order + lag:order + lag + values.shape[1]] for lag in range(-order, order + 1)],
        axis=0,
    )


@dataclass
class ArtifactEstimate:
    """An aligned artifact estimate against one declared raw EEG input."""

    artifact: np.ndarray
    output_mask: np.ndarray
    confidence: np.ndarray
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.artifact = _finite_matrix(self.artifact, "artifact")
        self.output_mask = np.asarray(self.output_mask, dtype=bool)
        self.confidence = np.asarray(self.confidence, dtype=np.float64)
        if self.output_mask.shape != (self.artifact.shape[0],):
            raise ValueError("Artifact output mask must align with channels")
        if self.confidence.shape not in {(self.artifact.shape[0],), self.artifact.shape}:
            raise ValueError("Artifact confidence must be [channels] or [channels, samples]")
        if not np.isfinite(self.confidence).all():
            raise ValueError("Artifact confidence must be finite")
        if np.any(self.artifact[~self.output_mask]):
            raise ValueError("Artifact values outside the declared output mask are forbidden")


@dataclass
class GEVDMWFState:
    """Frozen GEVD-MWF transform fitted on calibration only.

    In the generalized eigenbasis of total covariance ``R_x`` versus quiet
    covariance ``R_n``, the clean Wiener gain is ``1/lambda`` and the artifact
    gain is ``1 - 1/lambda``.  We retain only eigen-directions above the
    declared excess threshold.  This is a rank-regularized implementation of
    the GEVD-MWF principle; it makes the selected rank and delay embedding
    explicit rather than deleting arbitrary channels/components.
    """

    basis: np.ndarray
    artifact_gain: np.ndarray
    output_indices: np.ndarray
    channel_count: int
    delay_order: int
    eigenvalues: np.ndarray
    quiet_samples: int
    calibration_samples: int
    regularization: float
    excess_threshold: float

    def estimate(self, raw: np.ndarray) -> ArtifactEstimate:
        raw = _finite_matrix(raw, "raw EEG")
        if raw.shape[0] != self.channel_count:
            raise ValueError("GEVD-MWF raw channel count differs from calibration")
        embedded = delay_embed(raw, self.delay_order)
        coordinates = self.basis.T @ embedded
        selected = self.artifact_gain[:, None] * coordinates
        # x = inv(B.T) z because B.T R_n B = I in the generalized basis.
        embedded_artifact = np.linalg.solve(self.basis.T, selected)
        offset = self.delay_order * self.channel_count
        artifact = embedded_artifact[offset:offset + self.channel_count]
        output_mask = np.zeros(self.channel_count, dtype=bool)
        output_mask[self.output_indices] = True
        artifact[~output_mask] = 0.0
        scale = np.sqrt(np.mean(artifact ** 2, axis=1))
        denom = np.sqrt(np.mean(raw ** 2, axis=1)) + np.finfo(float).tiny
        confidence = np.clip(scale / denom, 0.0, 1.0)
        return ArtifactEstimate(
            artifact=artifact,
            output_mask=output_mask,
            confidence=confidence,
            method="gevd_mwf",
            diagnostics={
                "delay_order": self.delay_order,
                "selected_rank": int(np.count_nonzero(self.artifact_gain)),
                "eigenvalues": self.eigenvalues.tolist(),
                "quiet_samples": self.quiet_samples,
                "calibration_samples": self.calibration_samples,
                "excess_threshold": self.excess_threshold,
                "regularization": self.regularization,
                "frozen_calibration": True,
            },
        )


class GEVDMWFExpert:
    """Calibration-frozen, rank-regularized generalized-eigen MWF."""

    @staticmethod
    def fit(
        calibration_eeg: np.ndarray,
        quiet_mask: np.ndarray,
        output_indices: np.ndarray | list[int],
        *,
        delay_order: int = 0,
        shrinkage: float = 1e-3,
        excess_threshold: float = 1.05,
        max_rank: int | None = None,
    ) -> GEVDMWFState:
        calibration_eeg = _finite_matrix(calibration_eeg, "calibration EEG")
        quiet_mask = np.asarray(quiet_mask, dtype=bool)
        if quiet_mask.shape != (calibration_eeg.shape[1],):
            raise ValueError("Quiet mask must align with calibration samples")
        if quiet_mask.sum() < max(16, calibration_eeg.shape[0] * 2):
            raise ValueError("GEVD-MWF requires enough declared quiet calibration samples")
        if not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("GEVD-MWF shrinkage must be finite and nonnegative")
        if not np.isfinite(excess_threshold) or excess_threshold <= 1:
            raise ValueError("GEVD-MWF excess threshold must exceed one")
        output_indices = np.asarray(output_indices, dtype=int)
        if output_indices.ndim != 1 or not len(output_indices) or np.any(output_indices < 0) or np.any(output_indices >= calibration_eeg.shape[0]):
            raise ValueError("GEVD-MWF output indices are invalid")
        if len(np.unique(output_indices)) != len(output_indices):
            raise ValueError("GEVD-MWF output indices must be unique")

        embedded = delay_embed(calibration_eeg, delay_order)
        quiet = embedded[:, quiet_mask]
        total_covariance = _covariance(embedded)
        quiet_covariance = _covariance(quiet)
        dimension = total_covariance.shape[0]
        scale = np.trace(quiet_covariance) / max(dimension, 1)
        regularization = float(shrinkage * max(scale, np.finfo(float).tiny))
        quiet_covariance = quiet_covariance + regularization * np.eye(dimension)

        # Whitening turns the GEVD R_x v=lambda R_n v into a stable Hermitian
        # eigenproblem.  It avoids explicitly inverting an ill-conditioned R_n.
        values, vectors = np.linalg.eigh(quiet_covariance)
        floor = max(float(values.max(initial=0.0)) * 1e-12, np.finfo(float).tiny)
        inverse_sqrt = vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, floor))) @ vectors.T
        whitened_total = inverse_sqrt @ total_covariance @ inverse_sqrt
        eigenvalues, eigenvectors = np.linalg.eigh((whitened_total + whitened_total.T) / 2)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        basis = inverse_sqrt @ eigenvectors[:, order]
        gains = np.clip(1.0 - 1.0 / np.maximum(eigenvalues, 1.0), 0.0, 1.0)
        selected = eigenvalues > excess_threshold
        if max_rank is not None:
            if int(max_rank) != max_rank or max_rank < 1:
                raise ValueError("GEVD-MWF max rank must be a positive integer")
            rank_order = np.flatnonzero(selected)
            selected[rank_order[int(max_rank):]] = False
        gains[~selected] = 0.0
        return GEVDMWFState(
            basis=basis,
            artifact_gain=gains,
            output_indices=output_indices,
            channel_count=calibration_eeg.shape[0],
            delay_order=int(delay_order),
            eigenvalues=eigenvalues,
            quiet_samples=int(quiet_mask.sum()),
            calibration_samples=int(calibration_eeg.shape[1]),
            regularization=regularization,
            excess_threshold=float(excess_threshold),
        )


@dataclass
class RoutedICAState:
    """Frozen ICA calibration and declared posterior output routing."""

    expert: ICAExpert
    fit_indices: np.ndarray
    output_indices: np.ndarray
    threshold: float
    strength: float

    def estimate(self, raw: np.ndarray) -> ArtifactEstimate:
        raw = _finite_matrix(raw, "raw EEG")
        if raw.shape[0] <= int(self.fit_indices.max(initial=-1)):
            raise ValueError("ICA raw channel count differs from calibration")
        estimated = self.expert.residual(raw[self.fit_indices], threshold=self.threshold)
        artifact = np.zeros_like(raw)
        position = {int(index): offset for offset, index in enumerate(self.fit_indices)}
        artifact[self.output_indices] = self.strength * estimated[[position[int(index)] for index in self.output_indices]]
        mask = np.zeros(raw.shape[0], dtype=bool)
        mask[self.output_indices] = True
        confidence = np.zeros(raw.shape[0])
        confidence[self.output_indices] = np.clip(np.sqrt(np.mean(artifact[self.output_indices] ** 2, axis=1)) /
                                                  (np.sqrt(np.mean(raw[self.output_indices] ** 2, axis=1)) + np.finfo(float).tiny), 0, 1)
        return ArtifactEstimate(artifact, mask, confidence, "routed_ica", {
            "rank": self.expert.rank, "threshold": self.threshold,
            "strength": self.strength, "frozen_calibration": True,
        })


class RoutedICAExpert:
    """Adapter around the existing rank-aware ICA implementation."""

    @staticmethod
    def fit(calibration_eeg, calibration_references, fit_indices, output_indices, *, method="picard", threshold=0.3,
            strength=1.0, calibration_highpass=None) -> RoutedICAState:
        calibration_eeg = _finite_matrix(calibration_eeg, "calibration EEG")
        calibration_references = _finite_matrix(calibration_references, "calibration references")
        if calibration_eeg.shape[1] != calibration_references.shape[1]:
            raise ValueError("ICA calibration EEG and references must align")
        fit_indices = np.asarray(fit_indices, dtype=int)
        output_indices = np.asarray(output_indices, dtype=int)
        if not set(output_indices).issubset(set(fit_indices)):
            raise ValueError("ICA output indices must be included in fit indices")
        fitting = None if calibration_highpass is None else _finite_matrix(calibration_highpass, "ICA highpass calibration")[fit_indices]
        expert = ICAExpert.fit(calibration_eeg[fit_indices], calibration_references, method,
                               calibration_highpass=fitting)
        return RoutedICAState(expert, fit_indices, output_indices, float(threshold), float(strength))


class SGEYESUBExpert:
    """Explicitly unavailable until the verified author implementation is pinned.

    A superficial approximation would not be SGEYESUB.  The stage records this
    as unsupported rather than silently substituting ICA or regression.
    """

    availability_reason = "SGEYESUB author implementation is not pinned in this repository"

    @classmethod
    def fit(cls, *args, **kwargs):
        raise RuntimeError(f"SGEYESUB unavailable: {cls.availability_reason}")

