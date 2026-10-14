"""Homogeneous donor-calibration forward field for explicitly synthetic mixtures.

This producer is separate from denoisers. Reference excursions, not raw offsets,
are its coordinates. Physical donor units unknown in metadata remain unknown.
"""
from dataclasses import dataclass

import numpy as np

from .posterior import boundary_valid
from .reference import lag_matrix


def valid_driver_design(references, lags, boundaries):
    references = np.asarray(references, dtype=np.float64)
    if references.ndim != 2 or references.shape[0] != 2 or not np.isfinite(references).all():
        raise ValueError("Expected finite HEOG/VEOG driver coordinates")
    if not lags or any(not isinstance(lag, (int, np.integer)) for lag in lags) or len(set(lags)) != len(lags):
        raise ValueError("Lag bank requires distinct integer sample offsets")
    cuts = np.asarray([] if boundaries is None else boundaries)
    if (cuts.ndim != 1 or not np.isfinite(cuts).all() or np.any(cuts != np.floor(cuts))
            or np.any(cuts <= 0) or np.any(cuts >= references.shape[-1]) or np.any(np.diff(cuts) <= 0)):
        raise ValueError("Invalid original trial boundaries")
    design, valid = lag_matrix(references, lags)
    valid &= boundary_valid(references.shape[-1], cuts.astype(int), lags)
    return design, valid


@dataclass(frozen=True)
class HomogeneousField:
    lags: tuple
    baseline: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    calibration_valid_samples: int

    def excursions(self, raw_references):
        values = np.asarray(raw_references, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != 2 or not np.isfinite(values).all():
            raise ValueError("Raw references do not match frozen baseline")
        return values - self.baseline[:, None]

    def forward(self, excursions, boundaries=None):
        """F(aR)=aF(R); no intercept or newly fitted mean enters this operation."""
        design, valid = valid_driver_design(excursions, self.lags, boundaries)
        field = self.coefficients @ (design / self.scales[:, None])
        field[:, ~valid] = 0.
        if not np.isfinite(field).all():
            raise ValueError("Nonfinite homogeneous field")
        return field


def fit_donor_field(calibration_eeg, calibration_references, boundaries=None,
                    lags=(-10, 0, 10), penalty=.01):
    """Only original donor calibration is accepted; scoring targets are absent."""
    eeg = np.asarray(calibration_eeg, dtype=np.float64)
    refs = np.asarray(calibration_references, dtype=np.float64)
    if (eeg.ndim != 2 or refs.ndim != 2 or refs.shape[0] != 2
            or eeg.shape[-1] != refs.shape[-1] or not np.isfinite(eeg).all()):
        raise ValueError("Invalid original donor calibration EEG")
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("Ridge penalty must be positive")
    _, valid = valid_driver_design(refs, lags, boundaries)
    if valid.sum() <= 2 * len(lags) + 1:
        raise ValueError("Insufficient valid donor calibration")
    baseline = refs[:, valid].mean(axis=1)
    design, valid = valid_driver_design(refs - baseline[:, None], lags, boundaries)
    scales = design[:, valid].std(axis=1)
    if not np.any(scales > 1e-12):
        raise ValueError("No active original ocular driver")
    scales = np.where(scales > 1e-12, scales, 1.)
    standardized = design / scales[:, None]
    covariance = standardized[:, valid] @ standardized[:, valid].T / valid.sum()
    centered_eeg = eeg[:, valid] - eeg[:, valid].mean(axis=1, keepdims=True)
    cross = standardized[:, valid] @ centered_eeg.T / valid.sum()
    coefficients = np.linalg.solve(covariance + penalty * np.eye(len(scales)), cross).T
    return HomogeneousField(tuple(lags), baseline, scales, coefficients, int(valid.sum()))


def generated_pair(target, excursions, field_model, input_snr_db, boundaries=None):
    """Producer-only paired truth sets amplitude; no denoiser may call this API."""
    target = np.asarray(target, dtype=np.float64)
    excursions = np.asarray(excursions, dtype=np.float64)
    field = field_model.forward(excursions, boundaries)
    if target.shape != field.shape or not np.isfinite(target).all():
        raise ValueError("Target/forward-field shape differs")
    if input_snr_db is None:
        amplitude = 0.
    else:
        if not np.isfinite(input_snr_db):
            raise ValueError("Input SNR must be finite")
        field_norm, target_norm = np.linalg.norm(field), np.linalg.norm(target)
        if field_norm <= 1e-30 or target_norm <= 1e-30:
            raise ValueError("Zero energy cannot define a controlled mixture")
        amplitude = target_norm / (10 ** (input_snr_db / 20) * field_norm)
    references = amplitude * excursions
    artifact = amplitude * field
    return {"eeg": target + artifact, "paired_reference": target.copy(),
            "references": references, "true_artifact": artifact,
            "generation_amplitude": float(amplitude)}
