"""Frozen donor-calibration ridge fields for nonfrontal EEG channels.

The model fits only calibration EEG and aligned HEOG/VEOG predictors. Scoring
EEG and paired targets are deliberately absent from every interface. Callers
should fit/apply/gate one scored trial or window at a time so lag edges,
dilation, and tapering cannot cross trial joins.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from .frontal_event_support import AxisAmplitudeSupport
from .posterior import boundary_valid
from .reference import lag_matrix


MIN_VALID_SAMPLES = 128


def _validate_boundaries(boundaries, length):
    values = np.asarray(boundaries)
    if values.size == 0:
        return ()
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Calibration boundaries must be integer trial starts")
    if np.any(values <= 0) or np.any(values >= length) or np.any(np.diff(values) <= 0):
        raise ValueError("Invalid calibration trial boundaries")
    return tuple(int(value) for value in values)


def _validate_lags(lags):
    values = tuple(lags)
    if (not values or any(not isinstance(value, (int, np.integer)) for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("Lags must be a nonempty sequence of unique integers")
    if 0 not in values:
        raise ValueError("The frozen ridge lag bank must include zero")
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class FrozenRidgeModel:
    """Calibration-only standardized lag ridge with frozen coefficients."""

    lags: tuple
    penalty: float
    reference_mean: np.ndarray
    reference_scale: np.ndarray
    active_columns: np.ndarray
    coefficients: np.ndarray
    calibration_valid_samples: int
    calibration_channel_count: int
    predictor_count: int
    method: str = "frozen_joint_lagged_calibration_ridge"

    def _standardized_design(self, predictors):
        values = np.asarray(predictors, dtype=np.float64)
        if (values.ndim != 2 or values.shape[0] != self.predictor_count
                or values.shape[-1] <= max(abs(lag) for lag in self.lags)
                or not np.isfinite(values).all()):
            raise ValueError("Expected finite aligned predictors with the calibrated row count")
        design, valid = lag_matrix(values, self.lags)
        standardized = (design - self.reference_mean) / self.reference_scale
        standardized[~self.active_columns] = 0.
        return standardized, valid

    def artifact(self, predictors):
        """Apply fixed calibration coefficients; unavailable lag edges are zero."""
        design, valid = self._standardized_design(predictors)
        artifact = self.coefficients.T @ design
        artifact[:, ~valid] = 0.
        return artifact

    def axis_artifacts(self, references):
        """Return HEOG and VEOG contributions from the joint lag-major fit."""
        if self.predictor_count != 2:
            raise ValueError("Axis contributions require exactly HEOG and VEOG rows")
        design, valid = self._standardized_design(references)
        artifacts = np.stack([
            self.coefficients[axis::2].T @ design[axis::2]
            for axis in range(2)
        ])
        artifacts[:, :, ~valid] = 0.
        return artifacts

    @property
    def metadata(self):
        return {
            "method": self.method,
            "lags": list(self.lags),
            "penalty": float(self.penalty),
            "calibration_valid_samples": int(self.calibration_valid_samples),
            "calibration_channel_count": int(self.calibration_channel_count),
            "predictor_count": int(self.predictor_count),
            "active_lag_columns": int(np.count_nonzero(self.active_columns)),
            "scoring_eeg_access": False,
            "paired_target_access": False,
        }


def fit_frozen_ridge(calibration_eeg, calibration_predictors, boundaries=(),
                     lags=(-10, 0, 10), penalty=.01):
    """Fit one joint lagged field using calibration EEG and predictors only.

    EEG channel means and predictor means/scales are estimated on the same
    boundary-valid calibration samples. Constant predictor columns contribute
    exactly zero. No target, scoring EEG, or scoring-time coefficient update is
    accepted by this function or the returned model.
    """
    eeg = np.asarray(calibration_eeg, dtype=np.float64)
    predictors = np.asarray(calibration_predictors, dtype=np.float64)
    if (eeg.ndim != 2 or predictors.ndim != 2 or eeg.shape[-1] != predictors.shape[-1]
            or eeg.shape[0] == 0 or predictors.shape[0] == 0
            or not np.isfinite(eeg).all() or not np.isfinite(predictors).all()):
        raise ValueError("Expected finite aligned [channel,time] EEG and [predictor,time] data")
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("Ridge penalty must be finite and positive")

    lag_values = _validate_lags(lags)
    boundary_values = _validate_boundaries(boundaries, eeg.shape[-1])
    design, valid = lag_matrix(predictors, lag_values)
    valid &= boundary_valid(eeg.shape[-1], boundary_values, lag_values)
    minimum = max(MIN_VALID_SAMPLES, 2 * design.shape[0])
    if int(valid.sum()) < minimum:
        raise ValueError("Insufficient calibration samples after lag and boundary exclusions")

    reference_mean = design[:, valid].mean(axis=-1, keepdims=True)
    reference_scale = design[:, valid].std(axis=-1, keepdims=True)
    active_columns = reference_scale[:, 0] > 0.
    safe_scale = np.where(active_columns[:, None], reference_scale, 1.)
    standardized = (design - reference_mean) / safe_scale
    standardized[~active_columns] = 0.

    centered_eeg = eeg - eeg[:, valid].mean(axis=-1, keepdims=True)
    covariance = standardized[:, valid] @ standardized[:, valid].T / valid.sum()
    cross_covariance = standardized[:, valid] @ centered_eeg[:, valid].T / valid.sum()
    coefficients = np.linalg.solve(
        covariance + float(penalty) * np.eye(design.shape[0]),
        cross_covariance,
    )

    return FrozenRidgeModel(
        lags=lag_values,
        penalty=float(penalty),
        reference_mean=reference_mean,
        reference_scale=safe_scale,
        active_columns=active_columns,
        coefficients=coefficients,
        calibration_valid_samples=int(valid.sum()),
        calibration_channel_count=int(eeg.shape[0]),
        predictor_count=int(predictors.shape[0]),
    )


def _axis_support(references, detector):
    """Reproduce the frozen detector per axis, without its combined OR gate."""
    values = np.asarray(references, dtype=np.float64)
    center = np.asarray(detector.center, dtype=np.float64)
    thresholds = np.asarray(detector.thresholds, dtype=np.float64)
    if (values.ndim != 2 or values.shape[0] != 2 or not np.isfinite(values).all()
            or center.shape != (2,) or thresholds.shape != (2, 2)
            or not np.isfinite(center).all() or not np.isfinite(thresholds).all()
            or np.any(thresholds < 0)):
        raise ValueError("Expected finite HEOG/VEOG and a fitted axis-amplitude detector")
    window, hop = int(detector.window), int(detector.hop)
    if window != 256 or hop != 128:
        raise ValueError("Frozen axis gate requires the calibrated 256/128 window contract")
    length = values.shape[-1]
    if length < window:
        raise ValueError("Scored trial/window is shorter than one axis-gate window")

    starts = sorted(set([*range(0, length - window + 1, hop), length - window]))
    support = np.zeros((2, length), dtype=np.float64)
    detected_windows = np.zeros(2, dtype=int)
    for start in starts:
        segment = values[:, start:start + window] - center[:, None]
        amplitudes = np.stack([
            np.std(segment, axis=-1),
            np.quantile(np.abs(segment), .95, axis=-1),
        ])
        detected = np.any(amplitudes > thresholds, axis=0)
        detected_windows += detected.astype(int)
        for axis in range(2):
            if detected[axis]:
                support[axis, start:start + window] = 1.

    kernel = np.hanning(43)[1:-1]
    kernel /= kernel.sum()
    for axis in range(2):
        support[axis] = maximum_filter1d(support[axis], size=41, mode="constant")
        support[axis] *= np.convolve(support[axis], kernel, mode="same")

    details = {
        "axis_support_mean": support.mean(axis=1).tolist(),
        "axis_detected_windows": detected_windows.tolist(),
        "axis_thresholds": thresholds.tolist(),
        "support_window": window,
        "support_hop": hop,
        "dilation_samples": 41,
        "taper_hann_window": 43,
        "taper_kernel_samples": int(len(kernel)),
        "support_policy": "separate HEOG and VEOG STD or 95th-amplitude gates",
        "cross_axis_or_gate": False,
        "scoring_condition_access": False,
    }
    return support, details


def apply_axis_gate(axis_artifacts, references, detector):
    """Gate each calibrated axis field independently, then sum the fields.

    The caller supplies one scored trial/window at a time. The passed detector
    must be fitted by ``fit_axis_amplitude_support`` on resting calibration only.
    """
    fields = np.asarray(axis_artifacts, dtype=np.float64)
    values = np.asarray(references, dtype=np.float64)
    if (fields.ndim != 3 or fields.shape[0] != 2 or not np.isfinite(fields).all()
            or values.shape != (2, fields.shape[-1]) or not np.isfinite(values).all()):
        raise ValueError("Expected finite [2,channel,time] axis fields and aligned references")
    if not isinstance(detector, AxisAmplitudeSupport):
        raise ValueError("A fitted AxisAmplitudeSupport detector is required")

    support, details = _axis_support(values, detector)
    artifact = np.sum(fields * support[:, None, :], axis=0)
    return artifact, details
