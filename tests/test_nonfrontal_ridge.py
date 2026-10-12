"""Numerical contract fixtures for the frozen nonfrontal ridge module.

These execute only in the Kaggle test stage with the rest of the campaign.
"""
import inspect

import numpy as np
import pytest

from vmd_eog.frontal_event_support import AxisAmplitudeSupport
from vmd_eog.nonfrontal_ridge import apply_axis_gate, fit_frozen_ridge
from vmd_eog.posterior import boundary_valid
from vmd_eog.reference import lag_matrix


def test_frozen_zero_lag_ridge_generalizes_without_scoring_fit():
    generator = np.random.default_rng(314)
    calibration_references = generator.normal(size=(2, 2048))
    scoring_references = generator.normal(size=(2, 1024))
    true_weights = generator.normal(size=(3, 2))
    channel_offsets = np.array([[4.], [-2.], [1.5]])
    calibration_eeg = true_weights @ calibration_references + channel_offsets

    model = fit_frozen_ridge(
        calibration_eeg,
        calibration_references,
        lags=(0,),
        penalty=1e-8,
    )

    expected = true_weights @ (scoring_references - calibration_references.mean(axis=1, keepdims=True))
    np.testing.assert_allclose(model.artifact(scoring_references), expected, rtol=1e-4, atol=1e-5)
    assert model.metadata["scoring_eeg_access"] is False
    assert model.metadata["paired_target_access"] is False
    assert "target" not in inspect.signature(fit_frozen_ridge).parameters


def test_joint_axis_fields_sum_to_frozen_field_and_zero_lag_edges():
    generator = np.random.default_rng(2718)
    references = generator.normal(size=(2, 1024))
    calibration_eeg = generator.normal(size=(4, 1024))
    model = fit_frozen_ridge(calibration_eeg, references, lags=(-3, 0, 3))

    axis_fields = model.axis_artifacts(references)
    joint_field = model.artifact(references)
    assert axis_fields.shape == (2, 4, 1024)
    np.testing.assert_allclose(axis_fields.sum(axis=0), joint_field, rtol=1e-12, atol=1e-12)
    assert not axis_fields[:, :, :3].any()
    assert not axis_fields[:, :, -3:].any()


def test_lag_training_excludes_outer_edges_and_trial_joins():
    generator = np.random.default_rng(17)
    references = generator.normal(size=(2, 1024))
    calibration_eeg = generator.normal(size=(3, 1024))
    lags = (-10, 0, 10)
    boundaries = (512,)

    model = fit_frozen_ridge(
        calibration_eeg,
        references,
        boundaries=boundaries,
        lags=lags,
    )

    _, lag_valid = lag_matrix(references, lags)
    expected_valid = lag_valid & boundary_valid(len(references[0]), boundaries, lags)
    assert model.calibration_valid_samples == int(expected_valid.sum())


def test_constant_predictor_columns_are_zero_and_invalid_edges_stay_zero():
    generator = np.random.default_rng(33)
    varying = generator.normal(size=1024)
    predictors = np.stack([varying, np.full_like(varying, 7.)])
    eeg = generator.normal(size=(2, 1024))
    model = fit_frozen_ridge(eeg, predictors, lags=(-4, 0, 4))

    assert model.metadata["active_lag_columns"] == 3
    assert not model.coefficients[~model.active_columns].any()
    artifact = model.artifact(predictors)
    assert not artifact[:, :4].any()
    assert not artifact[:, -4:].any()


@pytest.mark.parametrize(
    "eeg,predictors,boundaries,lags,penalty",
    [
        (np.ones(64), np.ones((2, 64)), (), (0,), .01),
        (np.ones((2, 128)), np.ones((3, 127)), (), (0,), .01),
        (np.full((2, 128), np.nan), np.ones((2, 128)), (), (0,), .01),
        (np.ones((2, 128)), np.ones((2, 128)), (64, 64), (0,), .01),
        (np.ones((2, 128)), np.ones((2, 128)), (), (0,), 0.),
        (np.ones((2, 64)), np.ones((2, 64)), (), (-10, 0, 10), .01),
    ],
)
def test_bad_alignment_or_insufficient_calibration_is_rejected(eeg, predictors, boundaries, lags, penalty):
    with pytest.raises(ValueError):
        fit_frozen_ridge(eeg, predictors, boundaries=boundaries, lags=lags, penalty=penalty)


def test_axis_gate_does_not_or_gate_quiet_axis_into_correction():
    detector = AxisAmplitudeSupport(
        center=np.zeros(2),
        thresholds=np.ones((2, 2)),
        calibration_counts={"rest": 4, "ocular_fitting": 0},
    )
    time = np.arange(1024)
    references = np.stack([5 * np.sin(time / 16), np.zeros_like(time)])
    axis_fields = np.zeros((2, 2, 1024))
    axis_fields[0] = 1.
    axis_fields[1] = 1000.

    artifact, details = apply_axis_gate(axis_fields, references, detector)

    assert details["axis_support_mean"][0] > 0.
    assert details["axis_support_mean"][1] == 0.
    assert details["axis_detected_windows"][0] > 0
    assert details["axis_detected_windows"][1] == 0
    assert artifact.max() < 2.


def test_zero_reference_excursions_leave_identity_artifact_zero():
    detector = AxisAmplitudeSupport(
        center=np.zeros(2),
        thresholds=np.ones((2, 2)),
        calibration_counts={"rest": 4, "ocular_fitting": 0},
    )
    references = np.zeros((2, 512))
    axis_fields = np.ones((2, 3, 512))

    artifact, details = apply_axis_gate(axis_fields, references, detector)

    assert not artifact.any()
    assert details["axis_support_mean"] == [0., 0.]


def test_axis_gate_rejects_bad_shapes_and_short_scored_windows():
    detector = AxisAmplitudeSupport(np.zeros(2), np.ones((2, 2)), {})
    with pytest.raises(ValueError, match="aligned"):
        apply_axis_gate(np.ones((2, 3, 512)), np.ones((2, 511)), detector)
    with pytest.raises(ValueError, match="shorter"):
        apply_axis_gate(np.ones((2, 3, 128)), np.ones((2, 128)), detector)
