"""Kaggle numerical contracts for matched raw frontal context."""
import inspect

import numpy as np
import pytest

import vmd_eog.regional_context_match as match
from vmd_eog.regional_coverage import ARMS, BASE, COMPARATOR
from test_regional_coverage import fixture_arrays


def test_matched_parent_controls_and_metadata_must_be_exact():
    saved, data, _ = fixture_arrays()
    previous = {key: saved[key].copy() for key in ("original", "target", "names", "regions", "mask")}
    previous.update({arm: saved[BASE].copy() for arm in ARMS})
    previous["raw_frontal_comparator"] = saved[COMPARATOR].copy()
    match.assert_parent_predictions(previous, saved, data)
    previous["frozen_regional"][0, 0] += 1
    with pytest.raises(ValueError, match="frozen baseline"):
        match.assert_parent_predictions(previous, saved, data)


def test_matched_inference_uses_only_frozen_artifact_context():
    saved, data, _ = fixture_arrays()
    field = np.zeros_like(data["eeg"])
    field[0] = 2

    class FrozenModel:
        def artifact(self, predictors):
            np.testing.assert_array_equal(predictors[0], 2)
            return np.full((1, predictors.shape[-1]), 3.)

    artifact, failure, _ = match.estimate_raw_central(field, data["channel_names"],
        data["regions"], data["mask"], data["hemispheres"], (FrozenModel(), None, {}))
    result = match.central_compose(data["eeg"], saved[BASE], [2], artifact)
    assert failure is None
    np.testing.assert_array_equal(result[2], data["eeg"][2] - 3)
    np.testing.assert_array_equal(result[[0, 1, 3, 4]], saved[BASE][[0, 1, 3, 4]])
    forbidden = {"target", "eeg", "condition", "amplitude", "true_artifact"}
    assert not forbidden.intersection(inspect.signature(match.estimate_raw_central).parameters)


def test_zero_context_skips_ridge_intercept_and_retains_identity():
    _, data, _ = fixture_arrays()

    class MustNotPredict:
        def artifact(self, predictors):
            raise AssertionError("zero context must pass through")

    artifact, failure, details = match.estimate_raw_central(np.zeros_like(data["eeg"]),
        data["channel_names"], data["regions"], data["mask"], data["hemispheres"],
        (MustNotPredict(), None, {}))
    np.testing.assert_array_equal(artifact, 0.)
    assert failure is None
    assert details["zero_or_missing_context_pass_through"] is True


def test_unavailable_fit_failure_is_retained():
    _, data, _ = fixture_arrays()
    artifact, failure, _ = match.estimate_raw_central(np.zeros_like(data["eeg"]),
        data["channel_names"], data["regions"], data["mask"], data["hemispheres"],
        (None, "insufficient calibration", {}))
    np.testing.assert_array_equal(artifact, 0.)
    assert failure == "insufficient calibration"


def test_raw_fit_uses_only_raw_context_and_frozen_parameters(tmp_path, monkeypatch):
    _, data, calibration = fixture_arrays()
    calibration["references"] = np.zeros((2, 1024))
    calibration_path = tmp_path / "calibration.bin"
    calibration_path.write_bytes(b"calibration identity fixture")
    calls = []

    def context(cal, config, destination, cache_id, representation):
        assert cal is calibration
        assert representation == "raw"
        assert len(cache_id) == 64
        return {"metadata": {"available": True, "selected_chunks": [0]},
                "eeg": data["eeg"], "predictors": np.ones((5, 1024)), "boundaries": np.asarray([])}

    class Model:
        coefficients = np.zeros((15, 1))
        reference_mean = np.zeros(5)
        reference_scale = np.ones(5)
        active_columns = np.ones(15, dtype=bool)
        lags = (-10, 0, 10)
        penalty = .01
        metadata = {"fixture": True}

    def ridge(eeg, predictors, boundaries, lags, penalty):
        np.testing.assert_array_equal(eeg, data["eeg"][[2]])
        calls.append((lags, penalty))
        return Model()

    monkeypatch.setattr(match, "fit_context_calibration", context)
    monkeypatch.setattr(match, "fit_frozen_ridge", ridge)
    model, failure, record = match.fit_raw_context(calibration, calibration_path, tmp_path,
        {"coverage_recipe": {"lags": [-10, 0, 10], "penalty": .01}})
    assert model is not None and failure is None
    assert calls == [((-10, 0, 10), .01)]
    assert record["paired_target_access"] is False
    assert record["scoring_eeg_fit"] is False
