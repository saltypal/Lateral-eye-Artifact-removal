"""Declared-bank replay contracts; numerical execution belongs on Kaggle."""
import numpy as np
import pandas as pd
import pytest

from vmd_eog.frontal_bounded_feasibility import (
    BASELINES, cached_admission, candidate_bank, preservation_reviews, replay_window, reporting_condition)
from vmd_eog.frontal_axis_diagnostic import window_artifacts
from vmd_eog.native_protocol import multi_overlap_add


def fixture(association=.6):
    original = np.ones((3, 1024)) * 10
    fields = np.zeros((2, 3, 1024))
    fields[0, :2] = 3.; fields[1, :2] = -2.
    saved = dict(original=original, references=np.zeros((2, 1024)),
                 raw_axis_fields=fields, vmd_axis_fields=fields.copy(),
                 axis_supports=np.ones((2, 1024)), union_support=np.ones(1024))
    detail = dict(gain=1., field_association=association, triggered=bool(association >= .4))
    return saved, {representation: [detail.copy(), detail.copy()] for representation in ("raw", "vmd")}


def test_bank_is_twelve_settings_with_four_matched_paths():
    bank = candidate_bank()
    assert len(bank) == len({row["method"] for row in bank}) == 48
    assert len({(row["threshold"], row["strength"]) for row in bank}) == 12
    assert set((row["representation"], row["support"]) for row in bank) == set(BASELINES)


def test_baseline_parity_and_attenuation_preserve_signed_cancellation():
    saved, decisions = fixture()
    bank = candidate_bank(); outputs = replay_window(saved, decisions, bank)
    baseline = window_artifacts(saved)
    for row in bank:
        key = BASELINES[(row["representation"], row["support"])]
        expected = baseline[key] * (row["strength"] if row["threshold"] <= .6 else 0.)
        np.testing.assert_array_equal(outputs[row["method"]], expected)
        assert not outputs[row["method"]][2].any()
    assert np.all(outputs["vmd_or_a60_s100"][:2] == 1.)
    saved["axis_supports"][1] = 0.
    assert np.all(replay_window(saved, decisions, bank)["vmd_separate_a60_s100"][:2] == 3.)


def test_admission_boundary_and_unsupported_bank_fail_closed():
    _, decisions = fixture(.8)
    assert cached_admission(decisions["raw"], .8)
    with pytest.raises(ValueError, match="declared threshold"):
        cached_admission(decisions["raw"], .2)
    with pytest.raises(ValueError, match="original .4"):
        cached_admission([{**decisions["raw"][0], "triggered": False}], .4)
    with pytest.raises(ValueError, match="differ"):
        cached_admission([decisions["raw"][0], {**decisions["raw"][0], "gain": 2.}], .4)
    with pytest.raises(ValueError, match="association/gain"):
        cached_admission([{**decisions["raw"][0], "gain": np.nan}], .4)
    saved, decisions = fixture()
    bank = candidate_bank(); bank[0]["strength"] = 0.
    with pytest.raises(ValueError, match="strength"):
        replay_window(saved, decisions, bank)


def test_rejected_or_failed_zero_fields_remain_zero():
    saved, decisions = fixture(.3)
    saved["raw_axis_fields"][:] = 0.; saved["vmd_axis_fields"][:] = 0.
    assert all(not value.any() for value in replay_window(saved, decisions, candidate_bank()).values())


def test_worst_record_gate_is_unchanged_and_incomplete_coverage_fails():
    bank = candidate_bank()
    rows = [dict(record_id=name, method=bank[0]["method"], relative_error=value,
                 alpha_db=.1, beta_db=.1, covariance=.01)
            for name, value in (("study01_p02_prep", .005), ("study01_p03_prep", .011))]
    limits = dict(clean_change=.01, alpha_db=.5, beta_db=.5, covariance=.02)
    review = preservation_reviews(pd.DataFrame(rows), bank, limits)[0]
    assert review["gates"]["coverage"] and not review["gates"]["clean_modification"]
    assert not review["qualification"] and not review["global_recipe_selected"]
    assert not preservation_reviews(pd.DataFrame(rows[:1]), bank, limits)[0]["gates"]["coverage"]


def test_replay_overlap_add_subtracts_scaled_artifact_once():
    original = np.ones((3, 1400)) * 10.
    references = np.zeros((2, 1400))
    _, decisions = fixture()
    bank = candidate_bank()
    def estimate(values, refs):
        saved, _ = fixture(); saved["original"] = values; saved["references"] = refs
        return replay_window(saved, decisions, bank)
    outputs = multi_overlap_add(original, references, estimate,
                                methods=["identity", *[row["method"] for row in bank]])
    np.testing.assert_array_equal(outputs["identity"], original)
    np.testing.assert_array_equal(outputs["vmd_or_a60_s050"][2], original[2])
    np.testing.assert_allclose(outputs["vmd_or_a60_s050"][:2], 9.5)


def test_reporting_labels_require_source_split_and_rest_agreement():
    record = dict(record_id="p", partition=dict(role="development"),
                  trial_labels=[1, 2, 4], scoring_trial_ids=[1, 2], calibration_trial_ids=[0])
    assert reporting_condition(record, 1, set()) == "lateral"
    assert reporting_condition(record, 2, set()) == "blink"
    with pytest.raises(ValueError, match="resting reports"):
        reporting_condition(record, 1, {("p", 1)})
    record["calibration_trial_ids"] = [0, 1]
    with pytest.raises(ValueError, match="calibration"):
        reporting_condition(record, 2, set())
