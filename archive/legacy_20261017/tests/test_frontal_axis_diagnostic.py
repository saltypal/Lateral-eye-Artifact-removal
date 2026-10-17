"""Cached intervention reconstruction and descriptive energy contracts."""
import numpy as np
import pytest

from vmd_eog.frontal_axis_diagnostic import original_overlap, trial_identity, window_artifacts, window_measures


def fixture():
    original = np.ones((3, 512)) * 10
    fields = np.zeros((2, 3, 512))
    fields[0, :2] = 3
    fields[1, :2] = -2
    return dict(original=original, references=np.zeros((2, 512)), axis_supports=np.ones((2, 512)),
                union_support=np.ones(512), raw_axis_fields=fields, vmd_axis_fields=fields.copy())


def test_cached_reconstruction_retains_cancellation_and_nonfrontal_pass_through():
    saved = fixture()
    outputs = window_artifacts(saved)
    np.testing.assert_array_equal(outputs["identity"], np.zeros_like(saved["original"]))
    np.testing.assert_array_equal(outputs["K3_calibration_event_gain"], outputs["K3_calibration_axis_event_gain"])
    assert (outputs["K3_calibration_event_gain"][:2] == 1).all()
    assert not outputs["K3_calibration_event_gain"][2].any()
    saved["axis_supports"][1] = 0
    changed = window_artifacts(saved)
    assert (changed["K3_calibration_axis_event_gain"][:2] == 3).all()
    assert (changed["K3_calibration_event_gain"][:2] == 1).all()


def test_measures_distinguish_cancellation_support_and_shared_decisions():
    saved = fixture()
    details = [dict(gain=1., field_association=.6, triggered=True, failure=None)] * 2
    row = window_measures(saved, [0, 1], details, "vmd")
    assert row["field_cross_term"] < 0 and row["cancellation_fraction"] > 0
    assert row["or_correction_fraction"] == pytest.approx(.1)
    saved["axis_supports"][1] = 0
    row = window_measures(saved, [0, 1], details, "vmd")
    assert row["heog_only"] is True and row["separate_correction_fraction"] == pytest.approx(.3)
    with pytest.raises(ValueError, match="across channels"):
        window_measures(saved, [0, 1], [details[0], {**details[0], "gain": 2.}], "vmd")


def test_cached_artifact_rejects_shape_or_nonfinite_drift():
    saved = fixture(); saved["axis_supports"] = np.zeros((1, 512))
    with pytest.raises(ValueError, match="supports"):
        window_artifacts(saved)
    saved = fixture(); saved["raw_axis_fields"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="fields"):
        window_artifacts(saved)


def test_saved_trial_identity_is_explicit_and_strict():
    assert trial_identity("study01_p02_prep-trial-0026") == ("study01_p02_prep", 26)
    with pytest.raises(ValueError, match="identity"):
        trial_identity("study01_p02_prep-window-0001")


def test_detector_only_support_is_not_counted_as_nonzero_intervention():
    saved = fixture()
    saved["axis_supports"][1] = 0
    saved["vmd_axis_fields"][0] = 0
    details = [dict(gain=1., field_association=.6, triggered=True, failure=None)]
    row = window_measures(saved, [0, 1], details, "vmd")
    assert row["heog_only"] is True
    assert row["heog_only_intervention"] is False
    assert row["separate_nonzero_intervention"] is False


def test_original_overlap_distinguishes_padding_only_terminal_windows():
    assert original_overlap(1, 1024, 512, 1000) == (0, 512)
    assert original_overlap(2, 1024, 512, 1000) == (512, 1000)
    assert original_overlap(3, 1024, 512, 1000) == (1024, 488)
    assert original_overlap(4, 1024, 512, 1000) == (1536, 0)
    with pytest.raises(ValueError, match="overlap"):
        original_overlap(0, 1024, 512, 1000)
