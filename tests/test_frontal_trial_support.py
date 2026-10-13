"""Trial alignment and calibration admission contracts; execute on Kaggle."""
import json

import numpy as np
import pytest

from vmd_eog.frontal_event_support import AxisAmplitudeSupport
from vmd_eog.frontal_trial_support import (
    METHODS, TRIAL_METHODS, check_local_support, checked_calibration, support_slice, supported_fields)
from vmd_eog.frontal_bounded_feasibility import BASELINES
from vmd_eog.native_protocol import multi_overlap_add


def saved_window():
    original = np.ones((3, 1024)) * 10.
    fields = np.zeros((2, 3, 1024))
    fields[0, :2] = 3.; fields[1, :2] = -2.
    return dict(original=original, references=np.zeros((2, 1024)),
                raw_axis_fields=fields, vmd_axis_fields=fields.copy(),
                axis_supports=np.ones((2, 1024)), union_support=np.ones(1024))


def test_nine_outputs_and_signed_axis_cancellation_are_preserved():
    saved = saved_window()
    axes = np.ones((2, 1024)); axes[1] = 0
    result = supported_fields(saved, axes, np.ones(1024))
    assert set(result) == set(METHODS) and len(result) == 9
    np.testing.assert_array_equal(result["vmd_trial_or"][:2], np.ones((2, 1024)))
    np.testing.assert_array_equal(result["vmd_trial_separate"][:2], np.ones((2, 1024)) * 3)
    assert all(not values[2].any() for values in result.values())


def test_original_samples_have_identical_support_in_every_overlap():
    track = np.linspace(0, 1, 1400)
    for number in range(1, 6):
        sliced = support_slice(track, number)
        zero = support_slice(track, number, padding="constant")
        start = (number - 1) * 512
        for sample in (0, 100, 511, 512, 900, 1399):
            offset = 512 + sample - start
            if 0 <= offset < 1024:
                assert sliced[offset] == track[sample] == zero[offset]


def test_zero_and_reflection_padding_produce_identical_cropped_correction():
    original = np.ones((3, 1400)) * 10.
    references = np.zeros((2, 1400))
    union = np.linspace(0, 1, 1400)
    axes = np.stack([union, 1 - union])
    def run(padding):
        number = 0
        def estimate(values, refs):
            nonlocal number
            number += 1
            saved = saved_window(); saved["original"] = values; saved["references"] = refs
            return supported_fields(saved, support_slice(axes, number, padding=padding),
                                    support_slice(union, number, padding=padding))
        return multi_overlap_add(original, references, estimate, methods=METHODS)
    reflected, zeros = run("reflect"), run("constant")
    for method in METHODS:
        np.testing.assert_array_equal(reflected[method], zeros[method])
    expected = np.broadcast_to(10 - union[None], reflected["vmd_trial_or"][:2].shape)
    np.testing.assert_allclose(reflected["vmd_trial_or"][:2], expected)
    np.testing.assert_array_equal(reflected["vmd_trial_or"][2], original[2])


def test_zero_and_one_support_identity_and_baseline_parity():
    saved = saved_window()
    zero = supported_fields(saved, np.zeros((2, 1024)), np.zeros(1024))
    one = supported_fields(saved, np.ones((2, 1024)), np.ones(1024))
    for key, method in TRIAL_METHODS.items():
        assert not zero[method].any()
        np.testing.assert_array_equal(one[method], one[BASELINES[key]])
    saved["vmd_axis_fields"][:] = 0
    assert not supported_fields(saved, np.ones((2, 1024)), np.ones(1024))["vmd_trial_or"].any()


def test_invalid_support_position_shape_and_values_fail_closed():
    with pytest.raises(ValueError, match="identity"):
        support_slice(np.ones(500), 0)
    with pytest.raises(ValueError, match="padding contract"):
        support_slice(np.ones(500), 20)
    with pytest.raises(ValueError, match="Invalid"):
        support_slice(np.ones(1), 1)
    with pytest.raises(ValueError, match="bounded"):
        supported_fields(saved_window(), np.ones((1, 1024)), np.ones(1024))
    with pytest.raises(ValueError, match="bounded"):
        supported_fields(saved_window(), np.ones((2, 1024)) * 1.1, np.ones(1024))


def test_detector_parity_checks_thresholds_and_every_axis_track():
    gate = AxisAmplitudeSupport(np.zeros(2), np.ones((2, 2)), dict(rest=4))
    saved = saved_window()
    saved["axis_supports"] = gate.support_axes(saved["references"])[0]
    saved["union_support"] = gate.support(saved["references"])[0]
    details = dict(axis_thresholds=gate.thresholds.tolist(), support_window=256, support_hop=128)
    entries = {method: [details.copy()] for method in BASELINES.values()}
    check_local_support(saved, gate, entries)
    saved["axis_supports"][0, 200] = 1
    with pytest.raises(AssertionError):
        check_local_support(saved, gate, entries)
    saved["axis_supports"] = gate.support_axes(saved["references"])[0]
    entries[next(iter(entries))][0]["axis_thresholds"] = np.ones((2, 2)).tolist() * 2
    with pytest.raises(AssertionError):
        check_local_support(saved, gate, entries)


def test_gate_plateau_roundoff_is_accepted_without_clipping():
    gate = AxisAmplitudeSupport(np.zeros(2), np.full((2, 2), .1), dict(rest=4))
    refs = np.stack([np.sin(np.arange(1600) / 10), np.cos(np.arange(1600) / 10)]) * 5
    axes = gate.support_axes(refs)[0]
    union = gate.support(refs)[0]
    np.testing.assert_array_equal(support_slice(axes, 1)[..., 512:], axes[..., :512])
    saved = saved_window()
    result = supported_fields(saved, support_slice(axes, 1), support_slice(union, 1))
    assert np.isfinite(result["vmd_trial_or"]).all()


def calibration_fixture(directory):
    generator = np.random.default_rng(42)
    record = dict(channel_names=["F3", "F4"], partition=dict(role="development"),
                  calibration_trial_ids=[0, 1], scoring_trial_ids=[2], trial_labels=[1, 1, 4])
    (directory / "record.json").write_text(json.dumps(record))
    np.savez(directory / "calibration.npz", references=generator.normal(size=(2, 2048)),
             trial_types=np.ones(2048, int), boundaries=np.array([1024]), channel_names=np.array(["F3", "F4"]))
    return record


def test_only_exact_unscored_source_calibration_reconstructs_detector(tmp_path):
    record = calibration_fixture(tmp_path)
    calibration, gate = checked_calibration(tmp_path, record)
    assert calibration["references"].shape == (2, 2048) and gate.thresholds.shape == (2, 2)
    with pytest.raises(ValueError, match="ledgers"):
        checked_calibration(tmp_path, {**record, "partition": dict(role="confirmation")})
    record["calibration_trial_ids"] = [0, 2]
    (tmp_path / "record.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="trial overlap"):
        checked_calibration(tmp_path, record)


def test_calibration_sample_labels_and_trial_boundaries_must_match_ledger(tmp_path):
    record = calibration_fixture(tmp_path)
    record["trial_labels"][1] = 2
    (tmp_path / "record.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="sample labels"):
        checked_calibration(tmp_path, record)
