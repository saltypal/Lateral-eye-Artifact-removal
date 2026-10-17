import json

import numpy as np
import pytest

from vmd_eog.frontal_event_support import AxisAmplitudeSupport, gate_axis_artifact


def detector(threshold=0.5):
    return AxisAmplitudeSupport(
        center=np.zeros(2),
        thresholds=np.full((2, 2), threshold),
        calibration_counts={"rest": 12, "ocular_fitting": 0},
        window=256,
        hop=128,
    )


def references(length=512, *, horizontal=2.0, vertical=0.0):
    times = np.arange(length) / 200
    return np.stack([
        horizontal * np.sin(2 * np.pi * 3 * times),
        vertical * np.sin(2 * np.pi * 2 * times),
    ])


def test_axis_detection_is_independent_and_keeps_tapered_support():
    gate = detector()
    values = references(horizontal=2.0, vertical=0.0)
    supports, metadata = gate.support_axes(values)
    assert supports.shape == (2, values.shape[-1])
    assert np.any(supports[0] > 0)
    assert not np.any(supports[1])
    assert metadata["axis_detected_windows"]["HEOG"] > 0
    assert metadata["axis_detected_windows"]["VEOG"] == 0
    assert np.any((supports[0] > 0) & (supports[0] < 1))


def test_zero_reference_support_is_identity_and_artifact_zero():
    original = np.arange(3 * 512, dtype=float).reshape(3, 512)
    fields = np.ones((2, 3, 512))
    cleaned, artifact, metadata = gate_axis_artifact(
        original, fields, np.zeros((2, 512)), detector(), diagnostics={"fit": "frozen"}
    )
    np.testing.assert_array_equal(artifact, np.zeros_like(original))
    np.testing.assert_array_equal(cleaned, original)
    assert metadata["axis_support_means"] == {"HEOG": 0.0, "VEOG": 0.0}
    json.dumps(metadata)


def test_equal_and_opposite_fields_cancel_only_when_both_axes_are_supported():
    gate = detector()
    refs = references(horizontal=2.0, vertical=2.0)
    original = np.zeros((1, refs.shape[-1]))
    fields = np.stack([
        np.ones_like(original) * 3.0,
        np.ones_like(original) * -3.0,
    ])
    both_cleaned, both_artifact, both_details = gate_axis_artifact(
        original, fields, refs, gate, diagnostics=[]
    )
    np.testing.assert_array_equal(both_artifact, np.zeros_like(original))
    np.testing.assert_array_equal(both_cleaned, original)
    assert all(both_details["axis_support_means"][axis] > 0 for axis in ("HEOG", "VEOG"))

    one_axis_cleaned, one_axis_artifact, one_axis_details = gate_axis_artifact(
        original, fields, references(horizontal=2.0, vertical=0.0), gate, diagnostics=[]
    )
    assert np.any(one_axis_artifact != 0)
    assert np.any(one_axis_cleaned != original)
    assert one_axis_details["axis_support_means"]["HEOG"] > 0
    assert one_axis_details["axis_support_means"]["VEOG"] == 0


def test_equal_axis_detector_inputs_produce_equal_tapered_rows():
    gate = detector()
    values = references(horizontal=1.5, vertical=1.5)
    supports, _ = gate.support_axes(values)
    np.testing.assert_array_equal(supports[0], supports[1])
    assert np.any((supports[0] > 0) & (supports[0] < 1))


@pytest.mark.parametrize(
    "original,fields,refs",
    [
        (np.zeros((3, 512)), np.zeros((2, 2, 512)), np.zeros((2, 512))),
        (np.zeros((3, 512)), np.zeros((2, 3, 511)), np.zeros((2, 512))),
        (np.zeros((3, 512)), np.zeros((2, 3, 512)), np.zeros((1, 512))),
        (np.zeros((3, 512)), np.full((2, 3, 512), np.nan), np.zeros((2, 512))),
        (np.zeros((3, 512)), np.zeros((2, 3, 512)), np.full((2, 512), np.inf)),
    ],
)
def test_gate_rejects_invalid_axes_and_nonfinite_inputs(original, fields, refs):
    with pytest.raises(ValueError):
        gate_axis_artifact(original, fields, refs, detector(), diagnostics=[])


def test_support_axes_rejects_short_or_nonfinite_reference_inputs():
    gate = detector()
    with pytest.raises(ValueError, match="short"):
        gate.support_axes(np.zeros((2, 255)))
    values = np.zeros((2, 512))
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        gate.support_axes(values)


def test_gate_details_are_json_compatible_and_document_no_reference_scaling():
    refs = references()
    original = np.zeros((2, refs.shape[-1]))
    fields = np.ones((2, 2, refs.shape[-1]))
    _, _, details = gate_axis_artifact(
        original, fields, refs, detector(), diagnostics=[{"axis": "HEOG", "gain": 1.0}]
    )
    assert details["reference_scaling"] is False
    assert details["whole_window_routing"] is False
    assert details["scoring_condition_access"] is False
    json.dumps(details)


def test_all_one_supports_reconstruct_joint_field_and_zero_supports_preserve_input(monkeypatch):
    gate = detector()
    original = np.arange(1024, dtype=float).reshape(2, 512)
    fields = np.stack([np.ones_like(original) * 3.0, np.ones_like(original) * -2.0])
    monkeypatch.setattr(gate, "support_axes", lambda refs: (np.ones((2, 512)), {}))
    cleaned, artifact, _ = gate_axis_artifact(original, fields, np.zeros((2, 512)), gate, [])
    np.testing.assert_array_equal(artifact, fields.sum(0))
    np.testing.assert_array_equal(cleaned, original - fields.sum(0))
    monkeypatch.setattr(gate, "support_axes", lambda refs: (np.zeros((2, 512)), {}))
    cleaned, artifact, _ = gate_axis_artifact(original, fields, np.zeros((2, 512)), gate, [])
    np.testing.assert_array_equal(artifact, np.zeros_like(original))
    np.testing.assert_array_equal(cleaned, original)
