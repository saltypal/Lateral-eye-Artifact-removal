"""Contracts for opt-in shared-gain HEOG/VEOG field contributions."""
import numpy as np
import pytest

from vmd_eog.frontal_calibration_transfer import calibration_transfer
from vmd_eog.reference import lag_matrix


def transfer_fixture(seed=81, scale=1.0):
    generator = np.random.default_rng(seed)
    calibration_references = generator.normal(size=(2, 4096)) * scale
    references = generator.normal(size=(2, 512)) * scale
    lags = (-10, 0, 10)
    calibration_design, _ = lag_matrix(calibration_references, lags)
    score_design, score_valid = lag_matrix(references, lags)
    spatial_coefficients = generator.normal(size=(4, len(calibration_design)))
    spatial_coefficients[3] = 0.0
    calibration_eeg = spatial_coefficients @ calibration_design
    original = spatial_coefficients @ score_design
    original[:, ~score_valid] = 0.0
    original += generator.normal(scale=0.01 * scale, size=original.shape)
    names = np.asarray(["F3", "F4", "Fz", "Cz"])
    regions = np.asarray([0, 0, 0, 2])
    mask = np.ones(4, dtype=bool)
    calibration = {
        "eeg": calibration_eeg,
        "references": calibration_references,
        "boundaries": np.asarray([2048]),
    }
    return original, references, calibration, names, regions, mask


def test_opt_in_axis_fields_sum_to_shared_artifact_and_default_stays_three_tuple():
    original, references, calibration, names, regions, mask = transfer_fixture()

    default = calibration_transfer(
        original, references, calibration, names, regions, mask, threshold=0.0,
    )
    expanded = calibration_transfer(
        original, references, calibration, names, regions, mask,
        threshold=0.0, return_axis_fields=True,
    )

    assert len(default) == 3
    assert len(expanded) == 4
    for original_value, expanded_value in zip(default, expanded[:3]):
        if isinstance(original_value, np.ndarray):
            np.testing.assert_array_equal(original_value, expanded_value)
        else:
            assert original_value == expanded_value
    cleaned, artifact, details, axis_artifacts = expanded
    assert axis_artifacts.shape == (2, *original.shape)
    assert np.isfinite(axis_artifacts).all()
    np.testing.assert_allclose(axis_artifacts.sum(axis=0), artifact, rtol=1e-12, atol=1e-12)
    assert all(row["gain_mode"] == "shared" for row in details)
    assert all(row["axis_gains"] is None for row in details)
    np.testing.assert_array_equal(cleaned, original - artifact)
    np.testing.assert_array_equal(axis_artifacts[:, 3], np.zeros((2, original.shape[-1])))


def test_axis_field_outputs_follow_reference_axis_permutation():
    original, references, calibration, names, regions, mask = transfer_fixture(seed=107)
    _, _, _, axis_fields = calibration_transfer(
        original, references, calibration, names, regions, mask,
        threshold=0.0, return_axis_fields=True,
    )
    swapped_calibration = {
        **calibration,
        "references": calibration["references"][::-1].copy(),
    }
    _, _, _, swapped_fields = calibration_transfer(
        original, references[::-1].copy(), swapped_calibration,
        names, regions, mask, threshold=0.0, return_axis_fields=True,
    )

    np.testing.assert_allclose(swapped_fields, axis_fields[::-1], rtol=1e-10, atol=1e-10)


def test_axis_fields_obey_mask_failed_mode_and_unit_contracts():
    original, references, calibration, names, regions, mask = transfer_fixture(seed=202)
    mask[2] = False
    front_ids = (0, 1)
    bundle = {
        0: (np.stack([references[0], np.zeros_like(references[0]),
                      np.zeros_like(references[0])]), np.zeros_like(references[0]), {}),
        1: (np.zeros((3, len(references[0]))), np.zeros_like(references[0]),
            {"failure": "fixture failure"}),
    }
    first = calibration_transfer(
        original, references, calibration, names, regions, mask,
        bundles=bundle, threshold=0.0, return_axis_fields=True,
    )
    scaled_calibration = {
        **calibration,
        "eeg": calibration["eeg"] * 1e-6,
        "references": calibration["references"] * 1e-6,
    }
    scaled_bundle = {
        index: (modes * 1e-6, residual * 1e-6, info)
        for index, (modes, residual, info) in bundle.items()
    }
    scaled = calibration_transfer(
        original * 1e-6, references * 1e-6, scaled_calibration,
        names, regions, mask, bundles=scaled_bundle,
        threshold=0.0, return_axis_fields=True,
    )

    artifact, axis_fields = first[1], first[3]
    np.testing.assert_allclose(axis_fields.sum(axis=0), artifact, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(axis_fields[:, 1:], np.zeros_like(axis_fields[:, 1:]))
    assert next(row for row in first[2] if row["channel"] == "F4")["failure"] == "scored_passthrough"
    np.testing.assert_allclose(scaled[1], artifact * 1e-6, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(scaled[3], axis_fields * 1e-6, rtol=1e-9, atol=1e-12)
    assert all(row["gain_mode"] == "shared" for row in first[2])
    assert all(row["axis_gains"] is None for row in first[2])
    assert front_ids == (0, 1)


def test_axis_field_opt_in_requires_shared_gain_and_handles_empty_frontal_set():
    original, references, calibration, names, regions, mask = transfer_fixture(seed=304)
    with pytest.raises(ValueError, match="requires the shared gain mode"):
        calibration_transfer(
            original, references, calibration, names, regions, mask,
            gain_mode="axis", return_axis_fields=True,
        )

    no_front_regions = np.ones_like(regions)
    cleaned, artifact, details, axis_fields = calibration_transfer(
        original, references, calibration, names, no_front_regions, mask,
        return_axis_fields=True,
    )
    np.testing.assert_array_equal(cleaned, original)
    np.testing.assert_array_equal(artifact, np.zeros_like(original))
    assert details == []
    assert axis_fields.shape == (2, *original.shape)
    np.testing.assert_array_equal(axis_fields, np.zeros_like(axis_fields))
