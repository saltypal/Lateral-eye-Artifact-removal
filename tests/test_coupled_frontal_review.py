"""Regional mask protection for the fixed coupled pilot comparison."""
import numpy as np
import pytest

from vmd_eog.coupled_frontal_review import regional_direct, validate_mode_reconstruction


def test_direct_branch_cannot_modify_frontal_unknown_or_masked_rows():
    rng = np.random.default_rng(42)
    refs = rng.normal(size=(2, 512))
    eeg = rng.normal(size=(8, 512)) + np.arange(1, 9)[:, None] * refs[0]
    calibration = {"eeg": eeg, "references": refs, "boundaries": np.array([256])}
    regions = np.array([0, 0, 1, 1, 2, 2, -1, 1])
    mask = np.array([True] * 7 + [False])
    artifact, details, fitted = regional_direct(eeg, refs, calibration, regions, mask)
    np.testing.assert_array_equal(artifact[[0, 1, 6, 7]], np.zeros((4, 512)))
    assert np.any(artifact[[2, 3, 4, 5]])
    assert fitted.calibration_valid_samples < 512
    assert all(item["paired_target_access"] is False for item in details)


def test_storage_rounding_does_not_hide_broken_mode_reconstruction():
    rng = np.random.default_rng(2026)
    vectors = rng.normal(size=(3, 1024)) * 1e-5
    original = rng.normal(size=1024) * 1e-7
    remainder = original - vectors.sum(axis=0)
    cached_modes, cached_residual = vectors.astype(np.float32), remainder.astype(np.float32)
    diagnostics = validate_mode_reconstruction(cached_modes, cached_residual, original)
    assert diagnostics["accumulation_dtype"] == "float64"
    broken = cached_modes.copy()
    broken[0, 400] += 1e-6
    with pytest.raises(ValueError, match="rounding budget"):
        validate_mode_reconstruction(broken, cached_residual, original)
    with pytest.raises(ValueError, match="Invalid stored"):
        validate_mode_reconstruction(cached_modes[:, :-1], cached_residual, original)
