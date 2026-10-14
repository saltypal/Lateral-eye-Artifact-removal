"""Regional mask protection for the fixed coupled pilot comparison."""
import numpy as np

from vmd_eog.coupled_frontal_review import regional_direct


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
