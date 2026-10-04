"""Run only on Kaggle for this campaign. Numeric fixtures are not dataset evidence."""
import numpy as np
from eog_vmd_fcm_bgru.channel_regions import region_ids, fuse_residuals
from eog_vmd_fcm_bgru.vmd_expert import decompose


def test_vmd_odd_length_residual_keeps_alignment():
    t = np.arange(1001) / 200
    x = np.sin(2 * np.pi * 3 * t) + 0.3 * np.sin(2 * np.pi * 12 * t)
    vectors, residual, diagnostics = decompose(x)
    assert vectors.shape == (5, 1001)
    np.testing.assert_allclose(vectors.sum(axis=0) + residual, x, atol=1e-6)
    assert diagnostics["centers_hz"] == sorted(diagnostics["centers_hz"])


def test_fusion_identity_and_permutation():
    rng = np.random.default_rng(42)
    eeg = rng.normal(size=(4, 500))
    zero = np.zeros_like(eeg)
    names = ["Fp1", "FC1", "O1", "unknown"]
    assert region_ids(names).tolist() == [0, 2, 1, 3]
    np.testing.assert_array_equal(fuse_residuals(eeg, zero, zero, names), eeg)
    a, b = rng.normal(size=eeg.shape), rng.normal(size=eeg.shape)
    order = [2, 0, 3, 1]
    expected = fuse_residuals(eeg, a, b, names)[order]
    actual = fuse_residuals(eeg[order], a[order], b[order], [names[i] for i in order])
    np.testing.assert_allclose(actual, expected)
