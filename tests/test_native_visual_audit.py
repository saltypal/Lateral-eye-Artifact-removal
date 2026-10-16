"""Known lag/polarity and constant-mode audit contracts; run on Kaggle."""
import numpy as np

from vmd_eog.native_visual_audit import associations


def test_mode_audit_retains_reference_axis_sign_peak_lag_and_constant_modes():
    refs = np.random.default_rng(42).normal(size=(2, 256))
    modes = np.stack([np.pad(refs[0, :-7], (7, 0)),
                      -np.pad(refs[1, 3:], (0, 3)), np.zeros(256)])
    zero, peak, lag = associations(modes, refs)
    np.testing.assert_allclose(peak[0, 0], 1., atol=1e-12)
    np.testing.assert_allclose(peak[1, 1], -1., atol=1e-12)
    assert lag[0, 0] == 7 and lag[1, 1] == -3
    assert abs(zero[0, 0]) < .5 and abs(zero[1, 1]) < .5
    np.testing.assert_array_equal(peak[2], 0.)
    np.testing.assert_array_equal(lag[2], 0)
