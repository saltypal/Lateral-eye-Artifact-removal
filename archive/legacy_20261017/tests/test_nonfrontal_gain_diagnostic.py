import numpy as np
import pytest

from vmd_eog.nonfrontal_gain_diagnostic import compose_field, evaluator_bounds


def test_fixed_strength_and_blend_preserve_frontal_unknown_and_masks():
    original = np.ones((5, 32))
    baseline = original.copy(); baseline[:3] -= .2
    reference = original.copy()
    reference[:3] -= np.array([.2, .4, .6])[:, None]
    regions, mask = np.array([0, 1, 2, 3, 2]), np.array([1, 1, 1, 1, 0], bool)
    output = compose_field(original, baseline, reference, regions, mask, .5, False)
    np.testing.assert_array_equal(output[0], baseline[0])
    np.testing.assert_allclose(output[1:3], np.array([.8, .7])[:, None] * np.ones((2, 32)))
    np.testing.assert_array_equal(output[3:], original[3:])
    blended = compose_field(original, baseline, reference, regions, mask, .5, True)
    np.testing.assert_allclose(blended[1:3], np.array([.7, .6])[:, None] * np.ones((2, 32)))
    with pytest.raises(ValueError):
        compose_field(original, baseline, reference, regions, mask, 1.1, False)


def test_optimistic_scalar_is_labelled_and_error_accounting_closes():
    target = np.arange(1, 33, dtype=float)[None, :]
    original = target + 1.
    baseline = original.copy()
    reference = original - 2.
    report = evaluator_bounds(original, target, baseline, reference)
    assert report["oracle_scalar"] == pytest.approx(.5)
    assert report["oracle_blend"] == pytest.approx(.5)
    assert report["deployable"] is False
    assert report["diagnostic_target_fit"] is True
    assert np.isinf(report["oracle_scalar_pooled_snr_db"])
    assert report["residual_energy"] == pytest.approx(report["mean_bias_energy"] + report["centered_error_energy"])
