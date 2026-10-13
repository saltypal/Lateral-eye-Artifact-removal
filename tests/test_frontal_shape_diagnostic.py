import numpy as np
import pytest

from vmd_eog.frontal_shape_diagnostic import error_terms


def test_discarded_field_and_positive_cross_term_explain_error():
    original = np.array([[3., -3., 3., -3.]])
    target = np.zeros_like(original)
    full = original * 2 / 3
    shape = full / 2
    result = error_terms(original, target, full, shape)
    np.testing.assert_allclose(result['full_mse'], [1.])
    np.testing.assert_allclose(result['shape_mse'], [4.])
    np.testing.assert_allclose(result['discarded_field_mse'], [1.])
    np.testing.assert_allclose(result['cross_term'], [2.])
    np.testing.assert_allclose(result['mse_delta'], [3.])


def test_negative_cross_term_can_improve_a_saved_correction():
    original = np.array([[1., -1., 1., -1.]])
    target = np.zeros_like(original)
    full = 2 * original
    result = error_terms(original, target, full, original)
    np.testing.assert_allclose(result['discarded_field_mse'], [1.])
    np.testing.assert_allclose(result['cross_term'], [-2.])
    np.testing.assert_allclose(result['mse_delta'], [-1.])


def test_zero_fields_and_physical_unit_scaling():
    original = np.array([[3., -3.], [1., -1.]])
    zero = np.zeros_like(original)
    result = error_terms(original, zero, original / 2, zero)
    scaled = error_terms(original * 1e-6, zero, original * .5e-6, zero)
    for key in result:
        np.testing.assert_allclose(scaled[key], result[key] * 1e-12, atol=1e-26)
    identity = error_terms(original, original, zero, zero)
    for values in identity.values():
        np.testing.assert_array_equal(values, [0., 0.])


@pytest.mark.parametrize('bad', [np.zeros(3), np.zeros((0, 3)), np.zeros((1, 2)), np.full((1, 3), np.nan)])
def test_reject_malformed_or_nonfinite_inputs(bad):
    valid = np.zeros((1, 3))
    with pytest.raises(ValueError):
        error_terms(valid, valid, bad, valid)
