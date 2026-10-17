"""Solver flags may switch only eligible frontal rows to the matched raw arm."""
import numpy as np
import pytest

from vmd_eog.broader_cached_review import raw_failure_fallback


def test_failed_frontal_rows_use_raw_and_every_other_row_stays_bitwise_fixed():
    original = np.arange(32, dtype=float).reshape(4, 8)
    anchored, raw = original - 1., original - 2.
    names = np.array(["Fp1", "P3", "C3", "Fp2"])
    regions = np.array([0, 1, 2, 0])
    mask = np.array([True, True, True, False])
    result = raw_failure_fallback(original, anchored, raw, {0}, names, regions, mask)
    np.testing.assert_array_equal(result[0], raw[0])
    np.testing.assert_array_equal(result[1:], anchored[1:])
    np.testing.assert_array_equal(raw_failure_fallback(original, anchored, raw, set(), names, regions, mask), anchored)
    for invalid in ({1}, {2}, {3}):
        with pytest.raises(ValueError, match="failed verified frontal"):
            raw_failure_fallback(original, anchored, raw, invalid, names, regions, mask)
