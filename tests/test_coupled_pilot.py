"""Pilot integration fixtures execute only in the Kaggle numerical suite."""
import numpy as np
import pytest

from vmd_eog.coupled_pilot import channel_intersection, whole_trial_prefix


def test_complete_trial_prefix_does_not_cut_a_trial():
    duration, cuts = whole_trial_prefix([100, 200, 300], 400, 250)
    assert duration == 200
    np.testing.assert_array_equal(cuts, [100])
    duration, cuts = whole_trial_prefix([100, 200, 300], 400, 500)
    assert duration == 400
    np.testing.assert_array_equal(cuts, [100, 200, 300])
    with pytest.raises(ValueError, match="No complete"):
        whole_trial_prefix([100], 200, 50)


def test_intersection_preserves_recipient_order_and_rejects_duplicates():
    names = ["Fp1", "F3", "Fz", "F4", "C3", "Cz", "C4", "Pz"]
    left, right = channel_intersection(names, list(reversed(names)))
    assert left == list(range(8))
    assert right == list(reversed(range(8)))
    with pytest.raises(ValueError, match="Duplicate"):
        channel_intersection(names + ["FP1"], names)
