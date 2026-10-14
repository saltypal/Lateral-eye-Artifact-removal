"""Pilot integration fixtures execute only in the Kaggle numerical suite."""
import numpy as np
import pytest

from vmd_eog.coupled_pilot import channel_intersection, whole_trial_prefix, validate_generated_pair
from vmd_eog.coupled_field import fit_donor_field, generated_pair


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


def test_float32_source_validation_uses_producer_precision_and_rejects_bad_coupling():
    rng = np.random.default_rng(2026)
    refs = rng.normal(size=(2, 1024))
    eeg = rng.normal(size=(8, 1024))
    model = fit_donor_field(eeg, refs)
    target = (rng.normal(size=eeg.shape) * 1e-5).astype(np.float32)
    for level in (-5., 0., 5.):
        record = generated_pair(target, model.excursions(refs), model, level)
        validate_generated_pair(record, model, level)
        with pytest.raises(ValueError, match="SNR differs"):
            validate_generated_pair(record, model, level + 1.)
        broken = {**record, "references": record["references"] * 2.}
        with pytest.raises(AssertionError):
            validate_generated_pair(broken, model, level)
