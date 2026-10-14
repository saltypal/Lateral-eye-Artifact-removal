import numpy as np
import pytest

from vmd_eog.nonfrontal_review import inherited_failure, pooled_snr, swap_frontend


def test_frontend_swap_composes_once_and_preserves_other_regions():
    original = np.ones((5, 128))
    candidate = original.copy()
    candidate[:3] -= np.array([.1, .2, .3])[:, None]
    frontal = original.copy()
    frontal[0] -= .4
    regions, mask = np.array([0, 1, 2, 3, 2]), np.array([1, 1, 1, 1, 0], bool)
    output = swap_frontend(original, candidate, frontal, regions, mask)
    np.testing.assert_allclose(output[0], .6)
    np.testing.assert_array_equal(output[1:3], candidate[1:3])
    np.testing.assert_array_equal(output[3:], original[3:])
    bad = candidate.copy(); bad[3] -= .1
    with pytest.raises(ValueError, match="pass-through"):
        swap_frontend(original, bad, frontal, regions, mask)


def test_pooled_snr_is_not_mean_channel_db():
    target = np.array([[10., 10.], [1., 1.]])
    prediction = target - 1.
    expected = 10 * np.log10(202. / 4.)
    assert pooled_snr(prediction, target) == pytest.approx(expected)
    assert pooled_snr(prediction, target) != pytest.approx(10.)


def test_failure_lineage_is_specific_to_parent_arm():
    import pandas as pd
    scores = pd.DataFrame({"example_id": ["one", "one"], "scope": ["whole", "whole"],
                          "evaluated_family": ["ridge", "rank2"], "failure": ["ICA failed", None]})
    assert inherited_failure(scores, "one", "ridge") == "ICA failed"
    assert inherited_failure(scores, "one", "rank2") is None
    with pytest.raises(ValueError, match="Missing"):
        inherited_failure(scores, "one", "missing")
