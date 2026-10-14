import numpy as np
import pytest

from vmd_eog.nonfrontal_review import composed_failure_provenance, inherited_failure, pooled_snr, swap_frontend, review_recipes, reconcile_review_baselines


def test_frontend_main_region_parity_ignores_only_declared_diagnostic_scopes():
    import pandas as pd
    from vmd_eog.posterior_context import PAPER_KEYS
    row = {"example_id": "one", "scope": "whole", "channel": "F3", "evaluated_family": "identity",
           "recipient": "recipient", "donor": "donor", "condition": "blink", "input_snr_db": 0.,
           **{key: .5 for key in PAPER_KEYS}}
    current = pd.DataFrame([row])
    parent = pd.DataFrame([row, {**row, "scope": "middle_verified", "channel": "C3"}])
    assert reconcile_review_baselines(current, parent) == len(PAPER_KEYS)
    missing_main = pd.DataFrame([row, {**row, "channel": "F4"}])
    with pytest.raises(ValueError, match="cohort differs"):
        reconcile_review_baselines(current, missing_main)


def test_frozen_review_declares_the_strongest_new_frontend_control():
    old, comparator = review_recipes({})
    assert comparator == "raw_frontal+reference_rank2"
    current = ("ridge_direct:both", "cca_reference_rank2:both", "reference_rank2_control:both")
    assert review_recipes({"review_families": list(current)}) == (current, "raw_frontal+ridge_direct")
    with pytest.raises(ValueError, match="recipes differ"):
        review_recipes({"review_families": ["cca_reference_rank1:both"]})


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


def test_replaced_vmd_failure_is_historical_in_raw_frontend_composition():
    report = composed_failure_provenance("VMD failed", None, None)
    assert report["failure"] is None
    assert report["historical_parent_failures"] == "VMD failed"
    assert "not independently separable" in report["active_raw_frontend_failure_status"]
    report = composed_failure_provenance("VMD failed", "ICA failed", "raw parent flag")
    assert report["failure"] == "ICA failed; raw parent flag"
