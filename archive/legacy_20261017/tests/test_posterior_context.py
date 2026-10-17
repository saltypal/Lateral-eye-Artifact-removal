"""Context polarity, stable layouts and one-subtraction contracts (Kaggle only)."""
import numpy as np
import pytest

from vmd_eog.posterior_context import context_input, compose_correction, select_examples
from vmd_eog.reference import correlations


def montage():
    time = np.arange(400)/200
    lateral = np.sin(2*np.pi*time)
    return {"eeg": np.stack([lateral, -lateral, .1*lateral, -.1*lateral]),
            "channel_names": np.array(["F3", "F4", "P3", "P4"]),
            "regions": np.array([0, 0, 1, 1]), "hemispheres": np.array([0, 1, 0, 1]),
            "mask": np.ones(4, dtype=bool)}


def test_signed_context_retains_opposite_lateral_activity():
    data = montage()
    common, _, _, info = context_input(data, "common_context")
    signed, _, _, signed_info = context_input(data, "signed_context")
    np.testing.assert_allclose(common[-1], 0., atol=1e-15)
    lateral_row = signed_info["layout"].index("context:lateral")
    np.testing.assert_allclose(signed[lateral_row], data["eeg"][1]-data["eeg"][0])
    assert signed_info["frontal_available"]["lateral"]
    assert not info["frontal_available"]["midline"]


def test_context_layout_does_not_depend_on_scored_window_variance():
    data = montage()
    _, _, _, original_info = context_input(data, "signed_context")
    data["eeg"] = np.zeros_like(data["eeg"])
    _, _, _, zero_info = context_input(data, "signed_context")
    assert zero_info == original_info
    data["mask"][1] = False
    values, _, ids, info = context_input(data, "signed_context")
    assert not info["frontal_available"]["lateral"]
    assert "context:lateral" not in info["layout"]
    assert np.isfinite(values).all() and len(ids) == 2
    data["mask"][:2] = False
    values, _, _, info = context_input(data, "signed_context")
    assert values.shape == (2, 400)
    assert not any(info["frontal_available"].values())


def test_channel_permutation_preserves_named_signed_context():
    data = montage()
    original, _, _, original_info = context_input(data, "signed_context")
    order = np.array([3, 1, 2, 0])
    permuted = {key: value[order] for key, value in data.items()}
    transformed, _, _, info = context_input(permuted, "signed_context")
    for row, name in enumerate(original_info["layout"]):
        np.testing.assert_allclose(original[row], transformed[info["layout"].index(name)])


def test_corrections_compose_once_and_reject_overlapping_outputs():
    original = montage()["eeg"]
    front = np.zeros_like(original)
    front[:2] = .25*original[:2]
    posterior = .5*original[2:]
    cleaned, artifact = compose_correction(original, [2, 3], posterior, front)
    np.testing.assert_allclose(cleaned[:2], .75*original[:2])
    np.testing.assert_allclose(cleaned[2:], .5*original[2:])
    np.testing.assert_allclose(cleaned+artifact, original)
    front[2] = 1.
    with pytest.raises(ValueError, match="overlap"):
        compose_correction(original, [2, 3], posterior, front)


def test_recipient_donor_source_buckets_are_checked_before_sampling():
    rows = [{"example_id": f"e{fold}-{condition}", "target_kind": "controlled_recipient_reference",
             "recipient": f"r{fold}", "donor": f"d{fold}", "condition": condition,
             "input_snr_db": None if condition == "clean" else 0.,
             "partition": {"role": "development", "fold": fold}}
            for fold in range(5) for condition in ("clean", "blink", "lateral", "mixed")]
    assert len(select_examples(rows, "pilot")) == 20
    rows[-1]["donor"] = "d0"
    with pytest.raises(ValueError, match="source buckets"):
        select_examples(rows, "pilot")


def test_lagged_association_excludes_trial_joins():
    values = np.array([[0., 0., 9., 1., 1., 1.]])
    references = np.array([[0., 0., 0., 9., 1., 1.]])
    actual = correlations(values, references, max_lag=1, boundaries=(3,))
    candidates = []
    for lag in (-1, 0, 1):
        pairs = [(index, index-lag) for index in range(6)
                 if 0 <= index-lag < 6 and (index < 3) == (index-lag < 3)]
        x = np.array([values[0, i] for i, _ in pairs])
        y = np.array([references[0, j] for _, j in pairs])
        candidates.append(np.corrcoef(x, y)[0, 1])
    expected = max(candidates, key=abs)
    np.testing.assert_allclose(actual, [[expected]], atol=1e-12)
