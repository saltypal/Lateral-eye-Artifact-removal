"""Regional composition and the evaluator boundary for broader validation."""
import numpy as np
import pytest

from vmd_eog.broader_frontal_review import ARMS, compose_regions, score_predictions


def test_regional_fields_subtract_once_and_unknown_masked_channels_pass():
    eeg = np.arange(40, dtype=float).reshape(5, 8)
    regions = np.array([0, 1, 2, -1, 0])
    mask = np.array([True, True, True, True, False])
    frontal, posterior = np.zeros_like(eeg), np.zeros_like(eeg)
    frontal[0] = 1
    posterior[1:3] = 2
    cleaned = compose_regions(eeg, frontal, posterior, regions, mask)
    np.testing.assert_array_equal(cleaned[0], eeg[0] - 1)
    np.testing.assert_array_equal(cleaned[1:3], eeg[1:3] - 2)
    np.testing.assert_array_equal(cleaned[3:], eeg[3:])
    for index in (0, 3, 4):
        leaked = posterior.copy()
        leaked[index] = 1
        with pytest.raises(ValueError, match="unavailable anatomy"):
            compose_regions(eeg, frontal, leaked, regions, mask)


def test_evaluation_preserves_all_arms_and_marks_failed_frontal_channels():
    samples = np.arange(1024) / 200
    target = np.stack([np.sin(2 * np.pi * (8 + channel) * samples) for channel in range(3)])
    original = target + .2 * np.sin(2 * np.pi * samples)[None]
    stored = {"eeg": original, "paired_reference": target,
              "channel_names": np.array(["Fp1", "P3", "C3"]),
              "regions": np.array([0, 1, 2]), "mask": np.ones(3, dtype=bool)}
    predictions = {arm: original.copy() for arm in ARMS}
    row = {"example_id": "fixture", "recipient": "r", "donor": "d", "condition": "lateral",
           "input_snr_db": 0., "partition": {"fold": 0},
           "pilot_recipient_exposed": False, "pilot_donor_exposed": True}
    config = {"fs": 200, "preservation": {"pearson_deterioration": .005}}
    records, gates = score_predictions(stored, predictions, row, config, {0})
    assert {record["arm"] for record in records} == set(ARMS)
    assert {record["scope"] for record in records} == {"whole", "frontal", "posterior", "central_shared"}
    assert all(record["channel"] == "Fp1" and record["arm"].startswith("vmd")
               for record in records if record["failure"])
    assert sum(bool(record["failure"]) for record in records) == 4
    assert all(record["failure"] is None or isinstance(record["failure"], str) for record in records)
    assert all(gate["cc_pass"] for gate in gates)
    assert all(gate["clean_checks"] is None for gate in gates)
    # Scoring cannot overwrite the saved predictions.
    for prediction in predictions.values():
        np.testing.assert_array_equal(prediction, original)
