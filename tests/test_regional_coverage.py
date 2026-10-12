"""Numerical contracts are executed by the Kaggle runner only."""
import inspect

import numpy as np
import pytest

from vmd_eog.regional_coverage import (BASE, COMPARATOR, central_compose,
                                      estimate_central, verify_arrays, select_rows)


def fixture_arrays():
    original = np.arange(5 * 1024, dtype=float).reshape(5, 1024)
    names = np.asarray(["Fp1", "Pz", "Cz", "unknown", "C3"])
    regions = np.asarray([0, 1, 2, 3, 2])
    mask = np.asarray([True, True, True, True, False])
    hemispheres = np.asarray([0, 2, 2, 3, 0])
    data = {"eeg": original, "paired_reference": original.copy(), "channel_names": names,
            "regions": regions, "mask": mask, "hemispheres": hemispheres}
    cal = {key: value.copy() for key, value in data.items() if key != "paired_reference"}
    baseline = original.copy()
    baseline[0] -= 2
    baseline[1] -= 3
    saved = {"original": original.copy(), "target": original.copy(), "names": names,
             "regions": regions, "mask": mask, BASE: baseline, COMPARATOR: baseline.copy()}
    return saved, data, cal


def test_central_compose_preserves_frozen_and_unknown():
    saved, data, cal = fixture_arrays()
    _, groups = verify_arrays(saved, data, cal)
    result = central_compose(data["eeg"], saved[BASE], groups["central_shared"], np.ones((1, 1024)))
    assert np.array_equal(result[[0, 1, 3, 4]], saved[BASE][[0, 1, 3, 4]])
    assert np.array_equal(result[2], data["eeg"][2] - 1)


@pytest.mark.parametrize("mutation", ["target", "central", "masked", "anatomy", "hemisphere"])
def test_parent_mismatch_fails_closed(mutation):
    saved, data, cal = fixture_arrays()
    if mutation == "target":
        saved["target"][0, 0] += 1
    elif mutation == "central":
        saved[BASE][2, 0] -= 1
    elif mutation == "masked":
        saved[BASE][4, 0] -= 1
    elif mutation == "anatomy":
        data["channel_names"] = np.asarray(["Fp1", "Pz", "invalid", "unknown", "C3"])
        saved["names"] = data["channel_names"]
        cal["channel_names"] = data["channel_names"]
    else:
        cal["hemispheres"][0] = 1
    with pytest.raises(ValueError):
        verify_arrays(saved, data, cal)


def test_composition_rejects_overlap_and_nonfinite():
    original = np.zeros((3, 1024))
    frozen = original.copy()
    frozen[0] = 1
    with pytest.raises(ValueError):
        central_compose(original, frozen, [0], np.zeros((1, 1024)))
    with pytest.raises(ValueError):
        central_compose(original, original, [1, 1], np.zeros((2, 1024)))
    with pytest.raises(ValueError):
        central_compose(original, original, [1], np.full((1, 1024), np.nan))


def test_parent_permutation_with_metadata():
    saved, data, cal = fixture_arrays()
    order = np.asarray([3, 2, 4, 1, 0])
    perm_saved = {key: value[order] for key, value in saved.items()}
    perm_data = {key: value[order] for key, value in data.items()}
    perm_cal = {key: value[order] for key, value in cal.items()}
    available, groups = verify_arrays(perm_saved, perm_data, perm_cal)
    assert available.tolist() == [0, 1, 3, 4]
    assert groups["central_shared"].tolist() == [1]


def test_inference_has_no_scoring_target_or_eeg_fit_argument():
    parameters = set(inspect.signature(estimate_central).parameters)
    assert "eeg" not in parameters
    assert "target" not in parameters
    assert "condition" not in parameters
    assert "amplitude" not in parameters


def test_pilot_retains_every_condition_and_input_level():
    rows = []
    for recipient in range(32):
        for example in range(10):
            rows.append({"example_id": f"{recipient}-{example}", "recipient": str(recipient),
                         "partition": {"role": "development", "fold": recipient % 5}})
    result = select_rows(rows, "pilot")
    assert len(result) == 50
    assert len({row["recipient"] for row in result}) == 5
    bad = [dict(row, partition={"role": "confirmation", "fold": 0}) for row in rows]
    with pytest.raises(ValueError):
        select_rows(bad, "full")
