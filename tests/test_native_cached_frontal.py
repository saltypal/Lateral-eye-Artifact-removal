"""Native input reuse and inference boundaries; execute numerically on Kaggle."""
import copy

import numpy as np
import pytest

from vmd_eog.native_cached_frontal import METHODS, predict_window, validate_record, validate_trial
from vmd_eog.native_protocol import multi_overlap_add


def record():
    return {"record_id": "study01_p02_prep", "dataset": "osf",
            "partition": {"role": "development", "fold": 4},
            "calibration_trial_ids": [0, 1], "scoring_trial_ids": [2, 3, 4, 5],
            "trial_labels": [1, 2, 1, 2, 3, 4], "channel_names": ["Fp1", "Fp2", "P3", "C3"]}


def signals():
    rng = np.random.default_rng(42)
    refs = rng.normal(size=(2, 512))
    eeg = np.array([[2., 1.], [-2., 1.], [.2, .1], [.1, .2]]) @ refs + .1 * rng.normal(size=(4, 512))
    names, regions = np.array(record()["channel_names"]), np.array([0, 0, 1, 2])
    calibration = {"eeg": eeg.copy(), "references": refs.copy(), "boundaries": np.array([256]),
                   "channel_names": names, "regions": regions}
    stored = {"original": eeg, "identity": eeg.copy(), "inference_references": refs,
              "evaluation_references": refs * 100, "channel_names": names, "regions": regions}
    return stored, calibration


def test_native_record_exact_identity_and_trial_disjointness():
    row = record()
    validate_record(row, copy.deepcopy(row))
    altered = copy.deepcopy(row)
    altered["partition"]["fold"] = 0
    with pytest.raises(ValueError, match="ledgers differ"):
        validate_record(row, altered)
    row["calibration_trial_ids"].append(2)
    with pytest.raises(ValueError, match="identities are ineligible"):
        validate_record(row, copy.deepcopy(row))


@pytest.mark.parametrize("change", ["reserved", "duplicate", "missing_condition"])
def test_native_rejects_reserved_duplicate_and_incomplete_condition_coverage(change):
    row = record()
    if change == "reserved":
        row["partition"]["role"] = "confirmation"
    elif change == "duplicate":
        row["scoring_trial_ids"].append(2)
    else:
        row["trial_labels"][5] = 1
    with pytest.raises(ValueError):
        validate_record(row, copy.deepcopy(row))


def test_native_trial_requires_original_identity_reference_alignment_and_names():
    stored, calibration = signals()
    original = validate_trial(stored, calibration, record())[0]
    np.testing.assert_array_equal(original, stored["identity"])
    altered = copy.deepcopy(stored)
    altered["evaluation_references"] = altered["evaluation_references"][:, :-1]
    with pytest.raises(ValueError, match="axes"):
        validate_trial(altered, calibration, record())
    altered = copy.deepcopy(stored)
    altered["identity"][0, 0] += 1
    with pytest.raises(AssertionError):
        validate_trial(altered, calibration, record())
    altered = copy.deepcopy(stored)
    altered["regions"][2] = 0
    calibration = {**calibration, "regions": altered["regions"]}
    with pytest.raises(ValueError, match="verified electrode"):
        validate_trial(altered, calibration, record())


def test_native_window_fallback_and_nonfrontal_pass_through():
    stored, calibration = signals()
    eeg, refs, _, names, regions, mask = validate_trial(stored, calibration, record())
    bundles = {index: (np.stack([eeg[index] * .8, eeg[index] * .2, eeg[index] * 0]),
                       eeg[index] * 0, {"hit_iteration_limit": index == 0}) for index in (0, 1)}
    artifacts, _, failed = predict_window(eeg, refs, calibration, names, regions, mask, bundles)
    assert set(artifacts) == set(METHODS) and failed == [0]
    np.testing.assert_array_equal(artifacts["vmd_anchored_raw_fallback"][0], artifacts["raw_shared"][0])
    np.testing.assert_array_equal(artifacts["vmd_anchored_raw_fallback"][1:], artifacts["vmd_anchored"][1:])
    for values in artifacts.values():
        np.testing.assert_array_equal(values[2:], 0.)


@pytest.mark.parametrize("length", [401, 1600])
def test_native_full_trial_overlap_add_identity_and_single_subtraction(length):
    eeg = np.arange(2 * length, dtype=float).reshape(2, length)
    refs = np.zeros((2, length))
    output = multi_overlap_add(eeg, refs, lambda values, _: {"identity": values * 0, "fixed": values * .2},
                               methods=("identity", "fixed"))
    np.testing.assert_array_equal(output["identity"], eeg)
    np.testing.assert_allclose(output["fixed"], .8 * eeg, rtol=1e-14, atol=1e-12)
