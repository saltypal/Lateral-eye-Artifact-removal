"""MWF serialization and regional output contracts, run on Kaggle only."""
import numpy as np

from vmd_eog.nonfrontal_mwf import METHODS, estimate_fields, fit_experts, load_experts, save_experts


def test_mwf_saved_replay_and_zero_reference_pass_through(tmp_path):
    rng = np.random.default_rng(42)
    time = np.arange(2048)
    references = np.stack([np.sin(time / 19.), np.cos(time / 31.)])
    eeg = rng.normal(size=(4, 2048)) * .2
    eeg[:, 1024:] += np.array([2., 1., .5, -.5])[:, None] * references[0, 1024:]
    calibration = {"eeg": eeg, "references": references, "mask": np.ones(4, bool),
                   "trial_types": np.concatenate([np.ones(1024), np.full(1024, 2)]),
                   "boundaries": np.array([1024])}
    policy = {"lags": [-10, 0, 10], "penalty": .01, "association": .4, "gain_cap": 8.}
    fitted = fit_experts(calibration, policy)
    assert not fitted.failures
    save_experts(tmp_path, fitted)
    restored = load_experts(tmp_path)
    regions, mask = np.array([0, 1, 2, 3]), np.ones(4, bool)
    args = (eeg[:, 1024:], references[:, 1024:], regions, mask)
    live, details, failures = estimate_fields(*args, fitted, policy)
    loaded, loaded_details, loaded_failures = estimate_fields(*args, restored, policy)
    for method in METHODS:
        np.testing.assert_array_equal(live[method], loaded[method])
        np.testing.assert_array_equal(live[method][[0, 3]], np.zeros((2, 1024)))
    assert details == loaded_details and failures == loaded_failures
    zero, _, _ = estimate_fields(args[0], np.zeros_like(args[1]), regions, mask, restored, policy)
    for artifact in zero.values():
        np.testing.assert_array_equal(artifact, np.zeros_like(args[0]))
