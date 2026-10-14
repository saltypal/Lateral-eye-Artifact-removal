import numpy as np
import pytest

from vmd_eog.nonfrontal_cca import (METHODS, covariance_roots, estimate_fields, fit_experts,
                                   load_experts, reference_projection, save_experts)


def fixture():
    rng = np.random.default_rng(42)
    time = np.arange(4096)
    refs = np.stack([np.sin(time / 17.), np.cos(time / 31.)])
    eeg = rng.normal(size=(5, 4096)) * .2
    eeg += np.array([2., 1., .5, -.5, 0.])[:, None] * refs[0]
    eeg += np.array([.2, .6, -.7, 0., .5])[:, None] * refs[1]
    calibration = {"eeg": eeg, "references": refs, "mask": np.ones(5, bool),
                   "trial_types": np.concatenate([np.ones(2048), np.full(2048, 2)]),
                   "boundaries": np.array([2048])}
    policy = {"lags": [-10, 0, 10], "penalty": .01, "association": .4, "gain_cap": 8.,
              "cca_regularization": .01, "cca_r2_threshold": .16}
    return calibration, policy


def test_full_reference_projection_collapses_to_matched_ridge():
    cal, policy = fixture()
    fitted = fit_experts(cal, policy)
    p = reference_projection(fitted.reference_root, fitted.reference_inverse,
                             fitted.reference_directions, np.arange(6))
    np.testing.assert_allclose(p, np.eye(6), rtol=1e-10, atol=1e-10)
    design, valid = fitted.field_model._standardized_design(cal["references"])
    full = fitted.field_model.coefficients.T @ p @ design
    full[:, ~valid] = 0
    np.testing.assert_allclose(full, fitted.field_model.artifact(cal["references"]), rtol=1e-10, atol=1e-12)


def test_cca_replay_and_nonfrontal_mask_contract(tmp_path):
    cal, policy = fixture()
    fitted = fit_experts(cal, policy)
    assert not fitted.failures
    save_experts(tmp_path, fitted)
    loaded = load_experts(tmp_path)
    regions, mask = np.array([0, 1, 2, 3, 1]), np.ones(5, bool)
    args = (cal["eeg"][:, 2048:], cal["references"][:, 2048:], regions, mask)
    live, details, failures = estimate_fields(*args, fitted, policy)
    saved, saved_details, saved_failures = estimate_fields(*args, loaded, policy)
    for method in METHODS:
        np.testing.assert_array_equal(live[method], saved[method])
        np.testing.assert_array_equal(live[method][[0, 3]], np.zeros((2, 2048)))
        np.testing.assert_array_equal(live[method][:, :10], np.zeros((5, 10)))
        np.testing.assert_array_equal(live[method][:, -10:], np.zeros((5, 10)))
    assert details == saved_details and failures == saved_failures
    zero, _, _ = estimate_fields(args[0], np.zeros_like(args[1]), regions, mask, loaded, policy)
    for artifact in zero.values():
        np.testing.assert_array_equal(artifact, np.zeros_like(args[0]))


def test_cca_fit_is_boundary_and_channel_permutation_consistent():
    cal, policy = fixture()
    original = fit_experts(cal, policy)
    order = np.array([4, 2, 0, 3, 1])
    changed = {**cal, "eeg": cal["eeg"][order], "mask": cal["mask"][order]}
    permuted = fit_experts(changed, policy)
    regions, mask = np.array([0, 1, 2, 3, 1]), cal["mask"]
    before, _, _ = estimate_fields(cal["eeg"][:, 2048:], cal["references"][:, 2048:], regions, mask, original, policy)
    after, _, _ = estimate_fields(changed["eeg"][:, 2048:], cal["references"][:, 2048:], regions[order], mask, permuted, policy)
    for method in METHODS:
        np.testing.assert_allclose(after[method], before[method][order], rtol=1e-8, atol=1e-10)
    assert original.field_model.calibration_valid_samples == 4056


def test_cca_reference_scale_invariance_and_constant_reference_fallback():
    cal, policy = fixture()
    fitted = fit_experts(cal, policy)
    rescaled = fit_experts({**cal, "references": cal["references"] * 1e-6}, policy)
    regions, mask = np.array([0, 1, 2, 3, 1]), cal["mask"]
    before, _, _ = estimate_fields(cal["eeg"][:, 2048:], cal["references"][:, 2048:], regions, mask, fitted, policy)
    after, _, _ = estimate_fields(cal["eeg"][:, 2048:], cal["references"][:, 2048:] * 1e-6, regions, mask, rescaled, policy)
    for method in METHODS:
        np.testing.assert_allclose(after[method], before[method], rtol=1e-8, atol=1e-10)
    empty = fit_experts({**cal, "references": np.zeros_like(cal["references"])}, policy)
    assert "cca_reference_rank1" in empty.failures


def test_invalid_cca_family_or_threshold_cannot_silently_launch():
    from vmd_eog.nonfrontal_experiment import validate_family
    with pytest.raises(ValueError):
        validate_family({"scope": "fixed_nonfrontal_cca_adaptation", "method_family": "mwf_controls"})
    cal, policy = fixture()
    with pytest.raises(ValueError, match="policy"):
        fit_experts(cal, {**policy, "cca_r2_threshold": 1.1})
