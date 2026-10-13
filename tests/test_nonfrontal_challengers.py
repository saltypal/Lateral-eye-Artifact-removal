"""Meaningful numerical contracts; executed on Kaggle, never locally."""
import numpy as np
import pytest

from vmd_eog.nonfrontal_challengers import (
    balanced_gain, large_wavelet_contribution, replace_regions, rest_whitened_projection,
)


def test_gain_and_polarity_without_target():
    values = np.sin(np.arange(1024) / 20.)[None, :]
    gain, association = balanced_gain(2.5 * values, values)
    assert gain == pytest.approx(2.5)
    assert association == pytest.approx(1.)
    assert balanced_gain(-values, values) == pytest.approx((0., -1.))
    assert balanced_gain(values, np.zeros_like(values)) == (0., 0.)


def test_gain_shape_failure():
    with pytest.raises(ValueError):
        balanced_gain(np.ones((2, 32)), np.ones((3, 32)))


@pytest.mark.parametrize("region", [(1,), (2,), (1, 2)])
def test_single_subtraction_disjoint_regions(region):
    original = np.ones((5, 128))
    frozen = original.copy()
    frozen[0] -= .3
    frozen[1] -= .1
    artifact = np.ones_like(original) * .2
    regions = np.array([0, 1, 2, 3, 2])
    mask = np.array([True, True, True, True, False])
    cleaned = replace_regions(original, frozen, artifact, regions, mask, region)
    selected = mask & np.isin(regions, region)
    np.testing.assert_array_equal(cleaned[selected], original[selected] - .2)
    np.testing.assert_array_equal(cleaned[~selected], frozen[~selected])


def test_frontal_output_and_bad_anatomy_rejected():
    with pytest.raises(ValueError):
        replace_regions(np.ones((2, 64)), np.ones((2, 64)), np.ones((2, 64)), [0, 1], [True, True], (0,))


def test_rank_projection_metric_and_span():
    rng = np.random.default_rng(42)
    driver = np.sin(np.arange(1024) / 30.)
    direction = np.array([1., -2., .5])
    field = direction[:, None] * driver
    projection, spectrum, retained = rest_whitened_projection(field, rng.normal(size=(3, 1024)), 1)
    np.testing.assert_allclose(projection @ field, field, atol=1e-10)
    np.testing.assert_allclose(projection @ projection, projection, atol=1e-10)
    assert retained == 1
    assert np.all(np.diff(spectrum) <= 0)


def test_rank_projection_zero_and_insufficient_rest():
    rng = np.random.default_rng(42)
    projection, _, _ = rest_whitened_projection(np.zeros((3, 1024)), rng.normal(size=(3, 1024)), 2)
    np.testing.assert_array_equal(projection @ np.zeros((3, 1024)), np.zeros((3, 1024)))
    with pytest.raises(ValueError):
        rest_whitened_projection(np.ones((3, 1024)), np.ones((3, 16)), 1)


def test_wavelet_no_large_coefficients_means_no_correction():
    values = np.sin(np.arange(1024) / 30.) * .001
    artifact, _ = large_wavelet_contribution(values, 100.)
    np.testing.assert_array_equal(artifact, np.zeros(1024))


def test_selective_wavelet_only_changes_supported_interval():
    values = np.zeros(1024)
    values[400:430] = 8.
    artifact, info = large_wavelet_contribution(values, 1., selective=True, dilation=10, taper=3)
    np.testing.assert_array_equal(artifact[:386], np.zeros(386))
    np.testing.assert_array_equal(artifact[444:], np.zeros(580))
    assert np.any(artifact[400:430] != 0)
    assert 0 < info["support_fraction"] < 1


@pytest.mark.parametrize("threshold", [0., -1., float("nan")])
def test_invalid_wavelet_threshold_fails(threshold):
    with pytest.raises(ValueError):
        large_wavelet_contribution(np.ones(1024), threshold)


def test_reduced_rank_adapters_boundaries_and_saved_replay(monkeypatch, tmp_path):
    from vmd_eog import nonfrontal_challengers as module
    from vmd_eog.nonfrontal_experiment import load_experts, save_experts
    from vmd_eog.posterior import ICAExpert
    time = np.arange(1024)
    refs = np.stack([np.sin(time / 19.), np.cos(time / 31.)])
    eeg = np.stack([refs[0], refs[0], np.sin(time / 7.), np.cos(time / 11.)])
    unmixing = np.array([[.5, .5, 0., 0.], [0., 0., 1., 0.]])
    expert = ICAExpert(np.zeros(4), unmixing, np.linalg.pinv(unmixing),
                       np.array([0]), np.array([[1., 0.], [0., 0.]]), True)
    monkeypatch.setattr(module, "fit_ica", lambda *args, **kwargs: expert)
    policy = {"lags": [-10, 0, 10], "penalty": .01, "association": .4,
              "gain_cap": 8., "mad_multiplier": 3., "wavelet": "sym4",
              "wavelet_level": 6, "dilation": 50, "taper": 10}
    calibration = {"eeg": eeg, "references": refs, "mask": np.ones(4, bool),
                   "trial_types": np.concatenate([np.ones(512), np.full(512, 2)]),
                   "boundaries": np.array([512])}
    fitted = module.fit_experts(calibration, policy)
    assert fitted.ica.unmixing.shape == (2, 4)
    assert fitted.component_model.calibration_valid_samples == 984
    save_experts(tmp_path, fitted)
    restored = load_experts(tmp_path)
    regions, mask = np.array([0, 1, 2, 3]), np.ones(4, bool)
    live, decisions, failures = module.estimate_fields(eeg, refs, regions, mask, fitted, policy)
    replay, replay_decisions, replay_failures = module.estimate_fields(eeg, refs, regions, mask, restored, policy)
    for method in module.METHODS:
        np.testing.assert_array_equal(live[method], replay[method])
    assert decisions == replay_decisions
    assert failures == replay_failures
    # Correct only the selected ocular source contribution. Original brain
    # sources and the ICA rank residual remain, because reconstruction is a
    # subtraction from original EEG rather than a truncated inverse ICA.
    cleaned = module.replace_regions(eeg, eeg, live["ica_rejection"], regions, mask, (1, 2))
    np.testing.assert_array_equal(cleaned[2:], eeg[2:])
    np.testing.assert_allclose(cleaned[1], 0., atol=1e-12)
    assert np.all(live["ica_rejection"][[0, 3]] == 0)
    # The frozen calibration model may have a nonzero reference baseline;
    # identically constant scoring fields must still trigger pass-through.
    zero, _, _ = module.estimate_fields(eeg, np.zeros_like(refs), regions, mask, restored, policy)
    for field in zero.values():
        np.testing.assert_array_equal(field, np.zeros_like(eeg))
