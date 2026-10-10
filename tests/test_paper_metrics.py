"""Analytical paper-equation fixtures, executed on Kaggle only."""
import numpy as np
import pytest
from vmd_eog.paper_metrics import (
    paired_channels, psnr_channels, native_condition_channels,
    resting_spectrum, paired_permutation,
)


def test_channel_snr_is_not_pooled_and_does_not_remove_offset_error():
    target = np.array([[-1., 1., -1., 1.], [-10., 10., -10., 10.]])
    cleaned = target+1.
    metrics = paired_channels(cleaned, target)
    np.testing.assert_allclose(metrics["snr_energy_db"], [0., 20.])
    np.testing.assert_allclose(metrics["rrmse_time"], [1., .1])
    np.testing.assert_allclose(metrics["mse"], [1., 1.])
    assert metrics["snr_energy_db"].mean() == 10.
    assert not np.isclose(10*np.log10(50.5), 10.)


def test_literal_vmd_printed_cc_discrepancy_is_visible():
    target = np.array([-2., -1., 1., 2.])
    result = paired_channels(2*target, target)
    np.testing.assert_allclose(result["pearson_cc"], [1.])
    np.testing.assert_allclose(result["ivmd_eq15_printed_cov_over_target_var"], [2.])


def test_perfect_and_constant_metrics_remain_undefined_or_infinite():
    target = np.array([-1., 1., -1., 1.])
    assert np.isposinf(paired_channels(target, target)["snr_energy_db"][0])
    result = paired_channels(np.zeros(4), np.zeros(4))
    assert np.isnan(result["snr_energy_db"][0])
    assert np.isnan(result["pearson_cc"][0])
    with pytest.raises(ValueError):
        paired_channels(np.zeros((4, 2)), np.zeros((2, 4)))


def test_native_condition_uses_zero_lag_and_correct_reference():
    heog = np.array([1., 0., -1., 0., 1., 0., -1., 0.])
    veog = np.roll(heog, 1)
    refs = np.stack([heog, veog])
    lateral = native_condition_channels(veog, veog, refs, "lateral")
    blink = native_condition_channels(veog, veog, refs, "blink")
    np.testing.assert_allclose(lateral["eeg_eog_abs_r_after"], [0.], atol=1e-14)
    np.testing.assert_allclose(blink["eeg_eog_abs_r_after"], [1.])
    with pytest.raises(ValueError):
        native_condition_channels(veog, veog, refs, "mixed")


def test_psnr_requires_scale_and_is_not_reconstruction_snr():
    target = np.array([-1., 1., -1., 1.])
    psnr = psnr_channels(target+1, target, peak=255., scale_description="fixture only: 8-bit scale")
    np.testing.assert_allclose(psnr, [20*np.log10(255)])
    with pytest.raises(ValueError):
        psnr_channels(target, target, peak=255., scale_description="")


def test_resting_welch_bands_and_relative_power_identity():
    time = np.arange(1600)/200
    signal = np.sin(2*np.pi*10*time)
    result = resting_spectrum(signal*.5, signal, 200, normalization_band=(1., 40.))
    assert len(result["frequencies_hz"]) == 201
    assert result["alpha_relative_before"][0] > .999
    np.testing.assert_allclose(result["alpha_relative_change"], [0.], atol=1e-14)
    assert np.any(result["ivmd_eq19_absolute_psd_change"] > 0)
    with pytest.raises(ValueError):
        resting_spectrum(signal[:200], signal[:200], 200, normalization_band=(1., 40.))


def test_paired_permutation_identical_participants_and_missing_rejection():
    result = paired_permutation([1., 2., 3.], [1., 2., 3.], comparisons=5)
    assert result["p_two_sided"] == result["p_bonferroni"] == 1.
    assert result["permutations"] == 10000
    with pytest.raises(ValueError):
        paired_permutation([1., np.nan], [1., 2.], comparisons=1)
