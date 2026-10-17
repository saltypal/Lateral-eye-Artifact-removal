import numpy as np
import pytest
from vmd_eog.reference import lag_matrix, project, correlations, signed_context
from vmd_eog.signal import overlap_add_correction, calibration_trials
from vmd_eog.metrics import paired
from vmd_eog.vmd import rolling_decompose
from vmd_eog.contracts import RunSpec


def test_lag_positive_leading_negative_trailing_without_wrap():
    references = np.array([[1., 2., 3., 4., 5.]])
    positive, valid = lag_matrix(references, (2,))
    np.testing.assert_array_equal(positive, [[0, 0, 1, 2, 3]])
    np.testing.assert_array_equal(valid, [False, False, True, True, True])
    negative, valid = lag_matrix(references, (-2,))
    np.testing.assert_array_equal(negative, [[3, 4, 5, 0, 0]])
    np.testing.assert_array_equal(valid, [True, True, True, False, False])


def test_lagged_correlation_is_per_channel_and_axis():
    rng = np.random.default_rng(4)
    refs = rng.normal(size=(2, 400))
    eeg = np.stack([np.roll(refs[0], 3), np.roll(refs[1], -2)])
    result = correlations(eeg, refs, 5)
    assert abs(result[0, 0]) > .99 and abs(result[1, 1]) > .99
    assert abs(result[0, 1]) < .4 and abs(result[1, 0]) < .4


def test_projection_sign_and_collinear_references():
    t = np.arange(1024) / 200
    refs = np.stack([np.sin(2*np.pi*2*t), np.sin(2*np.pi*3*t)])
    eeg = 2 * refs[:1] + .1 * np.sin(2*np.pi*12*t)
    a, _ = project(eeg, refs, penalty=1e-4)
    b, _ = project(eeg, -refs, penalty=1e-4)
    np.testing.assert_allclose(a, b, atol=1e-10)
    c, _ = project(eeg, np.stack([refs[0], refs[0]]))
    assert np.isfinite(c).all()


@pytest.mark.parametrize("length", [128, 129])
def test_vmd_normalized_reference_parity_and_units(length):
    from vmdpy import VMD
    t = np.arange(length) / 200
    raw = np.sin(2*np.pi*4*t) + .25*np.sin(2*np.pi*18*t)
    modes, residual, info = rolling_decompose(raw, modes=3, alpha=1000, max_iterations=500)
    np.testing.assert_allclose(modes.sum(0) + residual, raw, atol=2e-7)
    scaled, _, _ = rolling_decompose(raw*1e-6, modes=3, alpha=1000, max_iterations=500)
    np.testing.assert_allclose(scaled/1e-6, modes, atol=2e-6)
    padded = np.pad(raw, (0, length % 2), mode="edge")
    scale = np.sqrt(np.mean(padded**2))
    expected, _, centers = VMD(padded/scale, 1000, 0, 3, 0, 1, 1e-6)
    expected = expected[np.argsort(centers[-1])][:, :length]*scale
    np.testing.assert_allclose(modes, expected, atol=3e-5, rtol=3e-4)


def test_overlap_add_identity_all_samples():
    raw = np.random.default_rng(3).normal(size=(3, 2300))
    cleaned, artifact = overlap_add_correction(raw, lambda window: np.zeros_like(window))
    np.testing.assert_array_equal(cleaned, raw)
    assert artifact.shape == raw.shape


def test_metrics_perfect_zero_and_waveform_error():
    raw = np.random.default_rng(1).normal(size=(2, 1024))
    assert paired(raw, raw)["snr_status"] == "perfect_reconstruction"
    assert paired(raw, np.zeros_like(raw))["snr_status"] == "zero_target_energy"
    result = paired(raw+.1, raw)
    assert result["snr_db"] > 10


def test_calibration_never_uses_scoring_samples():
    raw = np.ones((2, 6000))
    calibration, scoring, error = calibration_trials([raw], 200)
    assert error is None and calibration[0].shape[-1] == 2000
    assert scoring[0].shape[-1] == 4000


def test_signed_context_does_not_cancel_lateral_evidence():
    eeg = np.stack([np.ones(20), -np.ones(20), np.zeros(20)])
    context = signed_context(eeg, ["F3", "F4", "Pz"], [0,0,1], [0,1,2])
    np.testing.assert_array_equal(context["lateral"], -2*np.ones(20))
    np.testing.assert_array_equal(context["common"], np.zeros(20))


def test_run_spec_unsafe_id_rejected():
    with pytest.raises(ValueError):
        RunSpec("../unsafe", "contracts", "a"*40).validate()
