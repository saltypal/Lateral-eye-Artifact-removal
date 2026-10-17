"""Meaningful numerical contracts; executed only inside the Kaggle notebook."""
import numpy as np
import pytest
from eog_clean.data import region, calibration_split, osf_sessions, prepare_sources
from eog_clean.frontal import overlap_add, window_starts, combine
from eog_clean.metrics import paired, correlations, preservation
from eog_clean.methods import fit_ridge, fit_mwf
from eog_clean.vmd import rolling_decompose


def test_known_mixture_regression_and_signed_horizontal_polarity():
    t = np.arange(2000)/200
    eyes = np.stack([np.sin(2*np.pi*t), np.cos(4*np.pi*t)])
    neural = np.stack([np.sin(34*np.pi*t), np.cos(40*np.pi*t)])
    field = np.array([[2., 1.], [-2., .5]])@eyes
    dirty = neural+field
    fit = fit_ridge(dirty, eyes, penalty=1e-9)
    np.testing.assert_allclose(dirty-fit.artifact(dirty, eyes), neural, atol=1e-7)
    assert fit.coefficients[0, 0] > 0 and fit.coefficients[1, 0] < 0


def test_zero_reference_fails_instead_of_nan():
    with pytest.raises(ValueError, match="nonzero"):
        fit_ridge(np.ones((2, 128)), np.zeros((2, 128)))


def test_pearson_is_not_scale_sensitive_snr():
    target = np.sin(np.arange(200)/10)[None]
    score = paired(2*target+3, target)
    assert score["pearson_cc"][0] == pytest.approx(1.)
    assert score["snr_out_db"][0] < 0
    assert paired(target, target)["snr_out_db"][0] == np.inf


def test_correlation_separate_horizontal_and_vertical():
    t = np.arange(1024)/1024
    eyes = np.stack([np.sin(2*np.pi*t), np.cos(2*np.pi*t)])
    result = correlations(np.stack([-eyes[0], eyes[1]]), eyes)
    np.testing.assert_allclose(result, [[-1, 0], [0, 1]], atol=1e-12)


def test_identity_has_zero_preservation_error():
    target = np.random.default_rng(1).normal(size=(4, 1024))
    scores = preservation(target, target)
    assert all(value == 0 for value in scores.values())


@pytest.mark.parametrize("samples", [1024, 1025, 1536, 2049, 3200])
def test_overlap_add_has_exact_identity_and_complete_boundary_coverage(samples):
    target = np.random.default_rng(samples).normal(size=(3, samples))
    starts = window_starts(samples)
    windows = np.stack([target[:, start:start+1024] for start in starts])
    np.testing.assert_allclose(overlap_add(windows, starts, samples), target, atol=1e-12)


def test_fixed_vmd_retains_reconstruction_residual():
    t = np.arange(1024)/200
    signal = np.sin(2*np.pi*2*t)+.2*np.cos(2*np.pi*17*t)
    modes, residual, diagnostic = rolling_decompose(signal, 3, 2000, max_iterations=2000)
    assert modes.shape == (3, 1024)
    np.testing.assert_allclose(modes.sum(axis=0)+residual, signal, atol=4e-7)
    assert np.diff(diagnostic["centers_hz"]).min() >= 0


def test_no_anatomy_is_inferred_from_unknown_labels():
    assert [region(name) for name in ["Fp1", "Fz", "FC1", "Cz", "CP2", "POz", "O2", "row0", "EOG1"]] == [
        "frontal", "frontal", "central", "central", "central", "posterior", "posterior", "unknown", "unknown"]


def test_regional_composition_is_one_subtraction_and_keeps_unknown():
    raw = np.ones((4, 16))
    prediction, artifact = combine(raw, raw*.2, raw*.3, ["frontal", "posterior", "central", "unknown"])
    np.testing.assert_allclose(prediction[:, 0], [.8, .7, .7, 1.])
    np.testing.assert_allclose(raw-prediction, artifact)


def test_whole_trials_calibration_never_overlap_scoring():
    trials = [{"index": i, "condition": i%4+1, "block": i//4+1} for i in range(8)]
    calibration, scoring, policy = calibration_split(trials)
    assert [t["index"] for t in calibration] == [0, 1, 2, 3]
    assert [t["index"] for t in scoring] == [4, 5, 6, 7]
    assert "block" in policy


def test_mwf_suppresses_excess_variance_and_is_channel_permutation_equivariant():
    generator = np.random.default_rng(42)
    rest = generator.normal(size=(3, 8000))
    ocular = generator.normal(size=(3, 8000))*np.array([[5], [1], [1]])
    model = fit_mwf(ocular, rest, rank=1, strength=1)
    cleaned = ocular-model.artifact(ocular)
    assert np.std(cleaned[0]) < .2*np.std(ocular[0])
    order = [2, 0, 1]
    permuted = fit_mwf(ocular[order], rest[order], rank=1, strength=1)
    np.testing.assert_allclose(permuted.artifact(ocular[order]), model.artifact(ocular)[order], atol=1e-9)


def test_historical_model_starts_at_identity_and_requires_eog():
    import torch
    from eog_clean.historical_model import EOGContextGainStudent
    model = EOGContextGainStudent()
    assert sum(parameter.numel() for parameter in model.parameters()) == 2499
    raw = torch.randn(2, 3, 1024)
    modes = torch.randn(2, 3, 3, 1024)
    references = torch.randn(2, 2, 1024)
    result = model(raw, modes, references, torch.ones(2, 3), torch.full((2, 3), 3))
    assert torch.count_nonzero(result["artifact"]) == 0
    with pytest.raises(ValueError, match="HEOG"):
        model(raw, modes, references[:, :1], torch.ones(2, 3), torch.full((2, 3), 3))


def test_confirmation_sources_are_excluded_before_waveform_loading(tmp_path):
    folder = tmp_path/"Dataset1_OSF/study01"
    folder.mkdir(parents=True)
    for name in ("p01", "p02"):
        (folder/f"study01_{name}_prep.set").touch()
    partition = {"osf:p01": {"role": "confirmation"}, "osf:p02": {"role": "development"}}
    selected = osf_sessions(tmp_path, partition, 0)
    assert [path.stem for path in selected] == ["study01_p02_prep"]


def test_corrupted_source_container_fails_before_extraction(tmp_path):
    import json
    source = tmp_path/"inputs"
    source.mkdir()
    (source/"data.eogbundle").write_bytes(b"corrupted")
    (source/"source_manifest.json").write_text(json.dumps({"archive": {
        "name": "data.eogbundle", "bytes": 9, "sha256": "0"*64}}))
    with pytest.raises(ValueError, match="checksum"):
        prepare_sources(source, tmp_path/"extract", tmp_path/"output", "0"*64)
    assert not (tmp_path/"extract").exists()


def test_vmd_solver_cap_is_visible_instead_of_claimed_success():
    signal = np.random.default_rng(5).normal(size=1024)
    _, _, diagnostic = rolling_decompose(signal, 3, 2000, max_iterations=3)
    assert diagnostic["hit_iteration_limit"]


def test_new_window_regression_matches_per_window_reference_access():
    from eog_clean.experiment import _artifacts
    values = np.random.default_rng(2).normal(size=(3, 2049)).astype(np.float32)
    eyes = np.random.default_rng(3).normal(size=(2, 2049)).astype(np.float32)
    config = {"window": 1024, "hop": 512, "ridge": .001}
    empty_fits = {"direct_calibrated": None, "ica_component_regression": None, "mwf": None}
    result = _artifacts(values, eyes, empty_fits, config)["direct_window"]
    assert result.shape == values.shape and np.isfinite(result).all()
    assert not np.allclose(result, fit_ridge(values, eyes).artifact(values, eyes))
