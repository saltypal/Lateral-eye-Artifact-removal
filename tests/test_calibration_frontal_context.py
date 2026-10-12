"""Kaggle-only contracts for bounded OOF frontal calibration context."""
import json

import numpy as np
import pytest

import vmd_eog.calibration_frontal_context as context_module
from vmd_eog.calibration_frontal_context import (
    CHUNK_SAMPLES,
    CONTEXT_KEYS,
    fit_context_calibration,
    score_context,
    select_calibration_chunks,
)


def calibration(types=(1, 2, 3, 4), chunks_per_trial=1):
    generator = np.random.default_rng(19)
    chunk_count = len(types) * chunks_per_trial
    sample_count = chunk_count * CHUNK_SAMPLES
    eeg = generator.normal(size=(5, sample_count))
    references = generator.normal(size=(2, sample_count))
    return {
        "eeg": eeg,
        "references": references,
        "trial_types": np.repeat(np.repeat(types, chunks_per_trial), CHUNK_SAMPLES).astype(np.int64),
        "boundaries": np.arange(1, chunk_count, dtype=np.int64) * CHUNK_SAMPLES,
        "channel_names": np.asarray(["F3", "F4", "Fz", "Cz", "Pz"]),
        "regions": np.asarray([0, 0, 0, 2, 1]),
        "hemispheres": np.asarray([0, 1, 2, 2, 2]),
        "mask": np.ones(5, dtype=bool),
    }


def config():
    return {
        "fs": 200,
        "window": CHUNK_SAMPLES,
        "classical_grid": {"tolerance": 1e-6, "max_iterations": 2000},
    }


def test_chunk_selection_prioritizes_known_types_and_stays_within_joins():
    labels = np.concatenate([
        np.full(2 * CHUNK_SAMPLES, 1),
        np.full(CHUNK_SAMPLES, 3),
        np.full(2 * CHUNK_SAMPLES, 4),
    ])
    joins = np.asarray([2 * CHUNK_SAMPLES, 3 * CHUNK_SAMPLES])

    selected = select_calibration_chunks(labels, joins)

    assert len(selected) == 4
    assert [item["trial_type"] for item in selected] == [1, 1, 3, 4]
    assert selected[0]["selection_reason"] == "first_chunk_for_trial_type"
    assert selected[1]["selection_reason"] == "chronological_fill"
    for item in selected:
        assert item["stop"] - item["start"] == CHUNK_SAMPLES
        assert np.all(labels[item["start"]:item["stop"]] == item["trial_type"])
        assert not any(item["start"] < join < item["stop"] for join in joins)


def test_calibration_context_is_oof_bounded_cached_and_has_fixed_metadata(tmp_path, monkeypatch):
    data = calibration()
    # A valid rest holdout may have a zero artifact estimate; retain it as an
    # identity-preserving calibration block while other conditions add context.
    data["references"][:, :CHUNK_SAMPLES] = 0.
    decomposition_calls = []
    fit_calls = []

    def decompose(eeg, names, regions, recipe, solver_config, mask):
        decomposition_calls.append((eeg.shape, recipe.K, recipe.alpha))
        frontal_ids = np.flatnonzero(np.asarray(regions) == 0)
        result = {}
        for index in frontal_ids:
            result[int(index)] = (
                np.zeros((3, CHUNK_SAMPLES), dtype=np.float32),
                eeg[index].astype(np.float32, copy=True),
                {"iterations": 3},
            )
        return result

    def transfer(eeg, references, fitted_calibration, names, regions, mask, **kwargs):
        heldout_index = next(
            index for index in range(4)
            if np.array_equal(
                eeg,
                data["eeg"][:, index * CHUNK_SAMPLES:(index + 1) * CHUNK_SAMPLES],
            )
        )
        expected_training_indices = [index for index in range(4) if index != heldout_index]
        expected_training_eeg = np.concatenate([
            data["eeg"][:, index * CHUNK_SAMPLES:(index + 1) * CHUNK_SAMPLES]
            for index in expected_training_indices
        ], axis=-1)
        expected_training_references = np.concatenate([
            data["references"][:, index * CHUNK_SAMPLES:(index + 1) * CHUNK_SAMPLES]
            for index in expected_training_indices
        ], axis=-1)
        np.testing.assert_array_equal(fitted_calibration["eeg"], expected_training_eeg)
        np.testing.assert_array_equal(fitted_calibration["references"], expected_training_references)
        fit_calls.append({
            "training_length": fitted_calibration["eeg"].shape[-1],
            "training_boundaries": fitted_calibration["boundaries"].copy(),
            "heldout_index": heldout_index,
            "parameters": kwargs,
        })
        artifact = np.zeros_like(eeg, dtype=np.float64)
        if heldout_index != 0:
            artifact[:3] = references[0][None, :]
        return eeg - artifact, artifact, [{"triggered": True}]

    monkeypatch.setattr(context_module, "decompose_frontal", decompose)
    monkeypatch.setattr(context_module, "calibration_transfer", transfer)
    digest = "a" * 64

    result = fit_context_calibration(data, config(), tmp_path / "frontal_context", digest)

    assert result["eeg"].shape == (5, 4 * CHUNK_SAMPLES)
    assert result["predictors"].shape == (5, 4 * CHUNK_SAMPLES)
    np.testing.assert_array_equal(result["boundaries"], [1024, 2048, 3072])
    assert result["metadata"]["available"] is True
    assert result["metadata"]["context_feature_order"] == list(CONTEXT_KEYS)
    assert len(decomposition_calls) == 4
    assert len(fit_calls) == 4
    np.testing.assert_array_equal(result["predictors"][:, :CHUNK_SAMPLES],
                                  np.zeros((len(CONTEXT_KEYS), CHUNK_SAMPLES)))
    assert result["metadata"]["available"] is True
    for call in fit_calls:
        assert call["training_length"] == 3 * CHUNK_SAMPLES
        np.testing.assert_array_equal(call["training_boundaries"], [1024, 2048])
        assert call["parameters"]["threshold"] == 0.4
        assert call["parameters"]["lags"] == (-10, 0, 10)
        assert call["parameters"]["penalty"] == 0.01
        assert call["parameters"]["gain_mode"] == "shared"

    cache_files = list((tmp_path / "frontal_context").glob("*.npz"))
    metadata_files = list((tmp_path / "frontal_context").glob("*.json"))
    assert len(cache_files) == len(metadata_files) == 1
    cache_metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert cache_metadata["calibration_sha256"] == digest
    assert len(cache_metadata["implementation_sha256"]) == 64
    assert cache_metadata["recipe"]["K"] == 3
    assert cache_metadata["recipe"]["alpha"] == 2000
    assert len(cache_metadata["decomposition_diagnostics"]) == 4

    def unexpected_decomposition(*args, **kwargs):
        raise AssertionError("a verified passive cache should avoid VMD recomputation")

    monkeypatch.setattr(context_module, "decompose_frontal", unexpected_decomposition)
    reloaded = fit_context_calibration(data, config(), tmp_path / "frontal_context", digest)
    np.testing.assert_allclose(reloaded["predictors"], result["predictors"])


def test_fewer_than_two_chunks_fails_closed_without_decomposition(tmp_path, monkeypatch):
    data = calibration(types=(1,), chunks_per_trial=1)

    def unexpected(*args, **kwargs):
        raise AssertionError("not enough chunks to fit OOF context")

    monkeypatch.setattr(context_module, "decompose_frontal", unexpected)
    result = fit_context_calibration(data, config(), tmp_path, "b" * 64)

    assert not result["metadata"]["available"]
    assert "fewer than two" in result["metadata"]["failure"]
    assert result["eeg"].shape == (5, 0)
    assert result["predictors"].shape == (5, 0)
    assert result["boundaries"].size == 0


def test_raw_context_is_oof_bounded_and_skips_vmd_and_cache(tmp_path, monkeypatch):
    data = calibration()
    data["references"][:, :CHUNK_SAMPLES] = 0.
    fit_calls = []

    def unexpected_vmd(*args, **kwargs):
        raise AssertionError("raw context must not decompose or load VMD modes")

    def transfer(eeg, references, fitted_calibration, names, regions, mask, **kwargs):
        heldout_index = next(
            index for index in range(4)
            if np.array_equal(
                eeg,
                data["eeg"][:, index * CHUNK_SAMPLES:(index + 1) * CHUNK_SAMPLES],
            )
        )
        training_indices = [index for index in range(4) if index != heldout_index]
        expected_eeg = np.concatenate([
            data["eeg"][:, index * CHUNK_SAMPLES:(index + 1) * CHUNK_SAMPLES]
            for index in training_indices
        ], axis=-1)
        expected_references = np.concatenate([
            data["references"][:, index * CHUNK_SAMPLES:(index + 1) * CHUNK_SAMPLES]
            for index in training_indices
        ], axis=-1)
        np.testing.assert_array_equal(fitted_calibration["eeg"], expected_eeg)
        np.testing.assert_array_equal(fitted_calibration["references"], expected_references)
        fit_calls.append({
            "training_eeg": fitted_calibration["eeg"].copy(),
            "training_references": fitted_calibration["references"].copy(),
            "training_boundaries": fitted_calibration["boundaries"].copy(),
            "heldout_index": heldout_index,
            "bundle": kwargs["bundles"],
            "parameters": kwargs,
        })
        artifact = np.zeros_like(eeg, dtype=np.float64)
        if heldout_index != 0:
            artifact[:3] = references[0][None, :]
        return eeg - artifact, artifact, [{"triggered": True}]

    monkeypatch.setattr(context_module, "decompose_frontal", unexpected_vmd)
    monkeypatch.setattr(context_module, "_load_or_build_modes", unexpected_vmd)
    monkeypatch.setattr(context_module, "calibration_transfer", transfer)
    cache_folder = tmp_path / "raw_context_must_not_be_created"

    result = fit_context_calibration(
        data, config(), cache_folder, "d" * 64, representation="raw",
    )

    assert result["eeg"].shape == (5, 4 * CHUNK_SAMPLES)
    assert result["predictors"].shape == (5, 4 * CHUNK_SAMPLES)
    np.testing.assert_array_equal(result["boundaries"], [1024, 2048, 3072])
    assert result["metadata"]["available"] is True
    assert result["metadata"]["representation"] == "raw"
    assert result["metadata"]["cache_metadata"] is None
    raw_identity_keys = (
        "cache_version", "representation", "calibration_sha256",
        "implementation_sha256", "transfer_parameters", "chunk_samples",
        "chunk_policy", "max_chunks",
    )
    raw_identity = {key: result["metadata"][key] for key in raw_identity_keys}
    assert context_module.canonical_hash(raw_identity) == result["metadata"]["cache_hash"]
    expected_chunks = select_calibration_chunks(
        data["trial_types"], data["boundaries"], length=data["eeg"].shape[-1],
    )
    assert result["metadata"]["selected_chunks"] == expected_chunks
    assert len(fit_calls) == 4
    np.testing.assert_array_equal(result["predictors"][:, :CHUNK_SAMPLES],
                                  np.zeros((len(CONTEXT_KEYS), CHUNK_SAMPLES)))
    assert not cache_folder.exists()
    for call in fit_calls:
        assert call["bundle"] is None
        assert call["training_eeg"].shape[-1] == 3 * CHUNK_SAMPLES
        np.testing.assert_array_equal(call["training_boundaries"], [1024, 2048])
        expected_training_indices = [index for index in range(4)
                                     if index != call["heldout_index"]]
        expected_training_eeg = np.concatenate([
            data["eeg"][:, index * CHUNK_SAMPLES:(index + 1) * CHUNK_SAMPLES]
            for index in expected_training_indices
        ], axis=-1)
        expected_training_refs = np.concatenate([
            data["references"][:, index * CHUNK_SAMPLES:(index + 1) * CHUNK_SAMPLES]
            for index in expected_training_indices
        ], axis=-1)
        # Exact complement equality verifies the held-out chunk is excluded
        # from both EEG and reference fit samples, beyond a length check.
        np.testing.assert_array_equal(call["training_eeg"], expected_training_eeg)
        np.testing.assert_array_equal(call["training_references"], expected_training_refs)
        assert call["parameters"]["threshold"] == 0.4
        assert call["parameters"]["lags"] == (-10, 0, 10)
        assert call["parameters"]["penalty"] == 0.01
        assert call["parameters"]["gain_mode"] == "shared"


def test_raw_context_unavailable_and_fit_failures_are_closed(tmp_path, monkeypatch):
    data = calibration()

    def unexpected_vmd(*args, **kwargs):
        raise AssertionError("raw context must not call a VMD path")

    monkeypatch.setattr(context_module, "decompose_frontal", unexpected_vmd)
    monkeypatch.setattr(context_module, "_load_or_build_modes", unexpected_vmd)

    unknown = fit_context_calibration(data, config(), tmp_path / "unknown", "e" * 64,
                                      representation="guessed")
    assert unknown["metadata"]["available"] is False
    assert "unknown" in unknown["metadata"]["failure"]
    assert not (tmp_path / "unknown").exists()

    invalid_config = {**config(), "fs": 250}
    invalid = fit_context_calibration(data, invalid_config, tmp_path / "invalid", "e" * 64,
                                      representation="raw")
    assert invalid["metadata"]["available"] is False
    assert "invalid calibration contract" in invalid["metadata"]["failure"]
    assert not (tmp_path / "invalid").exists()

    short = calibration(types=(1,), chunks_per_trial=1)
    fewer = fit_context_calibration(short, config(), tmp_path / "short", "e" * 64,
                                    representation="raw")
    assert fewer["metadata"]["available"] is False
    assert "fewer than two" in fewer["metadata"]["failure"]
    assert not (tmp_path / "short").exists()

    def invalid_transfer(*args, **kwargs):
        raise ValueError("invalid raw-reference fit")

    monkeypatch.setattr(context_module, "calibration_transfer", invalid_transfer)
    failed_fit = fit_context_calibration(data, config(), tmp_path / "failed", "e" * 64,
                                         representation="raw")
    assert failed_fit["metadata"]["available"] is False
    assert "cross-fit failed" in failed_fit["metadata"]["failure"]
    assert not (tmp_path / "failed").exists()

    monkeypatch.setattr(
        context_module, "calibration_transfer",
        lambda eeg, references, *args, **kwargs: (eeg.copy(), np.zeros_like(eeg), []),
    )
    zero = fit_context_calibration(data, config(), tmp_path / "zero", "e" * 64,
                                   representation="raw")
    assert zero["metadata"]["available"] is False
    assert "entirely zero" in zero["metadata"]["failure"]
    assert not (tmp_path / "zero").exists()


def test_invalid_crossfit_fails_closed(tmp_path, monkeypatch):
    data = calibration()

    def decompose(eeg, names, regions, recipe, solver_config, mask):
        result = {}
        for index in np.flatnonzero(np.asarray(regions) == 0):
            result[int(index)] = (
                np.zeros((3, CHUNK_SAMPLES), dtype=np.float32),
                eeg[index].astype(np.float32, copy=True),
                {},
            )
        return result

    def invalid_fit(*args, **kwargs):
        raise ValueError("insufficient calibration after boundary exclusions")

    monkeypatch.setattr(context_module, "decompose_frontal", decompose)
    monkeypatch.setattr(context_module, "calibration_transfer", invalid_fit)
    result = fit_context_calibration(data, config(), tmp_path, "c" * 64)

    assert result["metadata"]["available"] is False
    assert result["eeg"].shape == (5, 0)
    assert result["predictors"].shape == (5, 0)
    assert "cross-fit failed" in result["metadata"]["failure"]


def test_score_context_uses_verified_frontal_whitelist_and_signed_sides():
    names = np.asarray(["F3", "F4", "Fz", "Cz", "Pz", "X1"])
    regions = np.asarray([0, 0, 0, 2, 1, 3])
    hemispheres = np.asarray([0, 1, 2, 2, 2, 3])
    mask = np.ones(len(names), dtype=bool)
    artifact = np.zeros((len(names), 16))
    artifact[0] = -2
    artifact[1] = 3
    artifact[2] = 1

    predictors, flags = score_context(artifact, names, regions, mask, hemispheres)

    assert predictors.shape == (5, 16)
    np.testing.assert_array_equal(predictors[0], -2)
    np.testing.assert_array_equal(predictors[1], 3)
    np.testing.assert_array_equal(predictors[2], 1)
    np.testing.assert_allclose(predictors[3], 2 / 3)
    np.testing.assert_array_equal(predictors[4], 5)
    assert flags["available"] is True
    assert flags["lateral_available"] is True
    assert flags["nonfrontal_input_used"] is False

    artifact[3, 0] = 1
    with pytest.raises(ValueError, match="outside verified masked frontal"):
        score_context(artifact, names, regions, mask, hemispheres)


def test_score_context_marks_missing_side_and_zero_context_unavailable():
    names = np.asarray(["Fz", "Cz"])
    regions = np.asarray([0, 2])
    hemispheres = np.asarray([2, 2])
    mask = np.ones(2, dtype=bool)
    artifact = np.zeros((2, 8))
    artifact[0] = 1

    predictors, flags = score_context(artifact, names, regions, mask, hemispheres)
    assert flags["available"] is True
    assert flags["midline_available"] is True
    assert flags["left_available"] is False
    assert flags["right_available"] is False
    assert flags["lateral_available"] is False
    np.testing.assert_array_equal(predictors[4], 0)

    artifact[:] = 0
    _, zero_flags = score_context(artifact, names, regions, mask, hemispheres)
    assert zero_flags["available"] is False


def test_score_context_rejects_guessed_frontal_anatomy():
    artifact = np.ones((2, 8))
    with pytest.raises(ValueError, match="Frontal region labels require verified electrode names"):
        score_context(artifact, ["X1", "F3"], [0, 0], [True, True], [0, 1])
