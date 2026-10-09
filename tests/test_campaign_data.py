"""Kaggle-run governance and metric fixtures for campaign-v2 data stages."""
import json
import numpy as np
import pytest

from eog_vmd_fcm_bgru.campaign_contracts import RunSpec, atomic_json, canonical_hash, read_json_verified
from eog_vmd_fcm_bgru.controlled_corpus import build_corpus, fit_corpus
from eog_vmd_fcm_bgru.metrics_v2 import cross_fitted_reference_r2, lagged_hv_correlation
from eog_vmd_fcm_bgru.splits_v2 import assert_oof_fit_ids, freeze_splits


def _record(record_id, participant=None, low=False, eog=True):
    return {"record_id": record_id, "dataset": "osf", "participant_id": participant,
            "participant_verification": "verified" if participant else "unverified",
            "clean_target_group": None, "eog": {"heog": "HEOG" if eog else None, "veog": "VEOG" if eog else None},
            "eligibility": {"low_ocular_candidate": low}, "channels": [], "fs_native": 200}


def test_run_spec_canonical_hash_and_atomic_json(tmp_path):
    spec = RunSpec.from_dict({"schema_version": 2, "campaign_id": "campaign-1", "run_id": "run-1",
                              "stage": "provenance", "git_sha": "a" * 40, "config": {"b": 1}, "inputs": {}})
    path = tmp_path / "run.json"
    spec.write(path)
    recovered = RunSpec.load(path)
    assert recovered.resolved_config_hash == canonical_hash({"b": 1})
    value = {"x": [1, 2]}
    digest = atomic_json(tmp_path / "artifact.json", value)
    assert read_json_verified(tmp_path / "artifact.json", digest) == value


def test_split_freeze_keeps_verified_participant_together_and_oof_rejects_overlap(tmp_path):
    records = [_record("a1", "p1"), _record("a2", "p1"), _record("b", "p2"), _record("c", "p3")]
    split = freeze_splits(records, {"seed": 7, "fractions": {"train": .5, "val": .25, "test": .25}}, tmp_path)
    assert split["assignment"]["a1"] == split["assignment"]["a2"]
    with pytest.raises(ValueError, match="OOF"):
        assert_oof_fit_ids(["a1"], ["a1"], source="teacher")


def test_corpus_never_crosses_partition_and_is_recipe_only(tmp_path):
    records = [_record("p1", "p1", low=True), _record("d1", "d1", eog=True),
               _record("p2", "p2", low=True), _record("d2", "d2", eog=True)]
    split = freeze_splits(records, {"seed": 2, "fractions": {"train": .5, "val": .25, "test": .25}}, tmp_path)
    # Make a deliberately tiny deterministic compatible example while retaining the split contract.
    same_partition = split["assignment"]["p1"]
    split["assignment"]["d1"] = same_partition
    split["split_hash"] = canonical_hash({key: value for key, value in split.items() if key != "split_hash"})
    fit = fit_corpus(records, split, tmp_path, {})
    summary = build_corpus(records, split, fit, tmp_path, {"recipes_per_recipient": 1})
    rows = [json.loads(line) for line in (tmp_path / "corpus_manifest.jsonl").read_text().splitlines()]
    assert summary["materialized_waveforms"] is False
    assert all(row["array_path"] is None for row in rows)
    assert all(row["split_role"] == split["assignment"][row["donor_id"]] for row in rows)


def test_lagged_metric_and_crossfit_are_finite_or_explicit_nan():
    rng = np.random.default_rng(4)
    heog, veog = rng.normal(size=(2, 160))
    eeg = np.vstack([np.roll(heog, 3), np.roll(veog, -2)]) + .01 * rng.normal(size=(2, 160))
    lagged = lagged_hv_correlation(eeg, np.vstack([heog, veog]), max_lag=5)
    assert lagged["heog"]["mean_abs_correlation"] > .9
    result = cross_fitted_reference_r2(eeg[:, :80], np.vstack([heog[:80], veog[:80]]),
                                       eeg[:, 80:], np.vstack([heog[80:], veog[80:]]))
    assert len(result["r2_by_channel"]) == 2
    assert np.isfinite(result["mean_r2"])
    degenerate = lagged_hv_correlation(np.zeros((1, 20)), np.zeros((2, 20)), max_lag=1)
    assert np.isnan(degenerate["heog"]["mean_abs_correlation"])
