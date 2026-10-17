"""Full-condition evaluation must match author-code semantics."""
import json
import numpy as np
import pandas as pd
import pytest
from vmd_eog.native_protocol import (
    ConditionMoments, multi_overlap_add, eight_second_bank,
    evaluation_references, chance_bootstrap,
)
from vmd_eog.paper_metrics import pearson


def test_condition_moments_match_concatenation_with_trial_offsets():
    generator = np.random.default_rng(42)
    first = [generator.normal(size=(3, 37)), generator.normal(size=(3, 91))+7]
    second = [first[0]+generator.normal(size=(3, 37)), -first[1]+13]
    summary = ConditionMoments(3)
    for x, y in zip(first, second):
        summary.update(x, y)
    x, y = np.concatenate(first, -1), np.concatenate(second, -1)
    np.testing.assert_allclose(summary.correlation(), pearson(x, y), atol=1e-12)
    np.testing.assert_allclose(summary.rmse(), np.sqrt(((x-y)**2).mean(-1)), atol=1e-12)
    assert summary.count == 128
    assert not np.allclose(summary.correlation(), np.mean([pearson(a, b) for a, b in zip(first, second)], axis=0))


def test_multi_overlap_add_preserves_reference_alignment_and_all_samples():
    generator = np.random.default_rng(3)
    references = generator.normal(size=(2, 2273))
    eeg = np.vstack([references[0]+4, references[1]-2])
    calls = []

    def estimator(window_eeg, window_refs):
        np.testing.assert_allclose(window_eeg-window_refs, np.broadcast_to([[4], [-2]], window_eeg.shape))
        calls.append(1)
        return {"identity": np.zeros_like(window_eeg), "remove": window_refs}

    result = multi_overlap_add(eeg, references, estimator, methods=("identity", "remove"))
    np.testing.assert_array_equal(result["identity"], eeg)
    np.testing.assert_allclose(result["remove"], np.broadcast_to([[4], [-2]], eeg.shape), atol=1e-12)
    assert len(calls) > 1
    with pytest.raises(ValueError, match="incomplete comparison family"):
        multi_overlap_add(eeg, references, lambda x, r: {"identity": x}, methods=("identity", "remove"))


def test_eight_second_banks_never_concatenate_short_trials():
    trials = [(3, np.zeros((2, 1500))), (4, np.ones((2, 1500))), (9, np.full((2, 3300), 9))]
    bank, identity = eight_second_bank(trials)
    assert len(bank) == 2
    assert identity == [{"trial_id": 9, "start": 0, "samples": 1600},
                        {"trial_id": 9, "start": 1600, "samples": 1600}]
    assert all(np.all(window == 9) for window in bank)


def test_publisher_lpf_channels_are_selected_without_recreating_filters():
    values = np.arange(4*2*1600, dtype=float).reshape(2, 4, 1600)
    item = {"names": ["HEOG", "VEOG", "HEOG_lpf", "VEOG_lpf"], "data": values, "fs": 200}
    refs, identity = evaluation_references(item, [0, 1])
    np.testing.assert_array_equal(refs, values[:, [2, 3]])
    assert identity["status"] == "publisher HEOG_lpf/VEOG_lpf"
    item["names"][3] = "UNKNOWN"
    refs, identity = evaluation_references(item, [0, 1])
    np.testing.assert_array_equal(refs, values[:, [0, 1]])
    assert "fallback" in identity["status"]


def test_chance_draws_group_sessions_by_verified_participant(tmp_path):
    generator = np.random.default_rng(9)
    records = []
    for participant in range(5):
        for session in range(2):
            records.append({"participant": "p"+str(participant), "record_id": str(participant)+"-"+str(session),
                "rest": [generator.normal(size=(3, 1600))],
                **{condition: [generator.normal(size=(2, 1600))] for condition in ("lateral", "vertical", "blink")}})
    result = chance_bootstrap(records, tmp_path, repetitions=8)
    draws = pd.read_csv(tmp_path/"chance_draws.csv.gz")
    assert (draws.groupby(["condition", "iteration"]).participant.nunique() == 5).all()
    assert all(row["status"] == "computed adapted null" for row in result)
    summary = json.loads((tmp_path/"native_chance_summary.json").read_text())
    assert summary["exact_paper_replication"] is False
    assert summary["equivalence_claim"] is False
