"""Metric availability must not automatically authorize neural development."""
import json
import pandas as pd
from vmd_eog.paper_qualification import qualify_paper_evidence
from vmd_eog.paper_evaluation import PROTOCOL_ID


def paired_evidence(tmp_path):
    regional, corpus = tmp_path/"regional", tmp_path/"corpus"
    regional.mkdir()
    corpus.mkdir()
    (regional/"paper_evaluation_protocol.json").write_text(json.dumps({
        "id": PROTOCOL_ID, "snr_aggregation": "per channel first; never pooled channel energy"}))
    (regional/"paper_metric_report_summary.json").write_text("{}")
    (regional/"regional_summary.json").write_text(json.dumps({"methods": ["identity", "direct"]}))
    rows = [{"dataset": "controlled_LEMON_OSF", "condition": "blink", "recipient": person,
             "method": method, "rrmse_time": .1, "mse": 1., "pearson_cc": .9,
             "snr_energy_db": snr} for person, snr in (("p1", 0.), ("p2", 20.))
            for method in ("identity", "direct")]
    pd.DataFrame(rows).to_csv(regional/"paper_paired_source_means.csv", index=False)
    examples = [{**row, "example_id": row["recipient"]} for row in rows]
    pd.DataFrame(examples).to_csv(regional/"paper_paired_example_means.csv", index=False)
    for name in ("paper_paired_permutation_tests.csv", "paper_native_permutation_tests.csv"):
        (regional/name).write_text("status\ncomputed\n")
    (corpus/"corpus_summary.json").write_text(json.dumps({"controlled_examples": 2, "recipient_sources": 2}))
    return regional, corpus


def test_finite_channel_first_scores_do_not_release_without_native_evidence(tmp_path):
    regional, corpus = paired_evidence(tmp_path)
    result = qualify_paper_evidence(regional, corpus)
    assert result["complete"] is False
    assert result["paired_synopsis"]["direct"]["snr_energy_db"] == 10.
    assert "Complete-trial native paper evaluation is not attached" in result["reasons"]


def test_missing_or_duplicate_example_units_are_rejected(tmp_path):
    regional, corpus = paired_evidence(tmp_path)
    path = regional/"paper_paired_example_means.csv"
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    result = qualify_paper_evidence(regional, corpus)
    assert any("Duplicate method/example" in reason for reason in result["reasons"])
    frame.iloc[1:].to_csv(path, index=False)
    result = qualify_paper_evidence(regional, corpus)
    assert any("every example" in reason for reason in result["reasons"])


def test_synthetic_scope_cannot_qualify_scientific_model_gate(tmp_path):
    regional, corpus = paired_evidence(tmp_path)
    (corpus/"corpus_summary.json").write_text(json.dumps({"synthetic_fixture": True}))
    result = qualify_paper_evidence(regional, corpus)
    assert result["complete"] is False
    assert any("Synthetic software fixtures" in reason for reason in result["reasons"])
