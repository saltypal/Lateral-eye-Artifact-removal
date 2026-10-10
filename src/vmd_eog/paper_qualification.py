"""Qualify completed paper evidence; scientific outcomes stay separate."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import canonical_hash
from .paper_evaluation import PROTOCOL_ID
from .paper_report import paired_synopsis


def qualify_paper_evidence(regional, corpus, native=None):
    """Completion is checked from artifacts, never a hard-coded approval flag.

    This verifies protocol coverage and aggregation. It does not invent a
    published suppression threshold or claim native clean-reference SNR.
    Project SNR/preservation gates are evaluated separately in the review.
    All parent bytes must first be verified by the caller.
    """
    regional, corpus = Path(regional), Path(corpus)
    reasons = []
    required = ("paper_evaluation_protocol.json", "paper_metric_report_summary.json",
                "paper_paired_source_means.csv", "paper_paired_example_means.csv",
                "paper_paired_permutation_tests.csv", "paper_native_permutation_tests.csv")
    missing = [name for name in required if not (regional/name).is_file()]
    if missing:
        return {"complete": False, "reasons": ["Missing paper artifacts: "+", ".join(missing)], "paired_synopsis": {}}
    protocol = json.loads((regional/required[0]).read_text())
    if protocol.get("id") != PROTOCOL_ID or protocol.get("snr_aggregation") != "per channel first; never pooled channel energy":
        reasons.append("Paper protocol identity or channel-first aggregation differs")
    sources = pd.read_csv(regional/"paper_paired_source_means.csv")
    controlled = sources[(sources.dataset == "controlled_LEMON_OSF") & (sources.condition != "clean")]
    metrics = ["rrmse_time", "mse", "pearson_cc", "snr_energy_db"]
    if not len(controlled) or not np.isfinite(controlled[metrics].to_numpy()).all():
        reasons.append("Controlled paper metrics are empty or contain undefined/infinite units")
    synopsis = paired_synopsis(controlled) if len(controlled) else pd.DataFrame()
    synopsis_dict = {str(method): {metric: (float(row[metric]) if np.isfinite(row[metric]) else None) for metric in metrics}
                     for method, row in synopsis.iterrows()}
    summary = json.loads((corpus/"corpus_summary.json").read_text())
    if summary.get("synthetic_fixture"):
        return {"complete": False, "reasons": ["Synthetic software fixtures cannot qualify scientific evidence"],
                "paired_synopsis": synopsis_dict, "protocol_id": PROTOCOL_ID}
    expected_methods = set(json.loads((regional/"regional_summary.json").read_text())["methods"])
    examples = pd.read_csv(regional/"paper_paired_example_means.csv")
    examples = examples[examples.dataset == "controlled_LEMON_OSF"]
    counts = examples.groupby("method").example_id.nunique().to_dict()
    if set(counts) != expected_methods or any(count != summary["controlled_examples"] for count in counts.values()):
        reasons.append("Controlled paper tables do not cover every example and declared method")
    if examples.duplicated(["method", "example_id"]).any():
        reasons.append("Duplicate method/example units in paired paper evidence")
    for method in expected_methods:
        group = controlled[controlled.method == method]
        if group.recipient.nunique() != summary["recipient_sources"]:
            reasons.append("Missing controlled recipients for method "+method)
    if native is None:
        reasons.append("Complete-trial native paper evaluation is not attached")
    else:
        native = Path(native)
        native_required = ("native_protocol_summary.json", "native_condition_channels.csv",
                           "native_condition_participants.csv", "native_condition_permutations.csv",
                           "native_source_ledger.jsonl", "native_chance_summary.json",
                           "native_recipe_selections.json", "native_bank_inventory.json")
        native_missing = [name for name in native_required if not (native/name).is_file()]
        if native_missing:
            reasons.append("Missing native paper artifacts: "+", ".join(native_missing))
        else:
            native_summary = json.loads((native/"native_protocol_summary.json").read_text())
            if (native_summary.get("profile") != "full" or native_summary.get("reserved_confirmation_opened") is not False
                    or native_summary.get("condition_concatenation") is not True
                    or native_summary.get("native_reconstruction_snr_claim") is not False):
                reasons.append("Native evaluation is not a full development-only condition-level run")
            expected_records = {}
            for path in (corpus/"calibration").glob("*/record.json"):
                record = json.loads(path.read_text())
                if record["dataset"] == "osf":
                    expected_records[record["record_id"]] = record
            actual_records = [json.loads(line) for line in (native/"native_source_ledger.jsonl").read_text().splitlines() if line.strip()]
            if {row["record_id"] for row in actual_records} != set(expected_records) or len(actual_records) != len(expected_records):
                reasons.append("Native recording coverage differs from the frozen development ledger")
            for row in actual_records:
                if row["partition"]["role"] != "development" or row != expected_records.get(row["record_id"]):
                    reasons.append("Native source identities/calibration/scoring metadata differ from corpus")
                    break
            recipes = json.loads((native/"native_recipe_selections.json").read_text())
            regional_recipes = json.loads((regional/"grouped_recipe_selections.json").read_text())
            if canonical_hash(recipes) != canonical_hash({key: regional_recipes[key] for key in ("frontal", "posterior")}):
                reasons.append("Native and controlled evaluations use different held-out-fold recipes")
            native_metrics = pd.read_csv(native/"native_condition_channels.csv")
            for condition, group in native_metrics.groupby("condition"):
                field = "rest_rmse" if condition == "rest" else "eeg_eog_abs_r_after"
                if not np.isfinite(group[field].to_numpy()).all():
                    reasons.append("Undefined complete-condition native metrics: "+condition)
            if set(native_metrics.condition) != {"rest", "lateral", "vertical", "blink"}:
                reasons.append("Native evidence lacks one of the four prescribed conditions")
            if set(native_metrics.record_id) != set(expected_records):
                reasons.append("Missing complete-condition recording results")
            for record_id, group in native_metrics.groupby("record_id"):
                covered=set(zip(group.condition,group.method))
                expected={(condition,method) for condition in ("rest","lateral","vertical","blink")
                          for method in ("identity","direct","regional_mwf","regional_ica")}
                if covered!=expected:
                    reasons.append("Incomplete native condition/method coverage for "+record_id)
            if native_metrics.duplicated(["record_id","condition","method","channel"]).any():
                reasons.append("Duplicate channel units in complete-condition native evaluation")
            chance = json.loads((native/"native_chance_summary.json").read_text())
            if chance.get("window_seconds") != 8 or chance.get("participants_per_draw") != 5:
                reasons.append("Native chance analysis does not use declared eight-second/five-person units")
            if chance.get("exact_paper_replication") is not False or chance.get("equivalence_claim") is not False:
                reasons.append("Chance adaptation incorrectly claims exact replication or equivalence")
            if {row["condition"] for row in chance.get("conditions", [])} != {"lateral", "vertical", "blink"}:
                reasons.append("Chance eligibility/result missing for an ocular condition")
            for row in chance.get("conditions", []):
                if row.get("status") == "computed adapted null":
                    if row.get("repetitions") != 5000 or row.get("p95") is None or not 0 <= row["p95"] <= 1:
                        reasons.append("Chance threshold/repetition contract differs")
                elif not row.get("status", "").startswith("unavailable:"):
                    reasons.append("Unexplained chance-analysis status")
    return {"complete": not reasons, "reasons": reasons, "paired_synopsis": synopsis_dict,
            "protocol_id": PROTOCOL_ID, "scope": "source-mapped adapted evaluation; not exact paper replication",
            "native_clean_reference_snr": False}
