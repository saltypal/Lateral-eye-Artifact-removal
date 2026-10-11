"""Review immutable frozen-gain results without fitting or altering predictions."""
import json
from pathlib import Path

import pandas as pd

from .artifacts import verify_parent
from .io import atomic_json, sha256_file
from .frontal_replication import uncertainty_report, plot_results
from .frontal_review import save_controlled_review, strict_mean


def run_evidence(input_root, output, config):
    candidates = list(Path(input_root).rglob("frontal_transfer_summary.json"))
    if len(candidates) != 1:
        raise ValueError("Attach exactly one completed fixed-gain hybrid parent")
    parent = candidates[0].parent
    verify_parent(parent, ("frontal_transfer_summary.json", "frontal_transfer_scores.csv"))
    summary = json.loads(candidates[0].read_text())
    if (summary.get("scope") != "verified_frontal_only" or
            summary.get("hybrid_ablation") is not True or
            summary.get("weights_trained") is not False or
            summary.get("autovmd_executed") is not False or
            summary.get("reserved_confirmation_opened") is not False):
        raise ValueError("Hybrid parent scope or information-access contract differs")
    scores = pd.read_csv(parent / "frontal_transfer_scores.csv")
    expected = {"identity", "direct", *summary["predeclared_arms"]}
    if set(scores.evaluated_family) != expected:
        raise ValueError("Required matched arms are missing or additional arms appeared")
    save_controlled_review(scores, output, config)
    uncertainty_report(scores, output, config)
    intervals_path = output / "frontal_uncertainty.json"
    intervals = json.loads(intervals_path.read_text())
    for item in intervals:
        item["scope"] = "fixed-recipe frontal development evidence; no final winner qualification"
    atomic_json(intervals_path, intervals)
    if summary.get("calibration_hybrid"):
        matched_arms = [("calibration_raw_gain","K3_calibration_mode_gain","matched_calibration")]
        if summary.get("axis_hybrid"):
            matched_arms.append(("calibration_raw_axis_gain","K3_calibration_axis_gain","matched_axis_calibration"))
        if summary.get("event_hybrid"):
            matched_arms.append(("calibration_raw_event_gain","K3_calibration_event_gain","matched_event_calibration"))
        if summary.get("event_gain_scaled"):
            matched_arms.append(("calibration_raw_scaled_event_gain","K3_calibration_scaled_event_gain","matched_gain_adjusted_events"))
        for raw_arm,vmd_arm,directory in matched_arms:
            matched = scores[scores.evaluated_family.isin([raw_arm,vmd_arm])].copy()
            matched.loc[matched.evaluated_family == raw_arm,"evaluated_family"] = "direct"
            matched_output = output / directory
            matched_output.mkdir()
            uncertainty_report(matched,matched_output,config)
            reports = json.loads((matched_output / "frontal_uncertainty.json").read_text())
            for item in reports:
                item["comparator"] = "matched "+raw_arm+"; identical information and gates"
                item["scope"] = "VMD gain estimation versus raw gain estimation on development sources"
            atomic_json(matched_output / "frontal_uncertainty.json",reports)
    plot_results(scores, output)
    condition_means = scores[scores.condition != "clean"].groupby(
        ["evaluated_family", "recipient", "condition", "input_snr_db"])[
        ["snr_energy_db", "snr_improvement_db", "mse", "rrmse_time", "pearson_cc"]].agg(strict_mean)
    condition_means.groupby(["evaluated_family", "condition", "input_snr_db"]).agg(strict_mean).to_csv(
        output / "frontal_hybrid_condition_means.csv")
    atomic_json(output / "frontal_hybrid_evidence_summary.json", {
        "scope": "verified_frontal_only", "scores_sha256": sha256_file(parent / "frontal_transfer_scores.csv"),
        "parent_run": json.loads((parent / "run_spec.json").read_text())["run_id"],
        "prediction_changes": False, "parameter_selection": False, "weights_trained": False,
        "autovmd_executed": False, "reserved_confirmation_opened": False,
        "uncertainty_method": "recipient/donor Poisson multiplier approximation; not a paper-native test",
        "native_validation_required": True, "qualification": False})
