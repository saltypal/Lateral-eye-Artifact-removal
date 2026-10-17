"""Controlled frontal gate reporting; this never authorizes dependent stages."""
import numpy as np
from .io import atomic_json


def strict_mean(values):
    """Propagate undefined observations instead of dropping them from averages."""
    return np.asarray(values, dtype=float).mean()


def controlled_gates(scores, config):
    """Report participant-balanced accuracy and worst-record clean preservation.

    Identity defines the required example/channel coverage. Perfect clean-input
    SNR is intentionally outside the accuracy gate; its denominator is zero.
    Native validation and independent qualification remain separate checkpoints.
    """
    keys = ["example_id", "channel"]
    identity = scores[scores.evaluated_family == "identity"]
    expected = set(map(tuple, identity[keys].to_numpy()))
    if not expected or identity.duplicated(keys).any():
        raise ValueError("A unique identity comparator is required for coverage")
    limits = config["preservation"]
    accuracy_fields = ["snr_energy_db", "mse", "rrmse_time", "pearson_cc", "cc_deterioration"]
    clean_fields = ["clean_change", "alpha_db", "beta_db", "covariance_error"]
    reports = []
    for arm, group in scores.groupby("evaluated_family"):
        dirty = group[group.condition != "clean"]
        clean = group[group.condition == "clean"]
        observed = set(map(tuple, group[keys].to_numpy()))
        row_coverage = observed == expected and not group.duplicated(keys).any()
        accuracy_coverage = bool(len(dirty) and np.isfinite(dirty[accuracy_fields].to_numpy(float)).all())
        clean_coverage = bool(len(clean) and np.isfinite(clean[clean_fields].to_numpy(float)).all())
        balanced = dirty.groupby(["recipient", "condition", "input_snr_db"])[accuracy_fields].agg(strict_mean)
        recipients = balanced.groupby("recipient").agg(strict_mean)
        means = recipients.agg(strict_mean)
        # A short bad record must not disappear inside a long good recording.
        record_means = clean.groupby(["recipient", "example_id"])[clean_fields].agg(strict_mean)
        participant_means = record_means.groupby("recipient").agg(strict_mean)
        worst = {field: float(max(record_means[field].max(), participant_means[field].max()))
                 if clean_coverage else None for field in clean_fields}
        mean_snr = float(means.snr_energy_db) if accuracy_coverage else None
        mean_cc_loss = float(means.cc_deterioration) if accuracy_coverage else None
        gates = {
            "row_coverage": bool(row_coverage),
            "accuracy_coverage": accuracy_coverage,
            "clean_coverage": clean_coverage,
            "snr_15_db": bool(accuracy_coverage and mean_snr >= 15.),
            "clean_modification": bool(clean_coverage and worst["clean_change"] <= limits["clean_change"]),
            "clean_alpha": bool(clean_coverage and worst["alpha_db"] <= limits["alpha_db"]),
            "clean_beta": bool(clean_coverage and worst["beta_db"] <= limits["beta_db"]),
            "clean_covariance": bool(clean_coverage and worst["covariance_error"] <= limits["covariance"]),
            "paired_correlation": bool(accuracy_coverage and mean_cc_loss <= limits["pearson_deterioration"]),
        }
        failure_count = int(group.failure.notna().sum()) if "failure" in group else 0
        reports.append({"arm": arm, "rows": len(group), "expected_rows": len(identity),
            "recipients": int(group.recipient.nunique()), "failure_rows_scored": failure_count,
            "mean_output_snr_db": mean_snr, "mean_cc_deterioration_vs_direct": mean_cc_loss,
            "worst_clean_record_or_participant": worst, "gates": gates,
            "controlled_development_gates_pass": all(gates.values()),
            "stretch_20_db": bool(accuracy_coverage and mean_snr >= 20.),
            "native_validation_required": True, "scientific_accuracy_qualified": False,
            "regional_integration_authorized": False, "model_training_authorized": False})
    return reports


def save_controlled_review(scores, output, config):
    reports = controlled_gates(scores, config)
    atomic_json(output / "frontal_controlled_gates.json", reports)
    lines = ["# Controlled frontal development review", "",
             "Paper metrics remain primary. These are separate project engineering gates.",
             "No row authorizes posterior integration or neural development; native validation is pending.", "",
             "| Fixed arm | Mean output SNR dB | CC deterioration | Failed gates |", "|---|---:|---:|---|"]
    for report in reports:
        failed = ", ".join(name for name, passed in report["gates"].items() if not passed) or "none"
        snr = report["mean_output_snr_db"]
        cc = report["mean_cc_deterioration_vs_direct"]
        lines.append(f"| {report['arm']} | {snr:.4f}" if snr is not None else f"| {report['arm']} | undefined")
        lines[-1] += f" | {cc:.6f} | {failed} |" if cc is not None else f" | undefined | {failed} |"
    (output / "frontal_controlled_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
