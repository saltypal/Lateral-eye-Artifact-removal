"""Summarize saved paired-clean SNR without choosing a new model or subset."""
import json
import numpy as np
import pandas as pd
from .campaign_report import unique_source, bootstrap_mean
from .provenance import save_json, sha256_file, environment


def audit_snr(output, repository):
    neural_path = unique_source("student_record_metrics.csv")
    classical_path = unique_source("benchmark_records.csv")
    predictions = neural_path.parent / "student_test_predictions.npz"
    paths = [neural_path, classical_path, predictions]
    save_json(output / "snr_source_ledger.json", {
        "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in paths],
        "source_runs": {name: json.loads((path.parent / "run_config.json").read_text())
                        for name, path in [("neural", neural_path), ("classical", classical_path)]},
        "selection_from_audit": False})
    save_json(output / "environment.json", environment(repository))
    neural = pd.read_csv(neural_path)
    classical = pd.read_csv(classical_path)
    classical = classical[classical.dataset == "klados"]
    generator = np.random.default_rng(42006)
    rows = []
    for scope, frame in [("classical_two_record_one_window", classical),
                         ("neural_all_eight_records_three_windows", neural),
                         ("neural_six_previously_fresh_records_three_windows", neural[~neural.earlier_smoke_record])]:
        for method, group in frame.groupby("method"):
            means = group.groupby("record").snr_db.mean()
            mean, lower, upper = bootstrap_mean(means, generator)
            row = {"scope": scope, "method": method, "records": len(means),
                   "mean_record_snr_db": mean, "ci95_record_bootstrap_lower": lower,
                   "ci95_record_bootstrap_upper": upper, "macro_snr_exceeds_10_db": mean > 10,
                   "participant_independence_verified": False}
            if "clean_relative_change" in group and group.clean_relative_change.notna().any():
                row["mean_clean_relative_change"] = float(group.clean_relative_change.mean())
            rows.append(row)
    pd.DataFrame(rows).to_csv(output / "existing_campaign_snr.csv", index=False)
    data = np.load(predictions, allow_pickle=False)
    pooled = []
    for scope, keep in [("all_eight_records", np.ones(len(data["record_ids"]), dtype=bool)),
                        ("six_previously_fresh_records", ~np.isin(data["record_ids"], [5, 14]))]:
        target = data["target"][keep].astype(np.float64)
        for method, key in [("raw", "raw"), ("eeg_only_student", "predictions")]:
            error = data[key][keep].astype(np.float64) - target
            pooled.append({"scope": scope, "method": method,
                "pooled_sample_snr_db": float(10 * np.log10(np.sum(target ** 2) / np.sum(error ** 2))),
                "records": len(np.unique(data["record_ids"][keep])), "windows": int(keep.sum())})
    save_json(output / "pooled_neural_snr.json", pooled)
    summary = {"record_macro_results": rows, "pooled_results": pooled,
        "snr_definition": "10*log10(sum(paired_clean^2)/sum((estimate-paired_clean)^2)) per window",
        "macro_aggregation": "mean windows within recording, then mean recording SNR in dB",
        "pooled_aggregation": "ratio of all selected sample energies before logarithm; a separate estimand",
        "comparison_to_colab_is_matched": False, "new_model_selected": False,
        "complete_eog_removal": False, "full_validation": False, "victory": False}
    save_json(output / "snr_audit_summary.json", summary)
    print(pd.DataFrame(rows)[["scope", "method", "records", "mean_record_snr_db"]].to_string(index=False), flush=True)
