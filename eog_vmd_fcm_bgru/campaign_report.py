"""Summarize saved remote evidence without tuning from OSF or test results."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .provenance import save_json, sha256_file, environment
from .research_plots import save, neural_search_figures, BLUE, ORANGE, GREY


def unique_source(filename):
    found = list(Path("/kaggle/input").rglob(filename))
    if len(found) != 1:
        raise RuntimeError(f"Expected one completed source artifact {filename}, found {len(found)}")
    return found[0]


def bootstrap_mean(values, generator):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None, None
    indices = generator.integers(0, len(values), size=(2000, len(values)))
    averages = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(averages, 0.025)), float(np.quantile(averages, 0.975))


def build_report(output, repository):
    neural = unique_source("selected_neural.json").parent
    convergence = unique_source("convergence_check_summary.json").parent
    calibration = unique_source("calibration_check_summary.json").parent
    paths = [neural / name for name in ["selected_neural.json", "training_summary.json", "student_record_metrics.csv",
        "fresh_record_bootstrap.csv", "student_osf_coverage.json", "student_osf_proxies.csv", "student_test_predictions.npz",
        "fresh_load_verification.json", "gate_diagnostics.json", "neural_grid.csv"]]
    paths += [convergence / name for name in ["convergence_check_summary.json", "vmd_amplitude_scaling.csv", "convergence_grid_summary.csv"]]
    paths += [calibration / "calibration_region_proxies.csv", calibration / "calibration_diagnostics.json"]
    save_json(output / "report_source_ledger.json", {"inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in paths],
        "source_runs": {"neural": json.loads((neural / "run_config.json").read_text()),
                        "vmd_convergence": json.loads((convergence / "run_config.json").read_text()),
                        "calibration": json.loads((calibration / "run_config.json").read_text())},
        "selection_from_report": False})
    save_json(output / "environment.json", environment(repository))
    frame = pd.read_csv(neural / "student_osf_proxies.csv")
    coverage = json.loads((neural / "student_osf_coverage.json").read_text())
    audit = json.loads((neural / "audit_summary.json").read_text())
    if frame.session.nunique() != audit["osf_usable_sessions"] or len(coverage) != audit["osf_usable_sessions"]:
        raise RuntimeError("Frozen OSF output does not cover the audited sessions")
    metrics = ["heog_abs_corr_before", "veog_abs_corr_before", "heog_corr_reduction", "veog_corr_reduction",
               "relative_change", "rest_relative_change", "rest_raw_alpha_error_db", "rest_raw_beta_error_db"]
    generator = np.random.default_rng(42005)
    study_rows = []
    for key, group in frame.groupby(["study", "coverage_type", "variant", "region"]):
        per_person = group.groupby("participant")[metrics].mean()
        for metric in metrics:
            mean, lower, upper = bootstrap_mean(per_person[metric], generator)
            study_rows.append(dict(zip(["study", "condition", "variant", "region"], key)) |
                {"metric": metric, "mean": mean, "ci95_lower": lower, "ci95_upper": upper,
                 "participants": len(per_person), "sessions": group.session.nunique()})
    studies = pd.DataFrame(study_rows)
    studies.to_csv(output / "osf_study_region_summary.csv", index=False)
    global_rows = []
    for key, group in frame.groupby(["coverage_type", "variant", "region"]):
        per_person = group.groupby("participant")[metrics].mean()
        for metric in metrics:
            mean, lower, upper = bootstrap_mean(per_person[metric], generator)
            global_rows.append(dict(zip(["condition", "variant", "region"], key)) |
                {"metric": metric, "mean": mean, "ci95_lower": lower, "ci95_upper": upper, "participants": len(per_person)})
    global_frame = pd.DataFrame(global_rows)
    global_frame.to_csv(output / "osf_participant_region_summary.csv", index=False)
    paired = frame.pivot(index=["study", "participant", "session", "coverage_type", "region"], columns="variant", values=metrics)
    differences = []
    for metric in metrics:
        delta = paired[metric]["regional_prior_student"] - paired[metric]["shared_student"]
        delta = delta.rename("difference").reset_index()
        for key, group in delta.groupby(["coverage_type", "region"]):
            values = group.groupby("participant").difference.mean()
            mean, lower, upper = bootstrap_mean(values, generator)
            differences.append({"condition": key[0], "region": key[1], "metric": metric,
                "regional_minus_shared": mean, "ci95_lower": lower, "ci95_upper": upper, "participants": len(values)})
    pd.DataFrame(differences).to_csv(output / "regional_prior_paired_differences.csv", index=False)
    calibration_frame = pd.read_csv(calibration / "calibration_region_proxies.csv")
    calibration_frame.groupby(["study", "condition", "method", "region"], as_index=False)[
        [column for column in metrics if column in calibration_frame]].mean().to_csv(output / "classical_four_session_region_summary.csv", index=False)
    regional_figures(studies, output)
    fresh_waveforms(neural, output)
    numerical_figures(convergence, output)
    for filename in ["neural_grid.csv", "selected_neural.json", "student_record_metrics.csv", "search_protocol.json"]:
        shutil.copyfile(neural / filename, output / filename)
    neural_search_figures(output)
    training = json.loads((neural / "training_summary.json").read_text())
    bounds = pd.read_csv(neural / "fresh_record_bootstrap.csv").set_index("metric")
    summary = {"osf_sessions_evaluated": int(frame.session.nunique()), "osf_participants": int(frame.participant.nunique()),
        "osf_studies": sorted(frame.study.unique().tolist()), "conditions_present_every_session": all(len(item["conditions_scored"]) == 4 for item in coverage),
        "paired_test_fresh_records": int(bounds.records.iloc[0]), "test_rmse_reduction_mean": float(bounds.loc["rmse_reduction_fraction", "mean"]),
        "test_snr_gain_mean_db": float(bounds.loc["snr_gain_db", "mean"]),
        "test_clean_relative_change_mean": float(bounds.loc["clean_relative_change", "mean"]),
        "test_clean_relative_change_one_sided_95_upper": float(bounds.loc["clean_relative_change", "one_sided_95_upper"]),
        "development_guard_pass": training["development_guard_pass"], "development_removal_target_pass": training["development_removal_target_pass"],
        "matched_legacy_comparison": False, "five_fold_three_seed_validation": False, "participant_purged_adaptation": False,
        "streaming_certified": False, "complete_eog_removal": False, "victory": False,
        "scope": "saved feasibility/development evidence; OSF correlation reduction and rest stability are proxies"}
    save_json(output / "campaign_evidence_summary.json", summary)
    text = ["# Measured EOG campaign evidence", "", "The current models reduce some ocular contamination. Complete blink/lateral removal and final superiority have not been established.", "",
        f"The frozen student covered {summary['osf_sessions_evaluated']} original OSF sessions, {summary['osf_participants']} globally identified people, and all four studies. Each session supplied one genuinely annotated post-calibration interval for each of four conditions. This is balanced condition coverage, not every time sample.", "",
        f"Across six previously untouched held-out Klados records, mean record RMSE reduction versus raw was {100 * summary['test_rmse_reduction_mean']:.2f}% and mean SNR gain was {summary['test_snr_gain_mean_db']:.3f} dB. Mean clean modification was {100 * summary['test_clean_relative_change_mean']:.3f}%, with a one-sided 95% record-bootstrap upper bound of {100 * summary['test_clean_relative_change_one_sided_95_upper']:.3f}%. Participant identities are unavailable for Klados. Two earlier inspected smoke records are excluded from these bounds.", "",
        "The selected model passes its development preservation guard but misses the 10% development error-reduction target. The full five-fold/three-seed matched-legacy comparison, participant-purged adaptation and streaming evaluation remain incomplete. OSF correlation reduction and stability relative to raw do not measure paired neural recovery.", "",
        "## Frozen OSF ocular proxies", "", "| Condition | Variant | Region | Mean EOG correlation reduction | 95% participant interval |", "|---|---|---|---:|---:|"]
    for condition, metric in [("blink", "veog_corr_reduction"), ("horizontal", "heog_corr_reduction")]:
        selected_rows = global_frame[(global_frame.condition == condition) & (global_frame.metric == metric)]
        for _, row in selected_rows.iterrows():
            text.append(f"| {condition} | {row.variant} | {row.region} | {row['mean']:.5f} | [{row.ci95_lower:.5f}, {row.ci95_upper:.5f}] |")
    text += ["", "## Numerical VMD finding", "", "Absolute stopping thresholds depend on input amplitude. RMS-normalized stopping restored unit invariance in the measured training-channel diagnostic. The six stopping-rule candidates still failed the combined development preservation/convergence guard. The expanded normalized K/alpha search is a separate phase; it does not retroactively change this student's teacher.", "",
        "## User concerns and provenance", "", "The repository documents the genuine K=3..10/alpha search, why five modes were only a starting choice, adaptive mode vectors versus fixed Welch bands, mode generation/residual retention, computational cost and transformer necessity, frontal/posterior hypotheses and their limitations. See docs/VMD_Concerns_and_Explanation.md and docs/Recovered_Klados_Provenance.md. All four NPY exports match publisher start crops; recording 45 was omitted. Channel-row and participant identities remain unverified.", "",
        "All calculations and figures were generated on Kaggle. report_source_ledger.json records exact input hashes and source commits. Figures are accompanied by SVG and source manifests."]
    (output / "Measured_Campaign_Report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def regional_figures(studies, output):
    regions = ["frontal", "posterior", "central_temporal"]
    study_names = sorted(studies.study.unique())
    figure, axes = plt.subplots(2, len(study_names), figsize=(15, 7), squeeze=False)
    for row, (condition, metric, title) in enumerate([("blink", "veog_corr_reduction", "Blink / VEOG"),
                                                    ("horizontal", "heog_corr_reduction", "Lateral / HEOG")]):
        for column, study in enumerate(study_names):
            axis = axes[row, column]
            selected = studies[(studies.study == study) & (studies.condition == condition) & (studies.metric == metric)]
            for variant, offset, color, label in [("shared_student", -0.17, BLUE, "Shared"),
                                                  ("regional_prior_student", 0.17, ORANGE, "Regional prior")]:
                group = selected[selected.variant == variant].set_index("region").reindex(regions)
                mean = group["mean"].to_numpy()
                errors = np.stack([mean - group.ci95_lower.to_numpy(), group.ci95_upper.to_numpy() - mean])
                axis.bar(np.arange(3) + offset, mean, width=0.32, yerr=np.maximum(errors, 0), capsize=2, color=color, label=label)
            axis.axhline(0, color=GREY, linewidth=0.8)
            axis.set(title=f"{study} · {title}", xticks=range(3), xticklabels=["Front", "Back", "Central"])
            axis.grid(axis="y", alpha=0.15)
            if column == 0:
                axis.set_ylabel("Absolute EOG correlation reduction")
            if row == column == 0:
                axis.legend(fontsize=8)
    figure.suptitle("Frozen EEG-only transfer · region and artifact type across all four studies")
    figure.text(0.01, 0.01, "One annotated interval per condition/session. Error bars: 95% participant bootstrap within study. Correlation reduction is an ocular proxy; regional priors were not anatomically trained on Klados.", fontsize=8)
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(figure, output, "osf_region_artifact_comparison", ["osf_study_region_summary.csv", "report_source_ledger.json"])


def fresh_waveforms(source, output):
    arrays = np.load(source / "student_test_predictions.npz", allow_pickle=False)
    record_ids = [record for record in np.unique(arrays["record_ids"]) if record not in [5, 14]][:2]
    figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for axis, record in zip(axes, record_ids):
        index = np.flatnonzero(arrays["record_ids"] == record)[0]
        time = np.arange(arrays["raw"].shape[-1]) / 200
        for name, color, label in [("raw", GREY, "Contaminated"), ("target", BLUE, "Paired clean"), ("predictions", ORANGE, "Selected student")]:
            axis.plot(time, arrays[name][index, 0], color=color, linewidth=1, label=label)
        axis.set(title=f"Fresh held-out record {record} · anonymous EEG channel 0 · first scoring window", ylabel="Source amplitude units")
        axis.grid(alpha=0.15)
        axis.legend(fontsize=9, ncol=3)
    axes[-1].set_xlabel("Time within window (seconds)")
    figure.suptitle("Conservative model · residual ocular contamination remains")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save(figure, output, "fresh_heldout_waveforms", ["report_source_ledger.json"])


def numerical_figures(source, output):
    frame = pd.read_csv(source / "vmd_amplitude_scaling.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for config, group in frame.groupby("configuration"):
        normalized = "True" in config
        axes[0].plot(group.input_scale, group.restored_vector_relative_difference, marker="o", linestyle="-" if normalized else "--", label=config)
        axes[1].plot(group.input_scale, group.scaled_iterations, marker="o", linestyle="-" if normalized else "--")
    axes[0].set_yscale("symlog", linthresh=1e-7)
    axes[0].set(xscale="log", xlabel="Input amplitude multiplier", ylabel="Rescaled vector relative difference")
    axes[1].set(xscale="log", xlabel="Input amplitude multiplier", ylabel="Solver iterations")
    axes[0].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.15)
    figure.suptitle("VMD stopping criterion · numerical unit-scaling diagnostic")
    figure.text(0.01, 0.01, "One fixed training EEG channel, K=5/alpha=2000. Rescale outputs before comparison. RMS-normalized stopping is an explicit variant; it does not prove better cleaning.", fontsize=8)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    save(figure, output, "vmd_unit_scaling_diagnostic", ["report_source_ledger.json"])
