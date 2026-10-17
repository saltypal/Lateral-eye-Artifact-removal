"""Measured qualification summaries, separate from paper accuracy metrics."""
from pathlib import Path
import json
import pandas as pd


def write_review(output):
    output = Path(output)
    paired = pd.read_csv(output/"klados_record_macro.csv").set_index("method")
    clean = pd.read_csv(output/"klados_clean_preservation.csv")
    clean_record = clean.groupby(["method", "record"]).mean(numeric_only=True)
    clean_worst = clean_record.groupby("method").max()
    rows = []
    comparator = "direct_calibrated"
    for method in paired.index:
        values = clean_worst.loc[method]
        difference = paired.loc[method, "pearson_cc"]-paired.loc[comparator, "pearson_cc"]
        passes = {"clean_modification_pass": values.relative_change <= .01,
            "alpha_pass": values.alpha_error_db <= .5, "beta_pass": values.beta_error_db <= .5,
            "covariance_pass": values.covariance_relative_error <= .02,
            "paired_cc_pass": difference >= -.005}
        rows.append({"method": method, "worst_record_mean_clean_change": values.relative_change,
            "worst_record_alpha_error_db": values.alpha_error_db,
            "worst_record_beta_error_db": values.beta_error_db,
            "worst_record_covariance_error": values.covariance_relative_error,
            "paired_cc_difference": difference, "comparator": comparator,
            **passes, "all_preservation_gates_pass": all(passes.values()),
            "mean_output_snr_db": paired.loc[method, "pooled_window_snr_db"],
            "target_15db_met": paired.loc[method, "pooled_window_snr_db"] >= 15})
    qualification = pd.DataFrame(rows)
    qualification.to_csv(output/"klados_engineering_gates.csv", index=False)
    rest = pd.read_csv(output/"osf_rest_preservation.csv")
    worst = rest.groupby(["method", "record", "region"]).mean(numeric_only=True).groupby("method").max()
    worst["rest_modification_pass"] = worst.relative_change <= .01
    worst["alpha_pass"] = worst.alpha_error_db <= .5
    worst["beta_pass"] = worst.beta_error_db <= .5
    worst["covariance_pass"] = worst.covariance_relative_error <= .02
    worst["all_native_preservation_proxies_pass"] = worst[["rest_modification_pass", "alpha_pass", "beta_pass", "covariance_pass"]].all(axis=1)
    worst.to_csv(output/"osf_engineering_preservation_proxies.csv")
    historical = json.loads((output/"historical_replay.json").read_text())
    mode_evidence = pd.read_csv(output/"osf_mode_reference_evidence.csv")
    decompositions = mode_evidence.drop_duplicates(["record", "trial", "channel", "window_start"])
    text = ["# Measured replacement-campaign review", "",
        "This report is generated from executed outputs. It is not a production qualification.", "",
        "## Historical checkpoint", "",
        f"Frozen archived-cache record mean: **{historical['measured_record_macro_snr_db']:.6f} dB**. "
        "Eight reused Klados records, all 19 unnamed rows. EOG and VMD are required at inference. "
        "This is not fresh frontal SNR.", "", "## New paired Klados comparison", "",
        "Each method fits without paired clean targets. Electrode/participant mapping is unverified. "
        "Paired scores are nonregional and development-exposed. Calibration/scoring filtering is separated. "
        "The dataset's linear ocular contamination favors EOG regression.", "",
        qualification.to_markdown(index=False, floatfmt=".5f"), "", "## Native OSF", "",
        "Named-channel regional evaluation has no paired clean EEG target. Correlation and resting "
        "modification are proxies; no native reconstruction SNR is reported. The frozen learned model "
        "applied to verified frontal channels is an out-of-dataset adaptation, with changed metadata "
        "features and independently measured preservation. It is distinct from archived replay.", "",
        f"Frontal decompositions: {len(decompositions)}; invalid/capped: {int((~decompositions.valid).sum())}. "
        "Every mode's signed HEOG/VEOG correlations, selection and solver evidence are saved.", "",
        worst.reset_index().to_markdown(index=False, floatfmt=".5f"), "", "## Interpretation", "",
        "Do not select a winner from ocular correlation alone. Inspect paired and clean-control evidence, "
        "native resting preservation and all failures. Statistical generalization is limited by the "
        "declared record/participant scope. The historical result, new Klados results and native OSF "
        "proxies cannot be merged into one SNR or added across regions.", "",
        "## Remaining work", "",
        "Verified Klados channel order and participant mapping remain unavailable. "
        "Frontal-to-posterior learned transfer, EEGOAR-Net/ASR adapters, fresh independent paired "
        "regional validation, model export and deployment scaling are not established by this run.", ""]
    (output/"RESULTS.md").write_text("\n".join(text), encoding="utf-8")
    return qualification, worst
