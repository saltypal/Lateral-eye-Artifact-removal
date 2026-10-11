"""Inspect saved frontal mechanisms; paired targets are diagnostics only."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .experiments import corpus_parent
from .frontal_methods import FixedRecipe, FrontalCorrector, frontal_indices, reference_intervals
from .frontal_replication import ArchivedModes, finite_value
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .paper_metrics import paired_channels
from .reference import project


def run_evidence(input_root, output, config):
    comparisons = list(Path(input_root).rglob("frontal_summary.json"))
    if len(comparisons) != 1:
        raise ValueError("Attach one completed frontal comparison")
    experiment = comparisons[0].parent
    verify_parent(experiment, ("frontal_summary.json", "frontal_selection_manifest.jsonl",
                              "frontal_fold_selections.jsonl", "frontal_detector_folds.jsonl"))
    summary = json.loads(comparisons[0].read_text())
    if summary["reserved_confirmation_opened"] or summary["scope"] != "verified_frontal_only":
        raise ValueError("Frontal evidence parent scope differs")
    evidence = {row["fold"]: row for row in read_jsonl(experiment / "frontal_detector_folds.jsonl")}
    if any("amplitude_scaling" not in row for row in evidence.values()):
        raise ValueError("Evidence interface requires the calibrated-detector follow-up")
    corpus, _ = corpus_parent(input_root)
    rows = read_jsonl(experiment / "frontal_selection_manifest.jsonl")
    # Predeclared first example per condition, independent of cleaning results.
    representative = {}
    for row in sorted(rows, key=lambda item: item["example_id"]):
        representative.setdefault(row["condition"], row)
    cache = ArchivedModes(input_root, corpus, config)
    by_fold = {}
    for row in read_jsonl(experiment / "frontal_fold_selections.jsonl"):
        by_fold.setdefault(row["fold"], {})[row["family"]] = row
    records, decisions = [], []
    vectors = output / "frontal_vectors"
    vectors.mkdir()
    parity_count = 0
    for row in representative.values():
        data = dict(np.load(corpus / row["array_path"], allow_pickle=False))
        calibration = dict(np.load(corpus / row["calibration"], allow_pickle=False))
        ids = frontal_indices(data["eeg"], data["channel_names"], data["regions"], data["mask"])
        mode_bundles = {K: cache.load(row, data, FixedRecipe(K)) for K in (3, 4)}
        fold = row["partition"]["fold"]
        path = f"detectors/fold-{fold}.joblib"
        verify_parent(experiment, (path,))
        detector = joblib.load(experiment / path)
        cleaned = {}
        for family, selection in by_fold[fold].items():
            corrector = FrontalCorrector(selection["selected"], config, detector)
            result, _, detail = corrector.denoise(data["eeg"], data["references"],
                calibration["references"], data["channel_names"], data["regions"], data["mask"],
                mode_bundles=mode_bundles.get(corrector.spec.get("K")), calibration_eeg=calibration["eeg"])
            cleaned[family] = result[ids]
            decisions.append({"example_id": row["example_id"], "family": family, "decisions": detail,
                              "inference_target_access": False})
            saved = experiment / "frontal_shards" / (row["example_id"] + ".npz")
            if saved.exists():
                verify_parent(experiment, (str(saved.relative_to(experiment)),))
                with np.load(saved, allow_pickle=False) as prediction:
                    np.testing.assert_allclose(result[ids], prediction[selection["candidate_hash"]],
                                               rtol=1e-5, atol=1e-11)
                parity_count += 1
        support = reference_intervals(data["references"], calibration["references"], config["fs"])
        original, target = data["eeg"][ids], data["paired_reference"][ids]
        artifact = original - target
        instantaneous = project(original, data["references"], [0], .01)[0]
        lagged = project(original, data["references"], [-10, 0, 10], .01)[0]
        # Evaluation-only matched lag diagnostic. Its design matches the donor
        # mixture construction; it is not evidence for native generalization.
        diagnostics = {"instantaneous": paired_channels(original-instantaneous, target, original),
                       "lagged": paired_channels(original-lagged, target, original)}
        for local, index in enumerate(ids):
            energy = float((artifact[local] ** 2).sum())
            retained = float((artifact[local] ** 2 * np.maximum(support[0], support[1])).sum())
            records.append({"example_id": row["example_id"], "condition": row["condition"],
                "channel": str(data["channel_names"][index]), "nominal_montage_input_db": row["input_snr_db"],
                "support_heog_fraction": float((support[0] > 0).mean()),
                "support_veog_fraction": float((support[1] > 0).mean()),
                "true_artifact_energy_supported": retained / energy if energy > 0 else None,
                "instantaneous_output_db": finite_value(diagnostics["instantaneous"]["snr_energy_db"][local]),
                "lagged_output_db": finite_value(diagnostics["lagged"]["snr_energy_db"][local]),
                "target_access": "diagnostic only; never used in estimate or selection"})
        np.savez_compressed(vectors / (row["example_id"] + ".npz"),
            original=original, target=target, references=data["references"], support=support,
            channel_names=data["channel_names"][ids],
            K3_modes=np.stack([mode_bundles[3][int(index)][0] for index in ids]),
            K3_residual=np.stack([mode_bundles[3][int(index)][1] for index in ids]),
            K4_modes=np.stack([mode_bundles[4][int(index)][0] for index in ids]),
            K4_residual=np.stack([mode_bundles[4][int(index)][1] for index in ids]), **cleaned)
        plot_example(row, data, calibration, ids, mode_bundles, cleaned, support, output, config)
    pd.DataFrame(records).to_csv(output / "frontal_mechanism_diagnostics.csv", index=False)
    write_jsonl(output / "frontal_inspected_decisions.jsonl", decisions)
    atomic_json(output / "frontal_evidence_summary.json", {
        "parent_manifest_sha256": sha256_file(experiment / "artifact_manifest.json"),
        "representative_policy": "first sorted example per condition; not selected for favorable output",
        "representative_examples": len(representative), "interface_prediction_parity_checks": parity_count,
        "source_profile": summary["profile"], "scope": "verified_frontal_only",
        "autovmd_executed": False, "new_decomposition_executed": False,
        "reserved_confirmation_opened": False, "scientific_accuracy_qualified": False,
        "teacher_qualified": False, "regional_integration_authorized": False,
        "diagnostic_limitation": "Lagged reference fields generated the corpus; no native generalization claim"})


def plot_example(row, data, calibration, ids, bundles, cleaned, support, output, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    local = next((position for position, index in enumerate(ids) if data["channel_names"][index] == "Fz"), 0)
    index = int(ids[local])
    times = np.arange(config["window"]) / config["fs"]
    figure, axes = plt.subplots(5, 1, figsize=(15, 17), sharex=True)
    scale = 1e6  # Controlled LEMON recipient is MNE volts; EOG is standardized separately.
    axes[0].plot(times, data["eeg"][index] * scale, label="contaminated frontal EEG", alpha=.7)
    axes[0].plot(times, data["paired_reference"][index] * scale, label="low-ocular recipient reference", color="black")
    axes[0].set_title(f"{row['condition']} / {row['example_id']} / {data['channel_names'][index]}")
    for axis, label in enumerate(("HEOG", "VEOG")):
        center, sd = calibration["references"][axis].mean(), calibration["references"][axis].std()
        standardized = ((data["references"][axis] - center) / sd if sd > 0 else np.zeros(config["window"]))
        axes[1].plot(times, standardized, label=label + " / unscored calibration SD")
        axes[1].fill_between(times, 0, support[axis], alpha=.12, label=label + " event support")
    axes[1].set_ylabel("Reference SD / support")
    for position, K in enumerate((3, 4), start=2):
        modes, residual, detail = bundles[K][index]
        centers = detail.get("centers_hz", [None] * K)
        for mode, center in zip(modes, centers):
            label = "mode center unavailable" if center is None else f"mode center {center:.2f} Hz"
            axes[position].plot(times, mode * scale, label=label, alpha=.7)
        axes[position].plot(times, residual * scale, label="retained residual", linestyle=":", color="black")
        axes[position].set_title(f"Fixed K={K}, alpha=2000; adaptive vectors, not fixed bands")
    axes[4].plot(times, data["paired_reference"][index] * scale, color="black", label="recipient reference")
    for family in ("direct", "vmd_projected_K3", "vmd_projected_K4"):
        if family in cleaned:
            axes[4].plot(times, cleaned[family][local] * scale, label=family, alpha=.7)
    for position, axis in enumerate(axes):
        axis.legend(loc="upper right", fontsize=8, ncol=2)
        if position != 1:
            axis.set_ylabel("EEG / microvolts")
    axes[-1].set_xlabel("Time within one original scoring window (seconds)")
    figure.tight_layout()
    figure.savefig(output / ("frontal_waveforms_" + row["condition"] + ".png"), dpi=130)
    plt.close(figure)
