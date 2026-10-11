"""Evaluate immutable source-excluded frontal recipes on the full corpus.

Correction settings were chosen on other development source buckets. This stage
never selects K/alpha, retunes a detector or uses a held-out target in inference.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .contracts import canonical_hash
from .experiments import corpus_parent
from .frontal_methods import FixedRecipe, FrontalCorrector, frontal_indices
from .frontal_replication import ArchivedModes, finite_value, plot_results, uncertainty_report
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired
from .paper_evaluation import save_protocol
from .paper_metrics import paired_channels
from .posterior_context import select_examples


def selected_parent(input_root):
    paths = list(Path(input_root).rglob("frontal_summary.json"))
    if len(paths) != 1:
        raise ValueError("Attach exactly one correction-screen parent")
    parent = paths[0].parent
    verify_parent(parent, ("frontal_summary.json", "frontal_fold_selections.jsonl",
                           "frontal_detector_folds.jsonl", "frontal_candidates.json"))
    report = json.loads(paths[0].read_text())
    if (report.get("profile") != "full" or report.get("selection_subset") != "source-balanced pilot recipients" or
            report.get("evaluation_complete") is not False or report.get("reserved_confirmation_opened") is not False):
        raise ValueError("Parent is not the declared full-bank, source-balanced correction screen")
    return parent


def validate_evaluation_sources(row, evidence):
    if row["partition"]["role"] != "development" or row["partition"]["fold"] != evidence["fold"]:
        raise ValueError("Selected evaluation partition differs or is reserved")
    if (row["recipient"] in evidence["training_recipients"] or
            row["donor"] in evidence["training_donors"]):
        raise ValueError("Selected evaluation source was exposed during detector fitting")


def evaluate(input_root, output, config):
    selection_parent = selected_parent(input_root)
    corpus, manifest = corpus_parent(input_root)
    rows = select_examples(manifest, "full")
    selected = read_jsonl(selection_parent / "frontal_fold_selections.jsonl")
    by_fold, detectors = {}, {}
    evidence = {row["fold"]: row for row in read_jsonl(selection_parent / "frontal_detector_folds.jsonl")}
    for selection in selected:
        if canonical_hash(selection["selected"]) != selection["candidate_hash"]:
            raise ValueError("Saved recipe hash differs")
        by_fold.setdefault(selection["fold"], {})[selection["family"]] = selection
    for fold in by_fold:
        path = f"detectors/fold-{fold}.joblib"
        verify_parent(selection_parent, (path,))
        detectors[fold] = joblib.load(selection_parent / path)
    # Later native jobs receive these exact source-excluded models and recipes.
    detector_dir = output / "detectors"
    detector_dir.mkdir()
    for fold, detector in detectors.items():
        joblib.dump(detector, detector_dir / f"fold-{fold}.joblib")
    write_jsonl(output / "frontal_detector_folds.jsonl", list(evidence.values()))
    write_jsonl(output / "frontal_fold_selections.jsonl", selected)
    write_jsonl(output / "frontal_selection_manifest.jsonl", rows)
    atomic_json(output / "frontal_recipe_parent.json", {
        "artifact_manifest_sha256": sha256_file(selection_parent / "artifact_manifest.json"),
        "source_spec": json.loads((selection_parent / "run_spec.json").read_text()),
        "retuning": False})
    cache = ArchivedModes(input_root, corpus, config)
    scores, failures, decisions, mode_records = [], [], [], []
    shards = output / "frontal_shards"
    shards.mkdir()
    save_protocol(output)
    started = time.perf_counter()
    for number, row in enumerate(rows):
        fold = row["partition"]["fold"]
        validate_evaluation_sources(row, evidence[fold])
        data = dict(np.load(corpus / row["array_path"], allow_pickle=False))
        calibration = dict(np.load(corpus / row["calibration"], allow_pickle=False))
        ids = frontal_indices(data["eeg"], data["channel_names"], data["regions"], data["mask"])
        if not len(ids):
            raise ValueError("No eligible frontal channels")
        bundles = {K: cache.load(row, data, FixedRecipe(K)) for K in (3, 4)}
        original, target = data["eeg"][ids], data["paired_reference"][ids]
        predictions, example_scores = {}, []
        for family, selection in by_fold[fold].items():
            corrector = FrontalCorrector(selection["selected"], config, detector=detectors[fold])
            result, artifact, diagnostic = corrector.denoise(data["eeg"], data["references"],
                calibration["references"], data["channel_names"], data["regions"], data["mask"],
                mode_bundles=bundles.get(corrector.spec.get("K")), calibration_eeg=calibration["eeg"])
            outside = np.ones(len(data["eeg"]), dtype=bool)
            outside[ids] = False
            if not np.array_equal(result[outside], data["eeg"][outside]):
                raise RuntimeError("Frontal evaluation changed a nonfrontal row")
            paper = paired_channels(result[ids], target, original)
            preserved = paired(result[ids], target, config["fs"])
            detail = {entry["channel"]: entry for entry in diagnostic}
            for local, index in enumerate(ids):
                name = str(data["channel_names"][index])
                failure = detail.get(name, {}).get("failure")
                example_scores.append({"example_id": row["example_id"], "recipient": row["recipient"],
                    "donor": row["donor"], "fold": fold, "selection_fold": fold, "condition": row["condition"],
                    "input_snr_db": row["input_snr_db"], "channel": name,
                    "candidate_hash": selection["candidate_hash"], "family": corrector.spec["family"],
                    "evaluated_family": family, "failure": failure,
                    "clean_change": float(paper["rrmse_time"][local]), "alpha_db": preserved["alpha_db"],
                    "beta_db": preserved["beta_db"], "covariance_error": preserved["covariance"],
                    **{metric: finite_value(values[local]) for metric, values in paper.items()}})
                if failure:
                    failures.append({"example_id": row["example_id"], "channel": name, "family": family,
                                     "failure": failure, "status": "scored_passthrough"})
            decisions.append({"example_id": row["example_id"], "family": family, "decisions": diagnostic})
            predictions[selection["candidate_hash"]] = result[ids].astype(np.float32)
        frame = pd.DataFrame(example_scores)
        frame.to_csv(shards / (row["example_id"] + ".csv"), index=False)
        np.savez_compressed(shards / (row["example_id"] + ".npz"), original=original, target=target,
            references=data["references"], channel_names=data["channel_names"][ids], **predictions)
        scores.extend(example_scores)
        for K in (3, 4):
            _, item = cache.index[(row["example_id"], K)]
            mode_records.extend({"example_id": row["example_id"], "K": K, "alpha": 2000, **detail}
                                for detail in item["diagnostics"])
        write_jsonl(output / "frontal_failures.jsonl", failures)
        atomic_json(output / "frontal_progress.json", {"completed": number + 1, "total": len(rows),
            "last_example": row["example_id"], "runtime_s": time.perf_counter() - started})
        print("FRONTAL_EVALUATION_SAVED", number + 1, len(rows), flush=True)
    frame = pd.DataFrame(scores)
    direct = frame[frame.evaluated_family == "direct"][["example_id", "channel", "pearson_cc"]]
    direct = direct.rename(columns={"pearson_cc": "direct_cc"})
    frame = frame.merge(direct, on=["example_id", "channel"], validate="many_to_one")
    frame["cc_deterioration"] = frame.direct_cc - frame.pearson_cc
    frame.to_csv(output / "frontal_oof_scores.csv", index=False)
    dirty = frame[frame.condition != "clean"]
    fields = ["snr_energy_db", "rrmse_time", "pearson_cc", "mse", "snr_improvement_db", "cc_deterioration"]
    balanced = dirty.groupby(["evaluated_family", "recipient", "condition", "input_snr_db"])[fields].agg(
        lambda values: np.asarray(values, float).mean())
    recipients = balanced.groupby(["evaluated_family", "recipient"]).mean()
    recipients.reset_index().to_csv(output / "frontal_oof_recipients.csv", index=False)
    recipients.groupby("evaluated_family").mean().to_csv(output / "frontal_oof_means.csv")
    balanced.groupby(["evaluated_family", "condition", "input_snr_db"]).mean().to_csv(
        output / "frontal_oof_conditions.csv")
    clean = frame[frame.condition == "clean"]
    clean.groupby(["evaluated_family", "recipient"])[
        ["clean_change", "alpha_db", "beta_db", "covariance_error"]].agg(
        lambda values: np.asarray(values, float).mean()).to_csv(output / "frontal_clean_preservation.csv")
    uncertainty_report(frame, output, config)
    write_jsonl(output / "frontal_component_diagnostics.jsonl", decisions)
    write_jsonl(output / "frontal_mode_diagnostics.jsonl", mode_records)
    plot_results(frame, output)
    atomic_json(output / "frontal_summary.json", {
        "profile": "full", "evaluation_complete": True, "selection_subset": None,
        "selection_sources": "20-example full correction bank; each recipe excludes its recipient/donor bucket",
        "examples": len(rows), "recipients": int(frame.recipient.nunique()),
        "arms": sorted(frame.evaluated_family.unique()), "failure_rows": len(failures),
        "scope": "verified_frontal_only", "autovmd_executed": False,
        "runtime_s": time.perf_counter() - started, "native_validation_complete": False,
        "teacher_qualified": False, "regional_integration_authorized": False,
        "model_training_authorized": False, "reserved_confirmation_opened": False,
        "paper_replication": False, "status": "full development fixed-arm evaluation; qualification pending"})
