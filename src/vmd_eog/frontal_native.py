"""Whole-trial frontal validation using unmodified native EOG references.

No posterior fitting or VMD is performed. Native recordings have no clean target;
resting EEG is a preservation reference, never an artifact-free reconstruction
target. Controlled source-excluded recipes are reused without native retuning.
"""
import json
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .contracts import canonical_hash
from .experiments import corpus_parent
from .frontal_methods import FixedRecipe, FrontalCorrector, decompose_frontal, frontal_indices
from .io import atomic_json, read_jsonl, sha256_file, unpack_source, write_jsonl
from .metrics import paired
from .native_protocol import (CONDITIONS, ConditionMoments, evaluation_references,
                              multi_overlap_add, write_native_reports)
from .osf_reader import read_osf
from .paper_evaluation import save_protocol
from .paper_metrics import resting_spectrum
from .readiness import verify_source_ledger
from .signal import preprocess


def controlled_parent(input_root):
    paths = list(Path(input_root).rglob("frontal_summary.json"))
    if len(paths) != 1:
        raise ValueError("Attach exactly one completed full frontal comparison")
    directory = paths[0].parent
    verify_parent(directory, ("frontal_summary.json", "frontal_fold_selections.jsonl",
                              "frontal_detector_folds.jsonl"))
    summary = json.loads(paths[0].read_text())
    if summary.get("profile") != "full" or summary.get("reserved_confirmation_opened") is not False:
        raise ValueError("Native validation requires full development comparison with confirmation closed")
    return directory


def validate_record_selection(record, evidence):
    """Reject held-out native participant exposure in detector fitting."""
    if record["partition"]["role"] != "development":
        raise ValueError("Reserved native source encountered")
    if record["partition"]["fold"] != evidence["fold"]:
        raise ValueError("Native recipe/detector source bucket differs")
    if record["participant"] in evidence["training_donors"]:
        raise ValueError("Native participant was exposed during detector fitting")
    if evidence.get("amplitude_scaling") != "unscored same-channel calibration SD; no scoring target scaling":
        raise ValueError("Native detector has no verified unit-invariant calibration normalization")


def run_native(input_root, output, config, profile):
    parent, _ = corpus_parent(input_root)
    experiment = controlled_parent(input_root)
    selections = read_jsonl(experiment / "frontal_fold_selections.jsonl")
    by_fold = {}
    for selection in selections:
        by_fold.setdefault(selection["fold"], {})[selection["family"]] = selection
    detector_evidence = {item["fold"]: item for item in read_jsonl(experiment / "frontal_detector_folds.jsonl")}
    source = unpack_source(input_root, tempfile.mkdtemp(prefix="frontal-native-"), output,
                           config["required_supplement_sha256"])
    records = [json.loads(path.read_text()) for path in (parent / "calibration").glob("*/record.json")]
    records = sorted((record for record in records if record["dataset"] == "osf"), key=lambda row: row["record_id"])
    if profile == "pilot":
        records = records[:2]
    if not records:
        raise ValueError("No frozen development OSF records")
    verify_source_ledger(source, records, output)
    write_jsonl(output / "native_frontal_sources.jsonl", records)
    atomic_json(output / "native_frontal_recipe_identity.json", {
        "parent_manifest_sha256": sha256_file(experiment / "artifact_manifest.json"),
        "policy": "source-excluded family-wise recipes reused unchanged; no native selection",
        "families": sorted({item["family"] for item in selections})})
    save_protocol(output)
    predictions = output / "native_frontal_predictions"
    predictions.mkdir()
    metrics, spectra, preservation, failures = [], [], [], []
    trial_count, started = 0, time.perf_counter()
    for record in records:
        fold = record["partition"]["fold"]
        validate_record_selection(record, detector_evidence[fold])
        item = read_osf(source / record["path"])
        from .data import eog_axes
        raw_refs = eog_axes(item)
        evaluation, ref_identity = evaluation_references(item, raw_refs, config["fs"])
        calibration_dir = parent / "calibration" / record["record_id"]
        verify_parent(parent, (str((calibration_dir / "calibration.npz").relative_to(parent)),))
        calibration = dict(np.load(calibration_dir / "calibration.npz", allow_pickle=False))
        names, regions = calibration["channel_names"], calibration["regions"]
        actual_names = np.asarray([item["names"][index] for index in item["eeg_indices"]])
        if not np.array_equal(names, actual_names):
            raise ValueError("Native montage differs from frozen calibration")
        mask = calibration.get("mask", np.ones(len(names), dtype=bool))
        ids = frontal_indices(calibration["eeg"], names, regions, mask)
        if not len(ids):
            raise ValueError("Native record has no verified frontal channels")
        detector_path = f"detectors/fold-{fold}.joblib"
        verify_parent(experiment, (detector_path,))
        detector = joblib.load(experiment / detector_path)
        estimators = {family: FrontalCorrector(selection["selected"], config, detector=detector)
                      for family, selection in by_fold[fold].items()}
        methods = tuple(sorted(estimators))
        moments = {}
        for trial_id in record["scoring_trial_ids"]:
            condition = CONDITIONS.get(item["trial_labels"][trial_id])
            if condition is None:
                failures.append({"record_id": record["record_id"], "trial": trial_id,
                                 "status": "ineligible", "reason": "unknown condition"})
                continue
            values = preprocess(item["data"][trial_id, item["eeg_indices"] + raw_refs], item["fs"])
            eeg, refs = values[:-2], values[-2:]
            eval_refs = evaluation[trial_id]
            if eval_refs.shape[-1] != eeg.shape[-1]:
                raise ValueError("Native reference alignment differs")
            window_count = 0

            def estimate(window_eeg, window_refs):
                nonlocal window_count
                window_count += 1
                # Each approved decomposition is computed once per frontal
                # window, regardless of how many correction families use it.
                required = {corrector.spec["K"] for corrector in estimators.values()
                            if corrector.spec["family"] not in ("identity", "direct")}
                bundles = {K: decompose_frontal(window_eeg, names, regions, FixedRecipe(K), config, mask)
                           for K in required}
                artifacts = {}
                for family, corrector in estimators.items():
                    _, artifact, diagnostic = corrector.denoise(window_eeg, window_refs,
                        calibration["references"], names, regions, mask,
                        mode_bundles=bundles.get(corrector.spec.get("K")), calibration_eeg=calibration["eeg"])
                    artifacts[family] = artifact
                    for detail in diagnostic:
                        if detail.get("failure"):
                            failures.append({"record_id": record["record_id"], "trial": trial_id,
                                             "window": window_count, "method": family, **detail})
                return artifacts

            cleaned = multi_overlap_add(eeg, refs, estimate, methods=methods,
                                        window=config["window"], hop=config["hop"])
            outside = np.ones(len(names), dtype=bool)
            outside[ids] = False
            for family, result in cleaned.items():
                if not np.array_equal(result[outside], eeg[outside]):
                    raise RuntimeError("Native frontal method altered a nonfrontal channel")
                summary = moments.setdefault((condition, family), ConditionMoments(len(ids)))
                reference = (eeg[ids] if condition == "rest" else
                             np.broadcast_to(eval_refs[0 if condition == "lateral" else 1], eeg[ids].shape))
                summary.update(result[ids], reference)
                if condition == "rest":
                    spectrum = resting_spectrum(result[ids], eeg[ids], config["fs"], normalization_band=(1., 40.))
                    spectra.append({"record_id": record["record_id"], "method": family,
                        "segments": 1 + (eeg.shape[-1] - 2 * config["fs"]) // config["fs"], "spectrum": spectrum})
                    engineering = paired(result[ids], eeg[ids], config["fs"])
                    preservation.append({"source": record["participant"], "record_id": record["record_id"],
                        "trial": trial_id, "method": family, "relative_error": engineering["relative_error"],
                        "alpha_db": engineering["alpha_db"], "beta_db": engineering["beta_db"],
                        "covariance": engineering["covariance"]})
            np.savez_compressed(predictions / f"{record['record_id']}-trial-{trial_id:04d}.npz",
                original=eeg, evaluation_references=eval_refs, inference_references=refs,
                channel_names=names, regions=regions, **cleaned)
            trial_count += 1
            print("FRONTAL_NATIVE_TRIAL", record["record_id"], trial_id, condition, window_count, flush=True)
        for (condition, family), summary in moments.items():
            before = moments[(condition, "identity")]
            for channel, index in enumerate(ids):
                metrics.append({"source": record["participant"], "record_id": record["record_id"], "dataset": "osf",
                    "condition": condition, "method": family, "channel": str(names[index]), "samples": summary.count,
                    "rest_rmse": float(summary.rmse()[channel]) if condition == "rest" else np.nan,
                    "eeg_eog_signed_r_after": float(summary.correlation()[channel]) if condition != "rest" else np.nan,
                    "eeg_eog_abs_r_after": float(abs(summary.correlation()[channel])) if condition != "rest" else np.nan,
                    "eeg_eog_abs_r_before": float(abs(before.correlation()[channel])) if condition != "rest" else np.nan,
                    "reference_status": ref_identity["status"], "scoring_unit": "complete condition concatenation",
                    "units": "original source array units; microvolt scale unverified"})
        pd.DataFrame(metrics).to_csv(output / "native_condition_channels.csv", index=False)
        pd.DataFrame(preservation).to_csv(output / "native_frontal_preservation_trials.csv", index=False)
        write_jsonl(output / "native_frontal_failures.jsonl", failures)
        atomic_json(output / "record_progress" / record["record_id"] / "completed_record.json", {
            "record_id": record["record_id"], "status": "complete", "source_sha256": record["source_sha256"],
            "configuration_hash": canonical_hash(config), "recipe_parent_sha256": sha256_file(experiment / "artifact_manifest.json"),
            "note": "Recovery requires matching source, code, configuration and recipe identities."})
        print("FRONTAL_NATIVE_RECORD_SAVED", record["record_id"], flush=True)
    write_native_reports(pd.DataFrame(metrics), spectra, output)
    fields = ["relative_error", "alpha_db", "beta_db", "covariance"]
    preserved = pd.DataFrame(preservation)
    if len(preserved):
        strict_mean = lambda values: np.asarray(values, float).mean()
        record_means = preserved.groupby(["source", "record_id", "method"])[fields].agg(strict_mean).reset_index()
        record_means.to_csv(output / "native_frontal_preservation_records.csv", index=False)
        record_means.groupby(["source", "method"])[fields].agg(strict_mean).reset_index().to_csv(
            output / "native_frontal_preservation_participants.csv", index=False)
    atomic_json(output / "native_frontal_summary.json", {
        "profile": profile, "recordings": len(records), "scoring_trials": trial_count,
        "failure_count": len(failures), "runtime_s": time.perf_counter() - started,
        "scope": "verified_frontal_only", "nonfrontal_changed": False,
        "reserved_confirmation_opened": False, "native_reconstruction_snr_claim": False,
        "trial_join_filtering": False, "native_parameter_retuning": False,
        "scientific_accuracy_qualified": False, "teacher_qualified": False,
        "paper_four_region_replication": False, "regional_integration_authorized": False})
