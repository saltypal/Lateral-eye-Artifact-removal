"""Frozen frontal diagnostic on verified original native trial caches.

The cache replaces source loading only. Historical cleaned outputs are never
algorithm inputs. Labels enter reporting after all trial predictions exist.
"""
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .broader_cached_review import raw_failure_fallback
from .broader_coupled_corpus import declared_parent
from .contracts import canonical_hash
from .coupled_frontal_review import validate_mode_reconstruction
from .frontal_calibration_transfer import calibration_transfer
from .frontal_methods import FixedRecipe, decompose_frontal, frontal_indices
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired
from .native_protocol import CONDITIONS, ConditionMoments, multi_overlap_add, write_native_reports
from .paper_metrics import resting_spectrum


METHODS = ("identity", "raw_shared", "vmd_shared", "vmd_anchored", "vmd_anchored_raw_fallback")


def validate_record(record, archived):
    """Require identical source annotations, not just the same filename."""
    if canonical_hash(record) != canonical_hash(archived):
        raise ValueError("Native cache and original calibration ledgers differ")
    scoring, calibration = record["scoring_trial_ids"], record["calibration_trial_ids"]
    if (record["dataset"] != "osf" or record["partition"]["role"] != "development"
            or not scoring or len(set(scoring)) != len(scoring)
            or set(scoring) & set(calibration)):
        raise ValueError("Native development/calibration/scoring identities are ineligible")
    if {CONDITIONS.get(record["trial_labels"][trial]) for trial in scoring} != set(CONDITIONS.values()):
        raise ValueError("Native pilot requires all four annotated conditions")


def validate_trial(stored, calibration, record):
    """Do not re-filter caches, normalize units or read old cleaned estimates."""
    eeg = np.asarray(stored["original"], dtype=np.float64)
    refs = np.asarray(stored["inference_references"], dtype=np.float64)
    evaluation = np.asarray(stored["evaluation_references"], dtype=np.float64)
    names, regions = stored["channel_names"], stored["regions"]
    if (eeg.ndim != 2 or eeg.shape[-1] < 400 or refs.shape != (2, eeg.shape[-1])
            or evaluation.shape != refs.shape
            or not all(np.isfinite(values).all() for values in (eeg, refs, evaluation))):
        raise ValueError("Native trial axes, finite samples or two-second PSD eligibility differ")
    for key in ("channel_names", "regions"):
        np.testing.assert_array_equal(stored[key], calibration[key])
    np.testing.assert_array_equal(names, record["channel_names"])
    np.testing.assert_array_equal(stored["identity"], eeg)
    mask = np.ones(len(names), dtype=bool)
    if not len(frontal_indices(eeg, names, regions, mask)):
        raise ValueError("Native trial has no verified frontal channels")
    return eeg, refs, evaluation, names, regions, mask


def predict_window(eeg, refs, calibration, names, regions, mask, bundles):
    """Fixed inference boundary: no condition labels or evaluation references."""
    predictions, details = {"identity": eeg.copy()}, {}
    for arm, gain_mode, modes in (("raw_shared", "shared", None),
                                  ("vmd_shared", "shared", bundles),
                                  ("vmd_anchored", "anchored_shared", bundles)):
        cleaned, _, details[arm] = calibration_transfer(
            eeg, refs, calibration, names, regions, mask, bundles=modes,
            gain_mode=gain_mode, anchor_weight=.25)
        predictions[arm] = cleaned
    failed = {index for index, (_, _, info) in bundles.items()
              if info.get("failure") or info.get("hit_iteration_limit")}
    predictions["vmd_anchored_raw_fallback"] = raw_failure_fallback(
        eeg, predictions["vmd_anchored"], predictions["raw_shared"],
        failed, names, regions, mask)
    outside = np.ones(len(eeg), dtype=bool)
    outside[frontal_indices(eeg, names, regions, mask)] = False
    for values in predictions.values():
        np.testing.assert_array_equal(values[outside], eeg[outside])
    return {name: eeg - values for name, values in predictions.items()}, details, sorted(failed)


def readiness(input_root, config):
    if (config.get("scope") != "frozen_native_cached_frontal_diagnostic"
            or config.get("autovmd_enabled") is not False
            or config.get("fixed_recipes") != [{"K": 3, "alpha": 2000}]
            or config.get("anchored_vmd_weight") != .25
            or config.get("reserved_access") != "closed until verified freeze"
            or config.get("selection_policy") != "first_two_record_ids_all_scoring_trials"):
        raise ValueError("Frozen native diagnostic policy differs")
    signal_bytes = Path(__file__).with_name("signal.py").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(signal_bytes).hexdigest() != config["preprocessing_module_sha256"]:
        raise ValueError("Archived native preprocessing implementation changed")
    corpus = declared_parent(input_root, "corpus_summary.json", config["calibration_parent"])
    cache = declared_parent(input_root, "native_protocol_summary.json", config["native_parent"])
    verify_parent(corpus, ("split_manifest.json",))
    if sha256_file(corpus / "split_manifest.json") != config["source_split_sha256"]:
        raise ValueError("Original native source split differs")
    previous = json.loads((cache / "configuration.json").read_text())
    if any(previous.get(key) != config[key] for key in ("fs", "window", "hop")):
        raise ValueError("Native preprocessing definitions differ")
    summary = json.loads((cache / "native_protocol_summary.json").read_text())
    if (summary.get("recordings") != 36 or summary.get("scoring_trials") != 846
            or summary.get("reserved_confirmation_opened") is not False
            or summary.get("reference_statuses") != ["publisher HEOG_lpf/VEOG_lpf"]):
        raise ValueError("Native cache scope or unchanged evaluation references differ")
    records = sorted(read_jsonl(cache / "native_source_ledger.jsonl"), key=lambda row: row["record_id"])
    if len(records) != 36 or len({row["record_id"] for row in records}) != 36:
        raise ValueError("Native source ledger coverage differs")
    for record in records:
        record_path = corpus / "calibration" / record["record_id"] / "record.json"
        validate_record(record, json.loads(record_path.read_text()))
    if [record["record_id"] for record in records[:2]] != config["pilot_record_ids"]:
        raise ValueError("Native pilot source selection differs")
    return corpus, cache, records[:2]


def preservation_reports(preservation, output, config, count):
    fields = ["relative_error", "alpha_db", "beta_db", "covariance"]
    frame = pd.DataFrame(preservation)
    strict_mean = lambda values: np.asarray(values, float).mean()
    records = frame.groupby(["source", "record_id", "method"])[fields].agg(strict_mean).reset_index()
    records.to_csv(output / "native_frontal_preservation_records.csv", index=False)
    records.groupby(["source", "method"])[fields].agg(strict_mean).reset_index().to_csv(
        output / "native_frontal_preservation_participants.csv", index=False)
    limits = config["preservation"]
    reviews = []
    for method, group in records.groupby("method"):
        finite = bool(np.isfinite(group[fields]).all().all())
        worst = {key: float(group[key].max()) if finite else None for key in fields}
        gates = {"coverage": finite and group.record_id.nunique() == count}
        for field, limit in (("relative_error", "clean_change"), ("alpha_db", "alpha_db"),
                             ("beta_db", "beta_db"), ("covariance", "covariance")):
            gates[field] = finite and worst[field] <= limits[limit]
        reviews.append({"method": method, "worst_record": worst, "gates": gates,
            "native_preservation_gates_pass": bool(all(gates.values())),
            "teacher_qualified": False, "rest_is_proven_artifact_free": False})
    atomic_json(output / "native_frontal_preservation_gates.json", reviews)


def run(input_root, output, config, profile):
    if profile != "pilot":
        raise ValueError("Frozen native diagnostic is limited to two recordings")
    corpus, cache, records = readiness(input_root, config)
    started = time.perf_counter()
    for folder in ("modes", "trial_predictions", "decisions", "record_progress"):
        (output / folder).mkdir()
    atomic_json(output / "native_cache_policy.json", {
        "input_adaptation": "archived original independently filtered trials; no second filtering",
        "units": "original source array units; physical units unverified",
        "evaluation_references": "unchanged publisher HEOG_lpf/VEOG_lpf",
        "calibration": "original unscored corpus002 calibration; trial-boundary lag exclusion",
        "selection": config["selection_policy"], "native_snr_claim": False,
        "posterior_processing": "pass-through", "teacher_qualified": False})
    metrics, preservation, spectra, failures = [], [], [], []
    trials = windows = decompositions = 0
    for record in records:
        name = record["record_id"]
        cal_path = "calibration/" + name + "/calibration.npz"
        verify_parent(corpus, (cal_path,))
        with np.load(corpus / cal_path, allow_pickle=False) as archive:
            calibration = dict(archive)
        moments = {}
        for trial in record["scoring_trial_ids"]:
            relative = f"native_trial_predictions/{name}-trial-{trial:04d}.npz"
            verify_parent(cache, (relative,))
            with np.load(cache / relative, allow_pickle=False) as archive:
                stored = dict(archive)
            eeg, refs, eval_refs, names, regions, mask = validate_trial(stored, calibration, record)
            ids = frontal_indices(eeg, names, regions, mask)
            trial_decisions = []

            def estimate(window_eeg, window_refs):
                nonlocal windows, decompositions
                bundles = decompose_frontal(window_eeg, names, regions, FixedRecipe(3), config, mask)
                arrays, info = {}, {}
                for index, (modes, residual, detail) in bundles.items():
                    check = validate_mode_reconstruction(modes, residual, window_eeg[index])
                    arrays[f"channel_{index}_modes"] = modes
                    arrays[f"channel_{index}_residual"] = residual
                    info[str(index)] = {**detail, "reconstruction": check}
                key = f"{name}-trial-{trial:04d}-window-{len(trial_decisions):03d}"
                np.savez_compressed(output / "modes" / (key + ".npz"), **arrays)
                artifacts, details, failed = predict_window(
                    window_eeg, window_refs, calibration, names, regions, mask, bundles)
                atomic_json(output / "modes" / (key + ".json"), {"channels": info, "K": 3, "alpha": 2000,
                    "original_trial_sha256": sha256_file(cache / relative),
                    "window_index": len(trial_decisions), "padding": "per-trial reflected overlap-add"})
                for index in failed:
                    failures.append({"record_id": name, "trial": trial, "window": len(trial_decisions),
                                     "channel_index": index, "channel": str(names[index]), **info[str(index)]})
                trial_decisions.append({"window": len(trial_decisions), "details": details,
                                        "failed_frontal_channels": failed})
                windows += 1
                decompositions += len(bundles)
                return artifacts

            cleaned = multi_overlap_add(eeg, refs, estimate, methods=METHODS,
                                        window=config["window"], hop=config["hop"])
            np.testing.assert_array_equal(cleaned["identity"], eeg)
            outside = np.ones(len(eeg), dtype=bool)
            outside[ids] = False
            for result in cleaned.values():
                np.testing.assert_array_equal(result[outside], eeg[outside])
            # Labels and publisher references become visible only to evaluation.
            condition = CONDITIONS[record["trial_labels"][trial]]
            for method, result in cleaned.items():
                second = (eeg[ids] if condition == "rest" else
                          np.broadcast_to(eval_refs[0 if condition == "lateral" else 1], eeg[ids].shape))
                moments.setdefault((condition, method), ConditionMoments(len(ids))).update(result[ids], second)
                if condition == "rest":
                    report = paired(result[ids], eeg[ids], config["fs"])
                    preservation.append({"source": record["participant"], "record_id": name,
                        "trial": trial, "method": method,
                        **{key: report[key] for key in ("relative_error", "alpha_db", "beta_db", "covariance")}})
                    spectra.append({"record_id": name, "method": method,
                        "segments": 1 + (eeg.shape[-1] - 2 * config["fs"]) // config["fs"],
                        "spectrum": resting_spectrum(result[ids], eeg[ids], config["fs"], normalization_band=(1., 40.))})
            np.savez_compressed(output / "trial_predictions" / f"{name}-trial-{trial:04d}.npz",
                original=eeg, inference_references=refs, evaluation_references=eval_refs,
                channel_names=names, regions=regions, **cleaned)
            write_jsonl(output / "decisions" / f"{name}-trial-{trial:04d}.jsonl", trial_decisions)
            trials += 1
            print("FROZEN_NATIVE_TRIAL_SAVED", name, trial, condition, trials, flush=True)
        for (condition, method), summary in moments.items():
            before = moments[(condition, "identity")]
            for local, index in enumerate(ids):
                metrics.append({"source": record["participant"], "record_id": name, "dataset": "osf",
                    "condition": condition, "method": method, "channel": str(names[index]), "samples": summary.count,
                    "rest_rmse": float(summary.rmse()[local]) if condition == "rest" else np.nan,
                    "eeg_eog_signed_r_after": float(summary.correlation()[local]) if condition != "rest" else np.nan,
                    "eeg_eog_abs_r_after": float(abs(summary.correlation()[local])) if condition != "rest" else np.nan,
                    "eeg_eog_abs_r_before": float(abs(before.correlation()[local])) if condition != "rest" else np.nan,
                    "reference_status": "publisher HEOG_lpf/VEOG_lpf", "units": "original array units; unverified"})
        pd.DataFrame(metrics).to_csv(output / "native_condition_channels.csv", index=False)
        pd.DataFrame(preservation).to_csv(output / "native_frontal_preservation_trials.csv", index=False)
        write_jsonl(output / "native_failures.jsonl", failures)
        atomic_json(output / "record_progress" / (name + ".json"), {
            "record_id": name, "status": "complete", "source_sha256": record["source_sha256"],
            "configuration_sha256": canonical_hash(config), "all_scoring_trial_ids": record["scoring_trial_ids"]})
    write_native_reports(pd.DataFrame(metrics), spectra, output)
    preservation_reports(preservation, output, config, len(records))
    atomic_json(output / "native_cached_frontal_summary.json", {
        "recordings": len(records), "trials": trials, "windows": windows, "decompositions": decompositions,
        "solver_flagged_channel_windows": len(failures), "runtime_s": time.perf_counter() - started,
        "methods": list(METHODS), "posterior_changed": False, "autovmd_executed": False,
        "reserved_confirmation_opened": False, "native_reconstruction_snr_claim": False,
        "teacher_qualified": False, "parameter_selection": False,
        "scope": "development-exposed two-record diagnostic, not independent validation"})
