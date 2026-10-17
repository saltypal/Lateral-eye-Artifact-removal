"""Bounded central/shared extension of immutable frontal/posterior predictions.

Calibration fits accept no paired targets. Scored targets enter only reporting.
This stage is exploratory; completion never qualifies a teacher or deployment.
"""
import json
import time

import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .data import _region
from .experiments import corpus_parent
from .frontal_event_support import fit_axis_amplitude_support
from .frontal_review import controlled_gates, strict_mean
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired
from .nonfrontal_ridge import fit_frozen_ridge, apply_axis_gate
from .paper_evaluation import save_protocol
from .paper_metrics import paired_channels
from .posterior_context import PAPER_KEYS
from .regional_diagnostic import groups_for, validate_rows
from .regional_diagnostic import readiness as diagnostic_readiness
from .regional_integration import unique_artifact


BASE = "K3_calibration_mode_gain+direct:posterior"
COMPARATOR = "calibration_raw_gain+direct:posterior"
ARMS = ("identity", "frozen_regional", "central_frozen_ridge",
        "central_axis_gated_ridge", "central_frontal_vmd_context", "raw_frontal_comparator")


def readiness(input_root, config):
    if config.get("scope") != "verified_central_extension_of_frozen_regional":
        raise ValueError("Central coverage scope differs")
    inherited = {**config, "scope": "frozen_regional_error_diagnostic"}
    regional, _ = diagnostic_readiness(input_root, inherited)
    diagnostic = unique_artifact(input_root, "diagnostic_summary.json").parent
    verify_parent(diagnostic, ("diagnostic_summary.json", "run_spec.json", "stage_tests_summary.json"))
    if sha256_file(diagnostic / "artifact_manifest.json") != config["coverage_diagnostic_manifest_sha256"]:
        raise ValueError("Coverage diagnostic manifest differs")
    summary = json.loads((diagnostic / "diagnostic_summary.json").read_text())
    tests = json.loads((diagnostic / "stage_tests_summary.json").read_text())
    spec = json.loads((diagnostic / "run_spec.json").read_text())
    if any(spec.get(key) != value for key, value in config["coverage_diagnostic_identity"].items()):
        raise ValueError("Coverage diagnostic identity differs")
    if (summary.get("profile") != "full" or summary.get("examples") != 320
            or summary.get("correction_changed") is not False or tests.get("passed") is not True):
        raise ValueError("Full frozen diagnostic did not verify")
    for key in ("autovmd_executed", "weights_trained", "reserved_confirmation_opened"):
        if summary.get(key) is not False:
            raise ValueError("Diagnostic scope differs: " + key)
    policy = config.get("coverage_recipe", {})
    if policy != {"lags": [-10, 0, 10], "penalty": .01, "max_calibration_chunks": 4,
                  "frontal_threshold": .4, "gain_mode": "shared"}:
        raise ValueError("Coverage fixed recipe differs")
    if config.get("fixed_recipes") != [{"K": 3, "alpha": 2000}]:
        raise ValueError("Only fixed frontal K3/2000 is authorized")
    return regional


def select_rows(rows, profile):
    validate_rows(rows, 320, 32)
    if profile == "full":
        return rows
    if profile != "pilot":
        raise ValueError("Unknown profile")
    first = {}
    for row in rows:
        first.setdefault(row["partition"]["fold"], row["recipient"])
    chosen = [row for row in rows if first[row["partition"]["fold"]] == row["recipient"]]
    if len(chosen) != 50 or len(first) != 5:
        raise ValueError("Pilot must retain five recipients and all ten conditions/strengths")
    return chosen


def central_compose(original, frozen, central_ids, central_artifact):
    """Compose disjoint original-input artifacts once, retaining frozen branches."""
    original, frozen = np.asarray(original), np.asarray(frozen)
    ids = np.asarray(central_ids, dtype=int)
    artifact = np.asarray(central_artifact, dtype=float)
    if (original.ndim != 2 or frozen.shape != original.shape
            or artifact.shape != (len(ids), original.shape[-1])
            or not np.isfinite(frozen).all() or not np.isfinite(artifact).all()
            or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(original))):
        raise ValueError("Invalid central composition axes/indices")
    if not np.array_equal(frozen[ids], original[ids]):
        raise ValueError("Central parent output was not pass-through")
    result = frozen.copy()
    result[ids] = original[ids] - artifact
    rest = np.setdiff1d(np.arange(len(original)), ids)
    if not np.array_equal(result[rest], frozen[rest]):
        raise ValueError("Central composition altered frozen outputs")
    return result


def save_fit(path, model):
    np.savez_compressed(path, coefficients=model.coefficients,
                        reference_mean=model.reference_mean, reference_scale=model.reference_scale,
                        active_columns=model.active_columns, lags=np.asarray(model.lags), penalty=model.penalty)


def verify_arrays(saved, data, calibration):
    for archived, current in (("original", "eeg"), ("target", "paired_reference"),
                              ("names", "channel_names"), ("regions", "regions"), ("mask", "mask")):
        if not np.array_equal(saved[archived], data[current]):
            raise ValueError("Regional/corpus waveform or metadata mismatch: " + archived)
    for key in ("channel_names", "regions", "mask", "hemispheres"):
        if not np.array_equal(calibration[key], data[key]):
            raise ValueError("Calibration/scoring channel metadata mismatch: " + key)
    available, groups = groups_for(saved)
    central = groups["central_shared"]
    if not len(central):
        raise ValueError("No verified central/shared outputs")
    if any(_region(str(data["channel_names"][index])) != 2 for index in central):
        raise ValueError("Central/shared labels require verified electrode names")
    for label in (BASE, COMPARATOR):
        prediction = saved[label]
        if prediction.shape != data["eeg"].shape or not np.isfinite(prediction).all():
            raise ValueError("Invalid frozen prediction")
        unchanged = np.concatenate([central, groups["unknown"], np.flatnonzero(~data["mask"])])
        if not np.array_equal(prediction[unchanged], data["eeg"][unchanged]):
            raise ValueError("Frozen central/unknown/masked parent was changed")
    return available, groups


def fit_calibration(calibration, path, output, config):
    from .calibration_frontal_context import fit_context_calibration
    ids = np.flatnonzero(calibration["mask"] & (calibration["regions"] == 2))
    policy = config["coverage_recipe"]
    cache_id = sha256_file(path)
    destination = output / "coverage_fits" / cache_id
    destination.mkdir(parents=True, exist_ok=True)
    record = {"calibration_sha256": cache_id, "central_channels": calibration["channel_names"][ids].tolist(),
              "paired_target_access": False, "scoring_eeg_fit": False}
    ridge = detector = context_model = None
    failures = {}
    try:
        ridge = fit_frozen_ridge(calibration["eeg"][ids], calibration["references"],
                                 calibration["boundaries"], tuple(policy["lags"]), policy["penalty"])
        save_fit(destination / "eog_ridge.npz", ridge)
        record["eog_fit"] = ridge.metadata
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        failures["central_frozen_ridge"] = f"{type(error).__name__}: {error}"
    try:
        detector = fit_axis_amplitude_support(calibration)
        record["detector"] = {"center": detector.center.tolist(), "thresholds": detector.thresholds.tolist(),
                              "calibration_counts": detector.calibration_counts}
    except (ValueError, RuntimeError) as error:
        failures["central_axis_gated_ridge"] = f"{type(error).__name__}: {error}"
    context = fit_context_calibration(calibration, config, destination / "frontal_context", cache_id)
    record["context"] = context["metadata"]
    if context["metadata"]["available"]:
        try:
            context_model = fit_frozen_ridge(context["eeg"][ids], context["predictors"],
                                             context["boundaries"], tuple(policy["lags"]), policy["penalty"])
            save_fit(destination / "context_ridge.npz", context_model)
            record["context_fit"] = context_model.metadata
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            failures["central_frontal_vmd_context"] = f"{type(error).__name__}: {error}"
    else:
        failures["central_frontal_vmd_context"] = context["metadata"].get("failure", "Calibration context unavailable")
    if ridge is None:
        failures.setdefault("central_axis_gated_ridge", failures["central_frozen_ridge"])
    record["failures"] = failures
    atomic_json(destination / "fit_manifest.json", record)
    return ridge, detector, context_model, failures, record


def estimate_central(references, frontal_artifact, names, regions, mask, hemispheres, fits):
    """Inference accepts predictors and frozen fits; never scoring EEG or targets."""
    from .calibration_frontal_context import score_context
    ridge, detector, context_model, fit_failures, _ = fits
    count = int(np.sum(np.asarray(mask) & (np.asarray(regions) == 2)))
    zeros = np.zeros((count, np.asarray(references).shape[-1]), dtype=float)
    artifacts = {name: zeros.copy() for name in ARMS[1:5]}
    failures = dict(fit_failures)
    details = {}
    if ridge is not None:
        artifacts["central_frozen_ridge"] = ridge.artifact(references)
        if detector is not None:
            artifacts["central_axis_gated_ridge"], details["axis_gate"] = apply_axis_gate(
                ridge.axis_artifacts(references), references, detector)
    predictors, flags = score_context(frontal_artifact, names, regions, mask, hemispheres)
    details["score_context"] = flags
    if context_model is not None and flags["available"] and np.any(predictors):
        artifacts["central_frontal_vmd_context"] = context_model.artifact(predictors)
    elif context_model is not None:
        details["context_pass_through"] = "absent or identically zero frontal estimated field"
    return artifacts, failures, details


def plot_results(output):
    import matplotlib.pyplot as plt
    means = pd.read_csv(output / "coverage_method_means.csv")
    for scope in ("whole", "central_shared"):
        table = means[means.scope == scope]
        figure, axis = plt.subplots(figsize=(11, 4))
        axis.barh(table.evaluated_family, table.snr_energy_db)
        axis.axvline(15, color="black", linestyle="--", label="15 dB project target")
        axis.set_xlabel("Channel-first recipient-balanced output SNR (dB)")
        axis.set_title(scope + ": controlled development; qualification pending")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / ("coverage_" + scope + ".png"), dpi=140)
        plt.close(figure)


def run(input_root, output, config, profile):
    started = time.perf_counter()
    regional = readiness(input_root, config)
    corpus, corpus_rows = corpus_parent(input_root)
    if sha256_file(corpus / "split_manifest.json") != config["diagnostic_parent"]["split_sha256"]:
        raise ValueError("Source partitions differ")
    original_rows = read_jsonl(regional / "regional_selection_manifest.jsonl")
    mapping = {row["example_id"]: row for row in corpus_rows}
    for row in original_rows:
        if mapping.get(row["example_id"]) != row:
            raise ValueError("Selected corpus identity or source metadata differs")
    rows = select_rows(original_rows, profile)
    write_jsonl(output / "coverage_selection_manifest.jsonl", rows)
    save_protocol(output)
    predictions = output / "coverage_predictions"
    predictions.mkdir()
    metric_shards = output / "coverage_metric_shards"
    metric_shards.mkdir()
    inherited = pd.read_csv(regional / "regional_example_scores.csv")
    inherited = inherited[(inherited.scope == "whole") & inherited.arm.isin((BASE, COMPARATOR))]
    inherited_failures = inherited.set_index(["example_id", "arm"], verify_integrity=True).failure.to_dict()
    calibration_fits, scores, examples, decisions = {}, [], [], []
    parity_count = 0
    for number, row in enumerate(rows):
        score_start, example_start = len(scores), len(examples)
        paths = (row["array_path"], row["calibration"])
        verify_parent(corpus, paths)
        prediction_path = "regional_predictions/" + row["example_id"] + ".npz"
        verify_parent(regional, (prediction_path,))
        data = dict(np.load(corpus / row["array_path"], allow_pickle=False))
        cal = dict(np.load(corpus / row["calibration"], allow_pickle=False))
        saved = dict(np.load(regional / prediction_path, allow_pickle=False))
        available, groups = verify_arrays(saved, data, cal)
        central = groups["central_shared"]
        key = row["calibration"]
        if key not in calibration_fits:
            print("COVERAGE_CALIBRATION_START", key, flush=True)
            calibration_fits[key] = fit_calibration(cal, corpus / key, output, config)
        frontal_artifact = np.zeros_like(data["eeg"], dtype=float)
        frontal_artifact[groups["frontal"]] = data["eeg"][groups["frontal"]] - saved[BASE][groups["frontal"]]
        artifacts, failures, details = estimate_central(data["references"], frontal_artifact,
            data["channel_names"], data["regions"], data["mask"], data["hemispheres"], calibration_fits[key])
        estimates = {"identity": data["eeg"], "raw_frontal_comparator": saved[COMPARATOR]}
        for arm, artifact in artifacts.items():
            estimates[arm] = central_compose(data["eeg"], saved[BASE], central, artifact)
            for region in ("frontal", "posterior", "unknown"):
                if not np.array_equal(estimates[arm][groups[region]], saved[BASE][groups[region]]):
                    raise ValueError("Frozen branch parity failed: " + region)
                parity_count += 1
        if not np.array_equal(estimates["frozen_regional"], saved[BASE]):
            raise ValueError("Frozen baseline parity failed")
        meta = {key: row[key] for key in ("example_id", "recipient", "donor", "condition", "input_snr_db")}
        meta["fold"] = row["partition"]["fold"]
        scopes = {"whole": available, **{key: value for key, value in groups.items() if len(value)}}
        for arm, cleaned in estimates.items():
            prior = inherited_failures[(row["example_id"], COMPARATOR if arm == "raw_frontal_comparator" else BASE)] if arm != "identity" else None
            prior = prior if isinstance(prior, str) and prior else None
            failure = "; ".join(value for value in (prior, failures.get(arm)) if value) or None
            for scope, ids in scopes.items():
                # Target access starts here, after every correction was estimated.
                target, original = data["paired_reference"][ids], data["eeg"][ids]
                paper = paired_channels(cleaned[ids], target, original)
                comparator = paired_channels(saved[COMPARATOR][ids], target, original)
                engineering = paired(cleaned[ids], target, config["fs"])
                examples.append({**meta, "arm": arm, "scope": scope, "failure": failure,
                    "pooled_energy_snr_db": engineering["snr_db"], "clean_change": engineering["relative_error"],
                    "alpha_db": engineering["alpha_db"], "beta_db": engineering["beta_db"],
                    "covariance_error": engineering["covariance"]})
                for local, index in enumerate(ids):
                    scores.append({**meta, "scope": scope, "channel": str(data["channel_names"][index]),
                        "evaluated_family": arm, "failure": failure,
                        **{name: float(value[local]) for name, value in paper.items()},
                        "cc_deterioration": float(comparator["pearson_cc"][local] - paper["pearson_cc"][local]),
                        "clean_change": float(paper["rrmse_time"][local]), "alpha_db": engineering["alpha_db"],
                        "beta_db": engineering["beta_db"], "covariance_error": engineering["covariance"]})
        np.savez_compressed(predictions / (row["example_id"] + ".npz"), original=data["eeg"],
            target=data["paired_reference"], names=data["channel_names"], regions=data["regions"], mask=data["mask"], **estimates)
        record = {**meta, "calibration": key, "failures": failures, **details}
        decisions.append(record)
        atomic_json(predictions / (row["example_id"] + ".json"), record)
        pd.DataFrame(scores[score_start:]).to_csv(metric_shards / (row["example_id"] + "-channels.csv.gz"),
                                               index=False, compression="gzip")
        pd.DataFrame(examples[example_start:]).to_csv(metric_shards / (row["example_id"] + "-examples.csv"), index=False)
        atomic_json(output / "coverage_progress.json", {"completed_examples": number + 1,
                    "total_examples": len(rows), "calibrations": len(calibration_fits)})
        print("COVERAGE_SAVED", number + 1, len(rows), flush=True)
    frame = pd.DataFrame(scores)
    frame.to_csv(output / "coverage_channel_scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame(examples).to_csv(output / "coverage_example_scores.csv", index=False)
    write_jsonl(output / "coverage_decisions.jsonl", decisions)
    write_jsonl(output / "coverage_calibration_fits.jsonl", [value[4] for value in calibration_fits.values()])
    fields = list(PAPER_KEYS) + ["cc_deterioration"]
    dirty = frame[frame.condition != "clean"]
    strata = dirty.groupby(["scope", "evaluated_family", "recipient", "condition", "input_snr_db"])[fields].agg(strict_mean)
    recipients = strata.groupby(["scope", "evaluated_family", "recipient"]).agg(strict_mean)
    recipients.to_csv(output / "coverage_recipient_means.csv")
    recipients.groupby(["scope", "evaluated_family"]).agg(strict_mean).to_csv(output / "coverage_method_means.csv")
    strata.groupby(["scope", "evaluated_family", "condition", "input_snr_db"]).agg(strict_mean).to_csv(output / "coverage_condition_means.csv")
    atomic_json(output / "coverage_controlled_gates.json", {scope: controlled_gates(part, config)
                for scope, part in frame.groupby("scope")})
    atomic_json(output / "coverage_summary.json", {"profile": profile, "examples": len(rows),
        "recipients": len({row["recipient"] for row in rows}), "arms": list(ARMS),
        "calibrations": len(calibration_fits), "frozen_branch_parity_checks": parity_count,
        "reference_ridge_available_calibrations": sum(value[0] is not None for value in calibration_fits.values()),
        "axis_detector_available_calibrations": sum(value[1] is not None for value in calibration_fits.values()),
        "frontal_context_available_calibrations": sum(value[2] is not None for value in calibration_fits.values()),
        "source_split_sha256": sha256_file(corpus / "split_manifest.json"),
        "autovmd_executed": False, "weights_trained": False, "reserved_confirmation_opened": False,
        "qualification": False, "native_preservation_unresolved": True,
        "VMD_scope": "verified frontal K3/2000; bounded OOF calibration cache only",
        "calibration_context_max_chunks": 4, "scoring_target_fit": False,
        "controlled_reference_amplitude_mismatch": True,
        "runtime_s": time.perf_counter() - started, "comparator": COMPARATOR,
        "scoring": "channel-first participant-balanced primary; pooled energy supplementary"})
    plot_results(output)
