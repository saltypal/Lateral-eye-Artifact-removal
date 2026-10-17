"""Attribute central transfer benefit with one matched raw-frontal context arm.

All previous outputs are immutable. Only the new arm's central rows change;
paired targets enter reporting after predictions have been estimated.
"""
import json
import time

import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .calibration_frontal_context import fit_context_calibration, score_context
from .experiments import corpus_parent
from .frontal_review import controlled_gates, strict_mean
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired
from .nonfrontal_ridge import fit_frozen_ridge
from .paper_evaluation import save_protocol
from .paper_metrics import paired_channels
from .posterior_context import PAPER_KEYS
from .regional_diagnostic import assert_parity
from .regional_coverage import ARMS as PREVIOUS_ARMS
from .regional_coverage import BASE, COMPARATOR, central_compose, save_fit, select_rows, verify_arrays
from .regional_coverage import readiness as coverage_readiness
from .regional_integration import unique_artifact


RAW_ARM = "central_frontal_raw_context"
VMD_ARM = "central_frontal_vmd_context"
ARMS = PREVIOUS_ARMS + (RAW_ARM,)


def validate_coverage_identity(parent, config):
    expected = config["matched_coverage_parent"]
    if sha256_file(parent / "artifact_manifest.json") != expected["manifest_sha256"]:
        raise ValueError("Matched coverage parent manifest differs")
    verify_parent(parent, ("coverage_summary.json", "run_spec.json", "stage_tests_summary.json",
                           "coverage_selection_manifest.jsonl", "coverage_example_scores.csv",
                           "coverage_channel_scores.csv.gz", "coverage_method_means.csv"))
    spec = json.loads((parent / "run_spec.json").read_text())
    summary = json.loads((parent / "coverage_summary.json").read_text())
    tests = json.loads((parent / "stage_tests_summary.json").read_text())
    for key in ("run_id", "git_sha", "campaign_id"):
        if spec.get(key) != expected[key]:
            raise ValueError("Matched coverage source identity differs: " + key)
    if (summary.get("profile") != "full" or summary.get("examples") != 320
            or summary.get("recipients") != 32 or summary.get("arms") != list(PREVIOUS_ARMS)
            or summary.get("frozen_branch_parity_checks") != 3840
            or tests.get("passed") is not True):
        raise ValueError("Matched coverage parent is not the verified full comparison")
    if summary.get("source_split_sha256") != config["diagnostic_parent"]["split_sha256"]:
        raise ValueError("Matched coverage source split differs")
    for key in ("autovmd_executed", "weights_trained", "reserved_confirmation_opened", "scoring_target_fit"):
        if summary.get(key) is not False:
            raise ValueError("Matched parent scope differs: " + key)
    return summary


def readiness(input_root, config):
    if config.get("scope") != "matched_raw_frontal_context_with_frozen_outputs":
        raise ValueError("Matched context scope differs")
    regional = coverage_readiness(input_root, {**config, "scope": "verified_central_extension_of_frozen_regional"})
    coverage = unique_artifact(input_root, "coverage_summary.json").parent
    validate_coverage_identity(coverage, config)
    if config.get("context_match_policy") != "one raw OOF context arm; same four chunks; frozen scored frontal/posterior":
        raise ValueError("Matched context comparison policy differs")
    return regional, coverage


def fit_raw_context(calibration, calibration_path, output, config):
    """Fit only on unscored calibration with the same bounded OOF policy."""
    central = np.flatnonzero(calibration["mask"] & (calibration["regions"] == 2))
    cache_id = sha256_file(calibration_path)
    destination = output / "coverage_fits" / cache_id
    destination.mkdir(parents=True, exist_ok=True)
    context = fit_context_calibration(calibration, config, destination / "raw_context", cache_id,
                                      representation="raw")
    record = {"calibration_sha256": cache_id, "representation": "raw", "context": context["metadata"],
              "paired_target_access": False, "scoring_eeg_fit": False,
              "central_channels": calibration["channel_names"][central].tolist()}
    model, failure = None, None
    if context["metadata"]["available"]:
        try:
            policy = config["coverage_recipe"]
            model = fit_frozen_ridge(context["eeg"][central], context["predictors"], context["boundaries"],
                                     tuple(policy["lags"]), policy["penalty"])
            save_fit(destination / "raw_context_ridge.npz", model)
            record["context_fit"] = model.metadata
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            failure = f"{type(error).__name__}: {error}"
    else:
        failure = context["metadata"].get("failure", "Raw calibration context unavailable")
    record["failure"] = failure
    atomic_json(destination / "fit_manifest.json", record)
    return model, failure, record


def estimate_raw_central(frontal_artifact, names, regions, mask, hemispheres, fitted):
    """Predict from an estimated frontal field, never clean truth or mixture strength."""
    model, failure, _ = fitted
    features, flags = score_context(frontal_artifact, names, regions, mask, hemispheres)
    count = int(np.sum(np.asarray(mask) & (np.asarray(regions) == 2)))
    artifact = np.zeros((count, frontal_artifact.shape[-1]), dtype=float)
    if model is not None and flags["available"]:
        artifact = model.artifact(features)
    return artifact, failure, {"score_context": flags,
        "zero_or_missing_context_pass_through": bool(model is not None and not flags["available"])}


def assert_parent_predictions(previous, saved, data):
    """Require exact whole-waveform parity for the two original frozen controls."""
    for archived, current in (("original", "eeg"), ("target", "paired_reference"),
                              ("names", "channel_names"), ("regions", "regions"), ("mask", "mask")):
        if not np.array_equal(previous[archived], data[current]):
            raise ValueError("Matched context/corpus arrays differ: " + archived)
    if not np.array_equal(previous["frozen_regional"], saved[BASE]):
        raise ValueError("Coverage frozen baseline differs from original regional parent")
    if not np.array_equal(previous["raw_frontal_comparator"], saved[COMPARATOR]):
        raise ValueError("Coverage raw comparator differs from original regional parent")
    for arm in PREVIOUS_ARMS:
        if previous[arm].shape != data["eeg"].shape or not np.isfinite(previous[arm]).all():
            raise ValueError("Invalid previous prediction: " + arm)


def plot_results(output):
    import matplotlib.pyplot as plt
    means = pd.read_csv(output / "coverage_method_means.csv")
    for scope in ("whole", "central_shared"):
        table = means[(means.scope == scope) & means.evaluated_family.isin(("frozen_regional", VMD_ARM, RAW_ARM))]
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.barh(table.evaluated_family, table.snr_energy_db)
        axis.axvline(15, color="black", linestyle="--", label="15 dB project target")
        axis.set_xlabel("Channel-first recipient-balanced output SNR (dB)")
        axis.set_title(scope + ": matched context development comparison")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / ("coverage_" + scope + ".png"), dpi=140)
        plt.close(figure)


def run(input_root, output, config, profile):
    started = time.perf_counter()
    regional, coverage = readiness(input_root, config)
    corpus, corpus_rows = corpus_parent(input_root)
    if sha256_file(corpus / "split_manifest.json") != config["diagnostic_parent"]["split_sha256"]:
        raise ValueError("Source partitions differ")
    all_rows = read_jsonl(coverage / "coverage_selection_manifest.jsonl")
    if all_rows != read_jsonl(regional / "regional_selection_manifest.jsonl"):
        raise ValueError("Matched parents use different selected examples")
    mapping = {row["example_id"]: row for row in corpus_rows}
    if any(mapping.get(row["example_id"]) != row for row in all_rows):
        raise ValueError("Matched corpus identities differ")
    rows = select_rows(all_rows, profile)
    write_jsonl(output / "coverage_selection_manifest.jsonl", rows)
    save_protocol(output)
    predictions, shards = output / "coverage_predictions", output / "coverage_metric_shards"
    predictions.mkdir()
    shards.mkdir()
    inherited = pd.read_csv(coverage / "coverage_example_scores.csv")
    inherited = inherited[inherited.scope == "whole"]
    failures_by_arm = inherited.set_index(["example_id", "arm"], verify_integrity=True).failure.to_dict()
    fits, scores, examples, decisions = {}, [], [], []
    parity_count, preserved_predictions = 0, 0
    for number, row in enumerate(rows):
        score_start, example_start = len(scores), len(examples)
        verify_parent(corpus, (row["array_path"], row["calibration"]))
        regional_path = "regional_predictions/" + row["example_id"] + ".npz"
        coverage_path = "coverage_predictions/" + row["example_id"] + ".npz"
        verify_parent(regional, (regional_path,))
        verify_parent(coverage, (coverage_path,))
        data = dict(np.load(corpus / row["array_path"], allow_pickle=False))
        calibration = dict(np.load(corpus / row["calibration"], allow_pickle=False))
        saved = dict(np.load(regional / regional_path, allow_pickle=False))
        previous = dict(np.load(coverage / coverage_path, allow_pickle=False))
        available, groups = verify_arrays(saved, data, calibration)
        assert_parent_predictions(previous, saved, data)
        key = row["calibration"]
        if key not in fits:
            print("MATCHED_RAW_CALIBRATION_START", key, flush=True)
            fits[key] = fit_raw_context(calibration, corpus / key, output, config)
            old_fit_path = "coverage_fits/" + sha256_file(corpus / key) + "/fit_manifest.json"
            verify_parent(coverage, (old_fit_path,))
            old_fit = json.loads((coverage / old_fit_path).read_text())
            new_chunks = fits[key][2]["context"].get("selected_chunks")
            if new_chunks != old_fit["context"].get("selected_chunks"):
                raise ValueError("Raw/VMD context calibration chunk selection differs")
            policy_fields = ["transfer_parameters", "crossfit_policy"]
            if fits[key][2]["context"]["available"] and old_fit["context"]["available"]:
                policy_fields += ["chunk_join_boundaries", "context_feature_order"]
            for field in policy_fields:
                if fits[key][2]["context"].get(field) != old_fit["context"].get(field):
                    raise ValueError("Raw/VMD context calibration policy differs: " + field)
        frontal_artifact = np.zeros_like(data["eeg"], dtype=float)
        frontal = groups["frontal"]
        frontal_artifact[frontal] = data["eeg"][frontal] - saved[COMPARATOR][frontal]
        artifact, failure, details = estimate_raw_central(frontal_artifact, data["channel_names"],
            data["regions"], data["mask"], data["hemispheres"], fits[key])
        estimates = {arm: previous[arm].copy() for arm in PREVIOUS_ARMS}
        estimates[RAW_ARM] = central_compose(data["eeg"], saved[BASE], groups["central_shared"], artifact)
        for arm in PREVIOUS_ARMS:
            if not np.array_equal(estimates[arm], previous[arm]):
                raise ValueError("Previous scored waveform changed: " + arm)
            preserved_predictions += 1
        for region in ("frontal", "posterior", "unknown"):
            if not np.array_equal(estimates[RAW_ARM][groups[region]], estimates[VMD_ARM][groups[region]]):
                raise ValueError("Matched context altered frozen region: " + region)
            parity_count += 1
        meta = {key: row[key] for key in ("example_id", "recipient", "donor", "condition", "input_snr_db")}
        meta["fold"] = row["partition"]["fold"]
        scopes = {"whole": available, **{key: value for key, value in groups.items() if len(value)}}
        for arm, cleaned in estimates.items():
            prior = failures_by_arm[(row["example_id"], "frozen_regional" if arm == RAW_ARM else arm)]
            prior = prior if isinstance(prior, str) and prior else None
            combined_failure = "; ".join(value for value in (prior, failure if arm == RAW_ARM else None) if value) or None
            for scope, ids in scopes.items():
                target, original = data["paired_reference"][ids], data["eeg"][ids]
                paper = paired_channels(cleaned[ids], target, original)
                comparator = paired_channels(saved[COMPARATOR][ids], target, original)
                engineering = paired(cleaned[ids], target, config["fs"])
                examples.append({**meta, "arm": arm, "scope": scope, "failure": combined_failure,
                    "pooled_energy_snr_db": engineering["snr_db"], "clean_change": engineering["relative_error"],
                    "alpha_db": engineering["alpha_db"], "beta_db": engineering["beta_db"],
                    "covariance_error": engineering["covariance"]})
                for local, index in enumerate(ids):
                    scores.append({**meta, "scope": scope, "channel": str(data["channel_names"][index]),
                        "evaluated_family": arm, "failure": combined_failure,
                        **{name: float(value[local]) for name, value in paper.items()},
                        "cc_deterioration": float(comparator["pearson_cc"][local] - paper["pearson_cc"][local]),
                        "clean_change": float(paper["rrmse_time"][local]), "alpha_db": engineering["alpha_db"],
                        "beta_db": engineering["beta_db"], "covariance_error": engineering["covariance"]})
        np.savez_compressed(predictions / (row["example_id"] + ".npz"), original=data["eeg"],
            target=data["paired_reference"], names=data["channel_names"], regions=data["regions"], mask=data["mask"], **estimates)
        decision = {**meta, "calibration": key, "raw_context_failure": failure, **details}
        decisions.append(decision)
        atomic_json(predictions / (row["example_id"] + ".json"), decision)
        pd.DataFrame(scores[score_start:]).to_csv(shards / (row["example_id"] + "-channels.csv.gz"), index=False, compression="gzip")
        pd.DataFrame(examples[example_start:]).to_csv(shards / (row["example_id"] + "-examples.csv"), index=False)
        atomic_json(output / "coverage_progress.json", {"completed_examples": number + 1, "total_examples": len(rows)})
        print("MATCHED_CONTEXT_SAVED", number + 1, len(rows), flush=True)
    frame = pd.DataFrame(scores)
    old_scores = pd.read_csv(coverage / "coverage_channel_scores.csv.gz")
    expected_scores = old_scores[old_scores.example_id.isin([row["example_id"] for row in rows])]
    metric_keys = ["example_id", "scope", "channel", "evaluated_family"]
    preserved_scores = frame[frame.evaluated_family.isin(PREVIOUS_ARMS)]
    if (len(preserved_scores) != len(expected_scores)
            or preserved_scores.duplicated(metric_keys).any() or expected_scores.duplicated(metric_keys).any()):
        raise ValueError("Preserved metric coverage differs")
    reconciled = preserved_scores.merge(expected_scores, on=metric_keys, suffixes=("_new", "_old"), validate="one_to_one")
    if len(reconciled) != len(expected_scores):
        raise ValueError("Preserved metric keys differ")
    for metric in PAPER_KEYS:
        assert_parity(reconciled[metric + "_new"], reconciled[metric + "_old"], metric)
    frame.to_csv(output / "coverage_channel_scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame(examples).to_csv(output / "coverage_example_scores.csv", index=False)
    write_jsonl(output / "coverage_decisions.jsonl", decisions)
    write_jsonl(output / "coverage_calibration_fits.jsonl", [value[2] for value in fits.values()])
    fields = list(PAPER_KEYS) + ["cc_deterioration"]
    dirty = frame[frame.condition != "clean"]
    strata = dirty.groupby(["scope", "evaluated_family", "recipient", "condition", "input_snr_db"])[fields].agg(strict_mean)
    recipients = strata.groupby(["scope", "evaluated_family", "recipient"]).agg(strict_mean)
    recipients.to_csv(output / "coverage_recipient_means.csv")
    recipients.groupby(["scope", "evaluated_family"]).agg(strict_mean).to_csv(output / "coverage_method_means.csv")
    strata.groupby(["scope", "evaluated_family", "condition", "input_snr_db"]).agg(strict_mean).to_csv(output / "coverage_condition_means.csv")
    atomic_json(output / "coverage_controlled_gates.json", {scope: controlled_gates(part, config) for scope, part in frame.groupby("scope")})
    from .coverage_uncertainty import uncertainty_report
    uncertainty_report(frame.rename(columns={"evaluated_family": "arm"}), output,
                       comparators=("frozen_regional", VMD_ARM), candidates=(RAW_ARM,), seed=42, repetitions=2000)
    # The original full coverage table is also analyzed without repeating its fits.
    old_output = output / "previous_coverage_uncertainty"
    old_output.mkdir()
    uncertainty_report(old_scores.rename(columns={"evaluated_family": "arm"}), old_output,
                       comparators=("frozen_regional",), candidates=(VMD_ARM,), seed=42, repetitions=2000)
    atomic_json(output / "coverage_summary.json", {"profile": profile, "examples": len(rows),
        "recipients": len({row["recipient"] for row in rows}), "arms": list(ARMS), "calibrations": len(fits),
        "frozen_branch_parity_checks": parity_count, "preserved_parent_predictions": preserved_predictions,
        "preserved_parent_metric_checks": len(reconciled) * len(PAPER_KEYS),
        "raw_frontal_context_available_calibrations": sum(value[0] is not None for value in fits.values()),
        "source_split_sha256": sha256_file(corpus / "split_manifest.json"),
        "autovmd_executed": False, "weights_trained": False, "reserved_confirmation_opened": False,
        "qualification": False, "native_preservation_unresolved": True, "scoring_target_fit": False,
        "VMD_scope": "verified frontal K3/2000 archived outputs only; no new decomposition",
        "calibration_context_max_chunks": 4, "raw_context_runtime_vmd": False,
        "controlled_reference_amplitude_mismatch": True, "runtime_s": time.perf_counter() - started,
        "comparator": COMPARATOR, "context_comparator": VMD_ARM,
        "scoring": "channel-first participant-balanced primary; pooled energy supplementary"})
    plot_results(output)
