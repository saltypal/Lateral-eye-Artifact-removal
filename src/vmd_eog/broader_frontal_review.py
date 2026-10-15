"""Frozen five-arm development validation; all numerical execution is on Kaggle.

This expands source coverage, not recipe selection. Calibration is observed
input; paired references enter scoring only after all five predictions exist.
"""
import json
import time

import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .broader_coupled_corpus import declared_parent
from .contracts import canonical_hash
from .coupled_frontal_review import regional_direct, validate_mode_reconstruction
from .coverage_uncertainty import uncertainty_report
from .frontal_calibration_transfer import calibration_transfer
from .frontal_methods import FixedRecipe, decompose_frontal
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired, preservation_pass
from .paper_metrics import paired_channels
from .reference import correlations


ARMS = ("identity", "raw_shared", "raw_axis", "vmd_shared", "vmd_anchored")
FRONTAL_ARMS = (("raw_shared", "shared", False), ("raw_axis", "axis", False),
                ("vmd_shared", "shared", True), ("vmd_anchored", "anchored_shared", True))


def compose_regions(original, frontal, nonfrontal, regions, mask):
    """Subtract disjoint original-input fields once; unavailable rows pass."""
    original = np.asarray(original, dtype=np.float64)
    frontal, nonfrontal = np.asarray(frontal), np.asarray(nonfrontal)
    regions, mask = np.asarray(regions), np.asarray(mask, dtype=bool)
    if (original.ndim != 2 or frontal.shape != original.shape or nonfrontal.shape != original.shape
            or regions.shape != (len(original),) or mask.shape != regions.shape
            or not all(np.isfinite(value).all() for value in (original, frontal, nonfrontal))):
        raise ValueError("Invalid original-input regional fields")
    if (np.any(frontal[(regions != 0) | ~mask])
            or np.any(nonfrontal[~(np.isin(regions, (1, 2)) & mask)])):
        raise ValueError("Regional fields overlap or correct unavailable anatomy")
    return original - (frontal + nonfrontal)


def readiness(input_root, config):
    if (config.get("scope") != "broader_fixed_anchored_frontal_validation"
            or config.get("autovmd_enabled") is not False
            or config.get("fixed_recipes") != [{"K": 3, "alpha": 2000}]
            or config.get("anchored_vmd_weight") != .25
            or config.get("reserved_access") != "closed until verified freeze"):
        raise ValueError("Frozen broader frontal policy differs")
    parent = declared_parent(input_root, "broader_corpus_summary.json", config["review_parent"])
    verify_parent(parent, ("corpus_manifest.jsonl", "source_selection.jsonl", "stage_tests_summary.json"))
    summary = json.loads((parent / "broader_corpus_summary.json").read_text())
    if (summary.get("examples") != 320 or summary.get("pairs") != 32
            or summary.get("construction_contracts_passed") is not True
            or summary.get("cohort_sha256") != config["cohort_sha256"]
            or summary.get("source_split_sha256") != config["source_split_sha256"]
            or summary.get("reserved_confirmation_opened") is not False
            or json.loads((parent / "stage_tests_summary.json").read_text()).get("passed") is not True):
        raise ValueError("Broader coupled parent failed readiness")
    if canonical_hash(read_jsonl(parent / "source_selection.jsonl")) != config["cohort_sha256"]:
        raise ValueError("Broader review source membership differs")
    rows = read_jsonl(parent / "corpus_manifest.jsonl")
    if (len(rows) != 320 or len({row["example_id"] for row in rows}) != 320
            or len({row["recipient"] for row in rows}) != 32
            or any(row["partition"]["role"] != "development" for row in rows)):
        raise ValueError("Broader review needs the complete development cohort")
    return parent, rows


def score_predictions(stored, predictions, row, config, failed_channels):
    """Evaluator-only boundary: no returned metric is used by correction."""
    original, target = stored["eeg"], stored["paired_reference"]
    names, regions, mask = stored["channel_names"], stored["regions"], stored["mask"]
    scopes = {"whole": np.flatnonzero(mask)}
    scopes.update({label: np.flatnonzero(mask & (regions == region))
                   for label, region in (("frontal", 0), ("posterior", 1), ("central_shared", 2))})
    records, gates = [], []
    for scope, ids in scopes.items():
        if not len(ids):
            continue
        controls = {arm: paired_channels(predictions[arm][ids], target[ids]) for arm in ("raw_shared", "raw_axis")}
        for arm, cleaned in predictions.items():
            metrics = paired_channels(cleaned[ids], target[ids], original[ids])
            engineering = paired(cleaned[ids], target[ids], config["fs"])
            losses = {key: float(np.mean(value["pearson_cc"] - metrics["pearson_cc"]))
                      for key, value in controls.items()}
            gates.append({"example_id": row["example_id"], "recipient": row["recipient"],
                          "donor": row["donor"], "condition": row["condition"], "arm": arm, "scope": scope,
                          "cc_deterioration": losses,
                          "cc_pass": all(np.isfinite(value) and value <= config["preservation"]["pearson_deterioration"]
                                         for value in losses.values()),
                          "clean_checks": engineering if row["condition"] == "clean" else None,
                          "clean_pass": preservation_pass(engineering, config["preservation"])
                                        if row["condition"] == "clean" else None})
            for local, index in enumerate(ids):
                records.append({**{key: row[key] for key in ("example_id", "recipient", "donor", "condition",
                                  "input_snr_db", "pilot_recipient_exposed", "pilot_donor_exposed")},
                                "fold": row["partition"]["fold"], "arm": arm, "scope": scope,
                                "channel": str(names[index]),
                                "failure": bool(arm.startswith("vmd") and int(index) in failed_channels),
                                **{key: float(value[local]) for key, value in metrics.items()},
                                "pooled_snr_db": engineering["snr_db"]})
    return records, gates


def save_tables(records, gates, output):
    table = pd.DataFrame(records)
    table.to_csv(output / "channel_metrics.csv", index=False)
    contaminated = table[table.condition != "clean"]
    keys = ["example_id", "recipient", "donor", "fold", "arm", "scope", "condition", "input_snr_db",
            "pilot_recipient_exposed", "pilot_donor_exposed"]
    columns = ["mse", "rrmse_time", "pearson_cc", "snr_energy_db", "snr_improvement_db", "pooled_snr_db"]
    grouped = contaminated.groupby(keys, dropna=False)[columns].mean().reset_index()
    grouped.to_csv(output / "record_metrics.csv", index=False)
    for filename, group in (("method_means", ["arm", "scope"]),
                            ("condition_means", ["arm", "scope", "condition", "input_snr_db"]),
                            ("fold_means", ["fold", "arm", "scope"]),
                            ("recipient_means", ["recipient", "arm", "scope"]),
                            ("donor_means", ["donor", "arm", "scope"]),
                            ("exposure_means", ["pilot_recipient_exposed", "pilot_donor_exposed", "arm", "scope"])):
        grouped.groupby(group)[columns].mean().to_csv(output / (filename + ".csv"))
    uncertainty_report(contaminated, output, comparators=("raw_shared", "raw_axis"),
                       candidates=("vmd_anchored",), repetitions=2000, seed=42)
    # Every record remains visible. This is deliberately more conservative than
    # judging only the mean, and does not by itself qualify a native teacher.
    atomic_json(output / "preservation_summary.json", {
        arm: {"all_record_cc_pass": all(row["cc_pass"] for row in gates if row["arm"] == arm),
              "all_clean_record_pass": all(row["clean_pass"] for row in gates
                                            if row["arm"] == arm and row["condition"] == "clean")}
        for arm in ARMS})
    return grouped


def plot_conditions(records, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    fig.subplots_adjust(top=.83, bottom=.09, hspace=.38, wspace=.3)
    for row_index, scope in enumerate(("frontal", "whole")):
        for axis, condition in zip(axes[row_index], ("blink", "lateral", "mixed")):
            shown = records[(records.scope == scope) & (records.condition == condition)]
            for arm in ARMS[1:]:
                points = shown[shown.arm == arm].groupby("input_snr_db")["pooled_snr_db"].mean()
                axis.plot(points.index, points.values, marker="o", label=arm.replace("_", " "))
            axis.set_title(scope.title() + ": " + condition)
            axis.set_xticks([-5, 0, 5])
            axis.set_xlabel("Nominal whole-input SNR (dB)")
            axis.set_ylabel("Mean pooled-window output SNR (dB)")
            axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=4, frameon=False)
    fig.suptitle("Frozen frontal recipes: 32 development recipients, five donors; not fresh confirmation", y=.99)
    fig.savefig(output / "broader_condition_snr.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(input_root, output, config, profile):
    if profile != "full":
        raise ValueError("Broader frontal validation requires the complete frozen cohort")
    parent, rows = readiness(input_root, config)
    started = time.perf_counter()
    for folder in ("predictions", "modes", "fits", "record_metrics", "record_decisions"):
        (output / folder).mkdir()
    records, gates, progress = [], [], []
    failed_count = channel_decompositions = 0
    for row in rows:
        verify_parent(parent, (row["array_path"], row["calibration"]))
        with np.load(parent / row["array_path"], allow_pickle=False) as archive:
            stored = dict(archive)
        with np.load(parent / row["calibration"], allow_pickle=False) as archive:
            calibration = dict(archive)
        # Explicitly exclude generator truth and condition labels from inference.
        eeg, refs = stored["eeg"], stored["references"]
        names, regions, mask = stored["channel_names"], stored["regions"], stored["mask"]
        bundles = decompose_frontal(eeg, names, regions, FixedRecipe(3), config, mask)
        vectors, metadata, failed = {}, {}, set()
        for index, (modes, residual, info) in bundles.items():
            reconstruction = validate_mode_reconstruction(modes, residual, eeg[index])
            vectors[f"channel_{index}_modes"], vectors[f"channel_{index}_residual"] = modes, residual
            metadata[str(index)] = {**info, "reconstruction": reconstruction,
                "mode_reference_cc": correlations(modes, refs, 20).tolist(),
                "residual_reference_cc": correlations(residual[None], refs, 20).tolist()}
            if info.get("failure") or info.get("hit_iteration_limit"):
                failed.add(index)
        channel_decompositions += len(bundles)
        failed_count += len(failed)
        np.savez_compressed(output / "modes" / (row["example_id"] + ".npz"), **vectors)
        atomic_json(output / "modes" / (row["example_id"] + ".json"), {
            "input_sha256": sha256_file(parent / row["array_path"]), "K": 3, "alpha": 2000, "channels": metadata})
        nonfrontal, nonfrontal_detail, fitted = regional_direct(eeg, refs, calibration, regions, mask)
        np.savez_compressed(output / "fits" / (row["example_id"] + ".npz"),
                            coefficients=fitted.coefficients, mean=fitted.reference_mean,
                            scale=fitted.reference_scale, active=fitted.active_columns, lags=fitted.lags)
        predictions, decisions = {"identity": eeg.copy()}, {"nonfrontal": nonfrontal_detail}
        for arm, mode, use_vmd in FRONTAL_ARMS:
            _, artifact, decisions[arm] = calibration_transfer(eeg, refs, calibration, names, regions, mask,
                bundles=bundles if use_vmd else None, gain_mode=mode, anchor_weight=config["anchored_vmd_weight"])
            predictions[arm] = compose_regions(eeg, artifact, nonfrontal, regions, mask)
        nonfrontal_ids = mask & np.isin(regions, (1, 2))
        for arm in ARMS[2:]:
            np.testing.assert_array_equal(predictions[arm][nonfrontal_ids], predictions["raw_shared"][nonfrontal_ids])
        # Paired truth first enters an API here, after predictions are complete.
        scored, checks = score_predictions(stored, predictions, row, config, failed)
        records.extend(scored)
        gates.extend(checks)
        np.savez_compressed(output / "predictions" / (row["example_id"] + ".npz"),
                            **predictions, channel_names=names, regions=regions, mask=mask)
        pd.DataFrame(scored).to_csv(output / "record_metrics" / (row["example_id"] + ".csv"), index=False)
        atomic_json(output / "record_decisions" / (row["example_id"] + ".json"), decisions)
        write_jsonl(output / "engineering_checks.jsonl", gates)
        progress.append({"example_id": row["example_id"], "failed_frontal_channels": sorted(failed),
                         "nonfrontal_parity": True})
        write_jsonl(output / "review_progress.jsonl", progress)
        print("BROADER_FRONTAL_SAVED", len(progress), len(rows), row["example_id"], flush=True)
    grouped = save_tables(records, gates, output)
    plot_conditions(grouped, output)
    atomic_json(output / "broader_frontal_summary.json", {"examples": len(rows), "arms": list(ARMS),
                "recipient_count": 32, "donor_count": 5, "source_cohort_sha256": config["cohort_sha256"],
                "channel_decompositions": channel_decompositions, "failed_channel_decompositions": failed_count,
                "nonfrontal_outputs_matched": True, "teacher_qualified": False,
                "reserved_confirmation_opened": False, "autovmd_restarted": False,
                "runtime_s": time.perf_counter() - started,
                "limitation": "development-exposed, regression-generated fields; native validation still required"})
