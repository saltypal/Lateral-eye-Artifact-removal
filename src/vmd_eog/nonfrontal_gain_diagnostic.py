"""Bounded saved-field strength/blend experiment and evaluator-only error bounds."""
import json
import time

import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .frontal_review import controlled_gates, strict_mean
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired
from .nonfrontal_review import MIDDLE, TEMPORAL, inherited_failure, pooled_snr
from .paper_metrics import paired_channels
from .posterior_context import PAPER_KEYS
from .regional_integration import unique_artifact

BASE = "central_frontal_vmd_context"
RAW_CONTEXT = "central_frontal_raw_context"
REFERENCE = "reference_rank2:both"
WEIGHTS = (.25, .5, .75)


def compose_field(original, baseline, reference, regions, mask, weight, blend):
    """Keep frontal baseline and alter only the nonfrontal artifact estimate."""
    arrays = [np.asarray(value, dtype=float) for value in (original, baseline, reference)]
    original, baseline, reference = arrays
    regions, mask = np.asarray(regions), np.asarray(mask, dtype=bool)
    if (original.ndim != 2 or any(value.shape != original.shape or not np.isfinite(value).all() for value in arrays)
            or regions.shape != (len(original),) or mask.shape != regions.shape
            or not np.isfinite(weight) or weight < 0 or weight > 1 or not isinstance(blend, bool)):
        raise ValueError("Invalid fixed-field composition")
    unknown = ~mask | ~np.isin(regions, (0, 1, 2))
    frontal, nonfrontal = mask & (regions == 0), mask & np.isin(regions, (1, 2))
    if (not np.array_equal(baseline[unknown], original[unknown])
            or not np.array_equal(reference[unknown], original[unknown])
            or not np.array_equal(baseline[frontal], reference[frontal])):
        raise ValueError("Parent pass-through/frontal parity differs")
    artifact = np.zeros_like(original)
    artifact[frontal] = original[frontal] - baseline[frontal]
    field = original[nonfrontal] - reference[nonfrontal]
    if blend:
        old_field = original[nonfrontal] - baseline[nonfrontal]
        artifact[nonfrontal] = (1 - weight) * old_field + weight * field
    else:
        artifact[nonfrontal] = weight * field
    result = original - artifact
    # Preserve the frozen frontal array bit-for-bit after the subtraction parity check.
    if not np.allclose(result[frontal], baseline[frontal], rtol=1e-12, atol=1e-20):
        raise RuntimeError("Frozen frontal subtraction parity failed")
    result[frontal] = baseline[frontal]
    return result


def evaluator_bounds(original, target, baseline, reference):
    """Optimistic target-fitted diagnostics only; never used to generate an arm."""
    truth = original - target
    field = original - reference
    old = original - baseline
    field_energy = float(np.sum(field ** 2))
    scalar = float(np.clip(np.sum(truth * field) / field_energy, 0, 8)) if field_energy > 0 else 0.
    difference = field - old
    difference_energy = float(np.sum(difference ** 2))
    blend = float(np.clip(np.sum((truth - old) * difference) / difference_energy, 0, 1)) if difference_energy > 0 else 0.
    error = truth - field
    bias_energy = float(error.shape[-1] * np.sum(np.mean(error, axis=-1) ** 2))
    centered_energy = float(np.sum((error - np.mean(error, axis=-1, keepdims=True)) ** 2))
    total = float(np.sum(error ** 2))
    if not np.isclose(total, bias_energy + centered_energy, rtol=1e-10, atol=1e-25):
        raise RuntimeError("Error mean/variance accounting did not close")
    return {"diagnostic_target_fit": True, "deployable": False, "oracle_scalar": scalar,
            "oracle_scalar_pooled_snr_db": pooled_snr(original - scalar * field, target),
            "oracle_blend": blend, "oracle_blend_pooled_snr_db": pooled_snr(original - old - blend * difference, target),
            "current_pooled_snr_db": pooled_snr(reference, target), "residual_energy": total,
            "mean_bias_energy": bias_energy, "centered_error_energy": centered_energy,
            "field_energy": field_energy, "true_artifact_energy": float(np.sum(truth ** 2))}


def readiness(input_root, config):
    if (config.get("scope") != "fixed_nonfrontal_gain_and_blend_diagnosis"
            or config.get("autovmd_enabled") is not False
            or config.get("reserved_access") != "closed until verified freeze"
            or config.get("diagnostic_weights") != [.25, .5, .75]):
        raise ValueError("Fixed nonfrontal strength policy differs")
    parent = unique_artifact(input_root, "nonfrontal_summary.json").parent
    expected = config["diagnostic_parent"]
    if sha256_file(parent / "artifact_manifest.json") != expected["manifest_sha256"]:
        raise ValueError("Saved-field parent manifest differs")
    verify_parent(parent, ("run_spec.json", "nonfrontal_summary.json", "stage_tests_summary.json",
                           "nonfrontal_selection_manifest.jsonl", "nonfrontal_channel_scores.csv.gz"))
    spec = json.loads((parent / "run_spec.json").read_text())
    summary = json.loads((parent / "nonfrontal_summary.json").read_text())
    if (any(spec.get(key) != expected[key] for key in ("run_id", "git_sha", "campaign_id"))
            or summary.get("profile") != "full" or summary.get("examples") != 320
            or summary.get("source_split_sha256") != config["source_split_sha256"]
            or summary.get("reserved_confirmation_opened") is not False
            or json.loads((parent / "stage_tests_summary.json").read_text()).get("passed") is not True):
        raise ValueError("Saved-field parent source/split/completion differs")
    rows = read_jsonl(parent / "nonfrontal_selection_manifest.jsonl")
    if len(rows) != 320 or any(row["partition"]["role"] != "development" for row in rows):
        raise ValueError("Saved-field cohort is not the frozen development set")
    return parent, rows


def run(input_root, output, config, profile):
    started = time.perf_counter()
    parent, rows = readiness(input_root, config)
    if profile not in ("pilot", "full"):
        raise ValueError("Unknown saved-field profile")
    if profile == "pilot":
        selected = sorted({row["recipient"] for row in rows})[:5]
        rows = [row for row in rows if row["recipient"] in selected]
    old_scores = pd.read_csv(parent / "nonfrontal_channel_scores.csv.gz")
    frames, examples, diagnostics, decisions = [], [], [], []
    shards, predictions = output / "gain_shards", output / "gain_predictions"
    shards.mkdir(); predictions.mkdir()
    write_jsonl(output / "gain_selection_manifest.jsonl", rows)
    for number, row in enumerate(rows):
        relative = "nonfrontal_predictions/" + row["example_id"] + ".npz"
        decision_relative = "nonfrontal_predictions/" + row["example_id"] + ".json"
        verify_parent(parent, (relative, decision_relative))
        saved = dict(np.load(parent / relative, allow_pickle=False))
        decision = json.loads((parent / decision_relative).read_text())
        if decision.get("example_id") != row["example_id"]:
            raise ValueError("Saved decision identity differs")
        estimates = {arm: saved[arm] for arm in ("identity", BASE, RAW_CONTEXT, REFERENCE)}
        original, target = saved["original"], saved["target"]
        regions, mask, names = saved["regions"], saved["mask"], saved["names"]
        for weight in WEIGHTS:
            for blend in (False, True):
                arm = ("blend_old_rank2" if blend else "strength_rank2") + ":" + str(weight)
                estimates[arm] = compose_field(original, saved[BASE], saved[REFERENCE], regions, mask, weight, blend)
        scopes = {"whole": np.flatnonzero(mask), **{label: np.flatnonzero(mask & (regions == index))
                  for index, label in enumerate(("frontal", "posterior", "central_shared"))},
                  "middle_verified": np.flatnonzero(mask & (regions == 2) & np.isin(names, list(MIDDLE))),
                  "temporal_shared": np.flatnonzero(mask & (regions == 2) & np.isin(names, list(TEMPORAL)))}
        scopes = {name: ids for name, ids in scopes.items() if len(ids)}
        meta = {key: row[key] for key in ("example_id", "recipient", "donor", "condition", "input_snr_db")}
        current = []
        prior = inherited_failure(old_scores, row["example_id"], REFERENCE)
        for arm, cleaned in estimates.items():
            failure = inherited_failure(old_scores, row["example_id"], arm) if arm in saved else prior
            for scope, ids in scopes.items():
                paper = paired_channels(cleaned[ids], target[ids], original[ids])
                comparison = paired_channels(saved[REFERENCE][ids], target[ids], original[ids])
                engineering = paired(cleaned[ids], target[ids], config["fs"]) if row["condition"] == "clean" else None
                examples.append({**meta, "scope": scope, "evaluated_family": arm,
                    "pooled_energy_snr_db": pooled_snr(cleaned[ids], target[ids])})
                for local, index in enumerate(ids):
                    current.append({**meta, "scope": scope, "channel": str(names[index]),
                        "evaluated_family": arm, "failure": failure,
                        **{name: float(value[local]) for name, value in paper.items()},
                        "cc_deterioration": float(comparison["pearson_cc"][local] - paper["pearson_cc"][local]),
                        "clean_change": float(paper["rrmse_time"][local]),
                        "alpha_db": engineering["alpha_db"] if engineering else np.nan,
                        "beta_db": engineering["beta_db"] if engineering else np.nan,
                        "covariance_error": engineering["covariance"] if engineering else np.nan})
        # All inference arms are complete before paired targets enter optimistic bounds.
        if row["condition"] != "clean":
            for scope in ("posterior", "central_shared", "middle_verified", "temporal_shared"):
                if scope in scopes:
                    ids = scopes[scope]
                    diagnostics.append({**meta, "scope": scope,
                        **evaluator_bounds(original[ids], target[ids], saved[BASE][ids], saved[REFERENCE][ids])})
        decisions.append({**meta, "rank2_region_decisions": decision["details"].get("reference_rank2", {}),
                          "inherited_parent_flags": prior, "nonfrontal_method_failure": decision["failures"].get("reference_rank2"),
                          "target_used_by_fixed_arms": False})
        frames.extend(current)
        np.savez_compressed(predictions / (row["example_id"] + ".npz"),
            **{arm: value for arm, value in estimates.items() if arm not in saved})
        pd.DataFrame(current).to_csv(shards / (row["example_id"] + ".csv.gz"), index=False, compression="gzip")
        atomic_json(output / "gain_progress.json", {"completed_examples": number + 1, "total_examples": len(rows)})
        print("FIXED_GAIN_SAVED", number + 1, len(rows), flush=True)
    frame, example_frame = pd.DataFrame(frames), pd.DataFrame(examples)
    frame.to_csv(output / "gain_channel_scores.csv.gz", index=False, compression="gzip")
    example_frame.to_csv(output / "gain_example_snr.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "evaluator_only_gain_bounds.csv", index=False)
    write_jsonl(output / "gain_decisions.jsonl", decisions)
    from .nonfrontal_experiment import reconcile_baselines
    checks = reconcile_baselines(frame[frame.scope.isin(("whole", "frontal", "posterior", "central_shared"))], old_scores)
    dirty = frame[frame.condition != "clean"]
    grouped = dirty.groupby(["scope", "evaluated_family", "recipient", "condition", "input_snr_db"])[
        list(PAPER_KEYS) + ["cc_deterioration"]].agg(strict_mean)
    means = grouped.groupby(["scope", "evaluated_family", "recipient"]).agg(strict_mean).groupby(
        ["scope", "evaluated_family"]).agg(strict_mean).reset_index()
    pool = example_frame[example_frame.condition != "clean"].groupby(
        ["scope", "evaluated_family", "recipient", "condition", "input_snr_db"])["pooled_energy_snr_db"].agg(strict_mean)
    pool = pool.groupby(["scope", "evaluated_family", "recipient"]).agg(strict_mean).groupby(
        ["scope", "evaluated_family"]).agg(strict_mean).reset_index()
    means.merge(pool, on=["scope", "evaluated_family"], validate="one_to_one").to_csv(output / "gain_method_means.csv", index=False)
    grouped.groupby(["scope", "evaluated_family", "condition", "input_snr_db"]).agg(strict_mean).reset_index().to_csv(
        output / "gain_condition_means.csv", index=False)
    gates = {scope: controlled_gates(part, config) for scope, part in frame.groupby("scope")}
    for reports in gates.values():
        for report in reports:
            report["comparator"] = REFERENCE
            report["conservative_parent_flag_rows"] = report.pop("failure_rows_scored")
            report["mean_cc_deterioration_vs_rank2"] = report.pop("mean_cc_deterioration_vs_direct")
    atomic_json(output / "gain_gates.json", gates)
    from .coverage_uncertainty import uncertainty_report
    uncertainty_report(frame[frame.scope.isin(("whole", "posterior", "central_shared"))].rename(
        columns={"evaluated_family": "arm"}), output, comparators=(REFERENCE,),
        candidates=tuple(arm for arm in estimates if arm not in saved), seed=42, repetitions=500,
        scopes=("whole", "posterior", "central_shared"))
    import matplotlib.pyplot as plt
    table = means.merge(pool, on=["scope", "evaluated_family"], validate="one_to_one")
    selected = table[table.scope == "whole"]
    figure, axes = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(selected))
    axes.barh(positions - .18, selected.snr_energy_db, height=.36, label="Mean channel SNR")
    axes.barh(positions + .18, selected.pooled_energy_snr_db, height=.36, label="Pooled window energy SNR")
    axes.set_yticks(positions, selected.evaluated_family)
    axes.set_xlabel("Participant-balanced output SNR (dB)")
    axes.set_title("Fixed nonfrontal gain/blend: " + profile + " development cohort")
    axes.legend()
    figure.tight_layout()
    figure.savefig(output / "gain_snr_aggregation.png", dpi=160)
    plt.close(figure)
    atomic_json(output / "gain_summary.json", {"profile": profile, "examples": len(rows),
        "recipients": len({row["recipient"] for row in rows}), "arms": list(estimates),
        "baseline_paper_metric_checks": checks, "weights_fitted": False,
        "fixed_arm_target_access": False, "oracle_target_access": "evaluation-only; cannot select per-window output",
        "global_selection_performed": False, "reserved_confirmation_opened": False,
        "autovmd_executed": False, "qualification": False, "new_calibration_fits": 0,
        "failure_flags": "conservative historical parent flags, not new active-failure counts",
        "source_split_sha256": config["source_split_sha256"], "runtime_s": time.perf_counter() - started})

