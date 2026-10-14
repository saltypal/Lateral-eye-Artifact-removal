"""Frozen-output frontend comparison and SNR aggregation audit; no fitting."""
import json
import time

import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .frontal_review import controlled_gates, strict_mean
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired
from .paper_metrics import paired_channels
from .posterior_context import PAPER_KEYS
from .regional_integration import unique_artifact


FRONTAL_RAW = "raw_frontal_comparator"
FAMILIES = ("reference_rank2:both", "ica_component_ridge:both")
CONTROL = "raw_frontal+reference_rank2"
MIDDLE = {"Cz", "CPz"} | {prefix + str(index) for prefix in ("FC", "C", "CP") for index in range(1, 7)}
TEMPORAL = {"FT7", "FT8", "T7", "T8", "TP7", "TP8"}


def swap_frontend(original, candidate, raw_frontal, regions, mask):
    """Two disjoint artifact estimates, one subtraction from original EEG."""
    original, candidate, raw_frontal = map(lambda value: np.asarray(value, dtype=np.float64),
                                           (original, candidate, raw_frontal))
    regions, mask = np.asarray(regions), np.asarray(mask, dtype=bool)
    if (original.ndim != 2 or candidate.shape != original.shape or raw_frontal.shape != original.shape
            or regions.shape != (len(original),) or mask.shape != regions.shape
            or not all(np.isfinite(value).all() for value in (original, candidate, raw_frontal))):
        raise ValueError("Frontend comparison input/metadata contract differs")
    frontal = mask & (regions == 0)
    nonfrontal = mask & np.isin(regions, (1, 2))
    unknown = ~mask | ~np.isin(regions, (0, 1, 2))
    if not np.array_equal(candidate[unknown], original[unknown]):
        raise ValueError("Unknown/masked candidate is not pass-through")
    artifact = np.zeros_like(original)
    artifact[frontal] = original[frontal] - raw_frontal[frontal]
    artifact[nonfrontal] = original[nonfrontal] - candidate[nonfrontal]
    cleaned = original - artifact
    if (not np.allclose(cleaned[frontal], raw_frontal[frontal], rtol=1e-12, atol=1e-20)
            or not np.allclose(cleaned[nonfrontal], candidate[nonfrontal], rtol=1e-12, atol=1e-20)
            or not np.array_equal(cleaned[unknown], original[unknown])):
        raise RuntimeError("Single-subtraction frontend composition failed")
    return cleaned


def pooled_snr(cleaned, target):
    cleaned, target = np.asarray(cleaned, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if cleaned.shape != target.shape or not np.isfinite(cleaned).all() or not np.isfinite(target).all():
        raise ValueError("Invalid pooled SNR arrays")
    energy = float(np.sum(target ** 2))
    error = float(np.sum((cleaned - target) ** 2))
    if energy <= 0:
        return np.nan
    return float(10 * np.log10(energy / error)) if error > 0 else np.inf


def inherited_failure(scores, example_id, arm):
    """Preserve the exact saved arm's failure lineage without borrowing another fit."""
    selected = scores[(scores.example_id == example_id) & (scores.scope == "whole") &
                      (scores.evaluated_family == arm)]
    if selected.empty:
        raise ValueError("Missing parent failure lineage: " + arm)
    labels = sorted({value for value in selected.failure if isinstance(value, str) and value})
    return "; ".join(labels) or None


def readiness(input_root, config):
    if (config.get("scope") != "frozen_frontend_and_snr_review"
            or config.get("autovmd_enabled") is not False
            or config.get("reserved_access") != "closed until verified freeze"):
        raise ValueError("Frozen review policy differs")
    full = unique_artifact(input_root, "nonfrontal_summary.json").parent
    context = unique_artifact(input_root, "coverage_summary.json").parent
    for kind, parent, summary_name in (("full", full, "nonfrontal_summary.json"),
                                       ("context", context, "coverage_summary.json")):
        expected = config["review_parents"][kind]
        if sha256_file(parent / "artifact_manifest.json") != expected["manifest_sha256"]:
            raise ValueError("Review parent manifest differs: " + kind)
        verify_parent(parent, ("run_spec.json", summary_name, "stage_tests_summary.json"))
        spec = json.loads((parent / "run_spec.json").read_text())
        if any(spec.get(key) != expected[key] for key in ("run_id", "git_sha", "campaign_id")):
            raise ValueError("Review parent identity differs: " + kind)
        summary = json.loads((parent / summary_name).read_text())
        if (summary.get("profile") != "full" or summary.get("examples") != 320
                or summary.get("reserved_confirmation_opened") is not False
                or summary.get("source_split_sha256") != config["source_split_sha256"]
                or json.loads((parent / "stage_tests_summary.json").read_text()).get("passed") is not True):
            raise ValueError("Review parent completion/split differs: " + kind)
    verify_parent(full, ("nonfrontal_selection_manifest.jsonl", "nonfrontal_channel_scores.csv.gz"))
    verify_parent(context, ("coverage_selection_manifest.jsonl", "coverage_channel_scores.csv.gz"))
    rows = read_jsonl(full / "nonfrontal_selection_manifest.jsonl")
    if rows != read_jsonl(context / "coverage_selection_manifest.jsonl"):
        raise ValueError("Review parents have different cohorts")
    if any(row["partition"]["role"] != "development" for row in rows):
        raise ValueError("Reserved source entered frozen review")
    return full, context, rows


def run(input_root, output, config, profile):
    started = time.perf_counter()
    if profile != "full":
        raise ValueError("Frozen review audits the complete cohort only")
    full, context, rows = readiness(input_root, config)
    source_scores = pd.read_csv(full / "nonfrontal_channel_scores.csv.gz")
    context_scores = pd.read_csv(context / "coverage_channel_scores.csv.gz")
    scores, pooled, checks = [], [], 0
    predictions = output / "frontend_predictions"
    shards = output / "review_shards"
    predictions.mkdir()
    shards.mkdir()
    write_jsonl(output / "review_selection_manifest.jsonl", rows)
    for number, row in enumerate(rows):
        relative = "nonfrontal_predictions/" + row["example_id"] + ".npz"
        control_relative = "coverage_predictions/" + row["example_id"] + ".npz"
        verify_parent(full, (relative,))
        verify_parent(context, (control_relative,))
        saved = dict(np.load(full / relative, allow_pickle=False))
        old = dict(np.load(context / control_relative, allow_pickle=False))
        for key in ("original", "target", "names", "regions", "mask"):
            if not np.array_equal(saved[key], old[key]):
                raise ValueError("Frontend parents differ: " + key)
            checks += 1
        estimates = {name: saved[name] for name in ("identity", "central_frontal_vmd_context",
            "central_frontal_raw_context", *FAMILIES)}
        new = {}
        for family in FAMILIES:
            arm = "raw_frontal+" + family.split(":")[0]
            new[arm] = swap_frontend(saved["original"], saved[family], old[FRONTAL_RAW],
                                     saved["regions"], saved["mask"])
            estimates[arm] = new[arm]
        regions, mask, names = saved["regions"], saved["mask"], saved["names"]
        central = mask & (regions == 2)
        if not set(str(name) for name in names[central]).issubset(MIDDLE | TEMPORAL):
            raise ValueError("Unknown central subgroup anatomy")
        scopes = {"whole": np.flatnonzero(mask), **{label: np.flatnonzero(mask & (regions == index))
                  for index, label in enumerate(("frontal", "posterior", "central_shared"))},
                  "middle_verified": np.flatnonzero(central & np.isin(names, list(MIDDLE))),
                  "temporal_shared": np.flatnonzero(central & np.isin(names, list(TEMPORAL)))}
        scopes = {name: ids for name, ids in scopes.items() if len(ids)}
        meta = {key: row[key] for key in ("example_id", "recipient", "donor", "condition", "input_snr_db")}
        failures = {arm: inherited_failure(source_scores, row["example_id"], arm) for arm in estimates if arm not in new}
        raw_failure = inherited_failure(context_scores, row["example_id"], FRONTAL_RAW)
        for family in FAMILIES:
            arm = "raw_frontal+" + family.split(":")[0]
            failures[arm] = "; ".join(value for value in (failures[family], raw_failure) if value) or None
        current = []
        for arm, cleaned in estimates.items():
            for scope, ids in scopes.items():
                target, original = saved["target"][ids], saved["original"][ids]
                paper = paired_channels(cleaned[ids], target, original)
                comparator = paired_channels(estimates[CONTROL][ids], target, original)
                engineering = paired(cleaned[ids], target, config["fs"])
                pooled.append({**meta, "scope": scope, "evaluated_family": arm,
                    "pooled_energy_snr_db": pooled_snr(cleaned[ids], target),
                    "channel_mean_snr_db": float(np.mean(paper["snr_energy_db"]))})
                for local, index in enumerate(ids):
                    current.append({**meta, "scope": scope, "channel": str(names[index]),
                        "evaluated_family": arm, "failure": failures[arm],
                        **{name: float(value[local]) for name, value in paper.items()},
                        "cc_deterioration": float(comparator["pearson_cc"][local] - paper["pearson_cc"][local]),
                        "clean_change": float(paper["rrmse_time"][local]), "alpha_db": engineering["alpha_db"],
                        "beta_db": engineering["beta_db"], "covariance_error": engineering["covariance"]})
        scores.extend(current)
        np.savez_compressed(predictions / (row["example_id"] + ".npz"), **new)
        pd.DataFrame(current).to_csv(shards / (row["example_id"] + ".csv.gz"), index=False, compression="gzip")
        atomic_json(output / "review_progress.json", {"completed_examples": number + 1, "total_examples": len(rows)})
        print("FRONTEND_REVIEW_SAVED", number + 1, len(rows), flush=True)
    frame, pool_frame = pd.DataFrame(scores), pd.DataFrame(pooled)
    frame.to_csv(output / "review_channel_scores.csv.gz", index=False, compression="gzip")
    pool_frame.to_csv(output / "review_example_snr.csv", index=False)
    from .nonfrontal_experiment import reconcile_baselines
    baseline_checks = reconcile_baselines(frame[frame.scope.isin(("whole", "frontal", "posterior", "central_shared"))], source_scores)
    dirty = frame[frame.condition != "clean"]
    fields = list(PAPER_KEYS) + ["cc_deterioration"]
    source = dirty.groupby(["scope", "evaluated_family", "recipient", "condition", "input_snr_db"])[fields].agg(strict_mean)
    means = source.groupby(["scope", "evaluated_family", "recipient"]).agg(strict_mean).groupby(
        ["scope", "evaluated_family"]).agg(strict_mean).reset_index()
    pools = pool_frame[pool_frame.condition != "clean"].groupby(
        ["scope", "evaluated_family", "recipient", "condition", "input_snr_db"])["pooled_energy_snr_db"].agg(strict_mean)
    pools = pools.groupby(["scope", "evaluated_family", "recipient"]).agg(strict_mean).groupby(
        ["scope", "evaluated_family"]).agg(strict_mean).reset_index()
    means.merge(pools, on=["scope", "evaluated_family"], validate="one_to_one").to_csv(output / "review_method_means.csv", index=False)
    source.groupby(["scope", "evaluated_family", "condition", "input_snr_db"]).agg(strict_mean).reset_index().to_csv(
        output / "review_condition_means.csv", index=False)
    gates = {scope: controlled_gates(group, config) for scope, group in frame.groupby("scope")}
    for reports in gates.values():
        for report in reports:
            report["comparator"] = CONTROL
            report["mean_cc_deterioration_vs_comparator"] = report.pop("mean_cc_deterioration_vs_direct")
    atomic_json(output / "review_gates.json", gates)
    from .coverage_uncertainty import uncertainty_report
    uncertainty_report(frame[frame.scope.isin(("whole", "frontal", "posterior", "central_shared"))].rename(
        columns={"evaluated_family": "arm"}), output, comparators=(CONTROL,),
        candidates=FAMILIES, seed=42, repetitions=500)
    matched_ica = output / "matched_ica_frontend_uncertainty"
    matched_ica.mkdir()
    uncertainty_report(frame[frame.scope.isin(("whole", "frontal", "posterior", "central_shared"))].rename(
        columns={"evaluated_family": "arm"}), matched_ica,
        comparators=("raw_frontal+ica_component_ridge",), candidates=(FAMILIES[1],), seed=42, repetitions=500)
    import matplotlib.pyplot as plt
    table = means.merge(pools, on=["scope", "evaluated_family"], validate="one_to_one")
    selected = table[table.scope == "whole"]
    figure, axes = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(selected))
    axes.barh(positions - .18, selected.snr_energy_db, height=.36, label="Mean channel SNR")
    axes.barh(positions + .18, selected.pooled_energy_snr_db, height=.36, label="Pooled window energy SNR")
    axes.set_yticks(positions, selected.evaluated_family)
    axes.set_xlabel("Participant-balanced output SNR (dB)")
    axes.set_title("Frozen frontend comparison: development cohort")
    axes.legend()
    figure.tight_layout()
    figure.savefig(output / "frontend_snr_aggregation.png", dpi=160)
    plt.close(figure)
    atomic_json(output / "frontend_review_summary.json", {"examples": len(rows), "recipients": 32, "donors": 5,
        "parent_array_checks": checks, "baseline_paper_metric_checks": baseline_checks, "arms": list(estimates),
        "new_fits": 0, "weights_trained": False, "autovmd_executed": False,
        "uncertainty_contrasts": "rank2 frontend matched in root; ICA frontend matched in subfolder; ICA versus common rank2 control is whole-pipeline contrast",
        "reserved_confirmation_opened": False, "qualification": False,
        "native_validation_required": True, "runtime_s": time.perf_counter() - started,
        "channel_first_snr": "per-channel dB, then balanced recipient/condition/input means",
        "pooled_energy_snr": "sum over available scope channels/time within window, then balanced recipient/condition/input means",
        "subgroups": {"middle_verified": sorted(MIDDLE), "temporal_shared": sorted(TEMPORAL)}})
