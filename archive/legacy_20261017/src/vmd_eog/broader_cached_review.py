"""Repair cached reporting and test a solver-failure-only frontal fallback.

No VMD, calibration fitting, parameter search or neural learning is performed.
The original five arms are re-scored and checked against their saved metrics.
"""
import json
import time

import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .broader_coupled_corpus import declared_parent
from .broader_frontal_review import (ARMS, plot_conditions, readiness as corpus_readiness,
                                    save_tables, score_predictions)
from .frontal_methods import frontal_indices
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl


FALLBACK = "vmd_anchored_raw_fallback"
ALL_ARMS = (*ARMS, FALLBACK)


def raw_failure_fallback(original, anchored, raw, failed, names, regions, mask):
    """Choose one original-input correction per row using solver flags only."""
    original, anchored, raw = (np.asarray(value, dtype=np.float64) for value in (original, anchored, raw))
    frontal = set(frontal_indices(original, names, regions, mask).tolist())
    failed = set(failed)
    if (anchored.shape != original.shape or raw.shape != original.shape
            or not np.isfinite(anchored).all() or not np.isfinite(raw).all()
            or not failed.issubset(frontal)):
        raise ValueError("Fallback requires finite matched predictions and failed verified frontal rows")
    result = anchored.copy()
    if failed:
        indices = sorted(failed)
        result[indices] = raw[indices]
    unchanged = np.asarray([index not in failed for index in range(len(original))])
    np.testing.assert_array_equal(result[unchanged], anchored[unchanged])
    return result


def readiness(input_root, config):
    if (config.get("scope") != "broader_cached_failure_review"
            or config.get("autovmd_enabled") is not False
            or config.get("anchored_vmd_weight") != .25
            or config.get("fixed_recipes") != [{"K": 3, "alpha": 2000}]
            or config.get("reserved_access") != "closed until verified freeze"):
        raise ValueError("Cached broader review policy differs")
    corpus, rows = corpus_readiness(input_root, {**config, "scope": "broader_fixed_anchored_frontal_validation"})
    cache = declared_parent(input_root, "broader_frontal_summary.json", config["cache_parent"])
    verify_parent(cache, ("channel_metrics.csv", "review_progress.jsonl", "stage_tests_summary.json"))
    summary = json.loads((cache / "broader_frontal_summary.json").read_text())
    if (summary.get("examples") != 320 or summary.get("arms") != list(ARMS)
            or summary.get("source_cohort_sha256") != config["cohort_sha256"]
            or summary.get("reserved_confirmation_opened") is not False
            or summary.get("nonfrontal_outputs_matched") is not True
            or json.loads((cache / "stage_tests_summary.json").read_text()).get("passed") is not True):
        raise ValueError("Cached comparison is not eligible")
    return corpus, cache, rows


def source_checks(gates, output, config):
    losses, clean = [], []
    for row in gates:
        metadata = {key: row[key] for key in ("example_id", "recipient", "donor", "condition", "arm", "scope")}
        if row["condition"] == "clean":
            clean.append({**metadata, **row["clean_checks"]})
        else:
            for comparator, loss in row["cc_deterioration"].items():
                losses.append({**metadata, "comparator": comparator, "cc_deterioration": loss})
    frame = pd.DataFrame(losses)
    frame.to_csv(output / "cc_record_checks.csv", index=False)
    summaries = []
    for source in ("recipient", "donor"):
        grouped = frame.groupby([source, "arm", "scope", "comparator"])["cc_deterioration"].mean().reset_index()
        grouped.to_csv(output / ("cc_" + source + "_means.csv"), index=False)
        for (arm, scope, comparator), subset in grouped.groupby(["arm", "scope", "comparator"]):
            worst = subset.loc[subset.cc_deterioration.idxmax()]
            summaries.append({"source_unit": source, "arm": arm, "scope": scope, "comparator": comparator,
                "worst_source": worst[source], "worst_source_mean_cc_deterioration": float(worst.cc_deterioration),
                "within_declared_cc_limit": bool(worst.cc_deterioration <= config["preservation"]["pearson_deterioration"])})
    pd.DataFrame(summaries).to_csv(output / "cc_source_limit_diagnostics.csv", index=False)
    pd.DataFrame(clean).to_csv(output / "clean_preservation_records.csv", index=False)
    # These source means supplement the unchanged per-record screens. Native
    # preservation and final teacher qualification are separate requirements.
    atomic_json(output / "source_check_definition.json", {
        "cc_mean": "channel mean per record, then equal contaminated condition/input strata within source",
        "source_units": ["recipient", "donor"], "per_record_checks_retained": True,
        "native_preservation_tested": False, "teacher_qualified": False})


def run(input_root, output, config, profile):
    if profile != "full":
        raise ValueError("Cached review requires all 320 records")
    corpus, cache, rows = readiness(input_root, config)
    started = time.perf_counter()
    (output / "fallback_predictions").mkdir()
    records, gates, progress = [], [], []
    changed_records = []
    for row in rows:
        example = row["example_id"]
        pred_path, mode_path = "predictions/" + example + ".npz", "modes/" + example + ".json"
        verify_parent(cache, (pred_path, mode_path))
        verify_parent(corpus, (row["array_path"],))
        with np.load(cache / pred_path, allow_pickle=False) as archive:
            saved = dict(archive)
        with np.load(corpus / row["array_path"], allow_pickle=False) as archive:
            stored = dict(archive)
        for key in ("channel_names", "regions", "mask"):
            np.testing.assert_array_equal(stored[key], saved[key])
        np.testing.assert_array_equal(stored["eeg"], saved["identity"])
        info = json.loads((cache / mode_path).read_text())
        if (info["input_sha256"] != sha256_file(corpus / row["array_path"])
                or info["K"] != 3 or info["alpha"] != 2000):
            raise ValueError("Cached mode/input identity differs")
        failed = {int(index) for index, detail in info["channels"].items()
                  if detail.get("failure") or detail.get("hit_iteration_limit")}
        predictions = {arm: saved[arm] for arm in ARMS}
        predictions[FALLBACK] = raw_failure_fallback(stored["eeg"], predictions["vmd_anchored"],
            predictions["raw_shared"], failed, stored["channel_names"], stored["regions"], stored["mask"])
        # Only evaluator code sees paired truth after the fallback is fixed.
        scored, checks = score_predictions(stored, predictions, row, config, failed)
        for item in scored:
            if item["arm"] == FALLBACK and item["failure"] is not None:
                item["failure"] = "mode_decomposition_failure; observed raw fallback applied"
        records.extend(scored)
        gates.extend(checks)
        np.savez_compressed(output / "fallback_predictions" / (example + ".npz"),
                            cleaned=predictions[FALLBACK], channel_names=stored["channel_names"],
                            regions=stored["regions"], mask=stored["mask"])
        if failed:
            changed_records.append({"example_id": example, "failed_frontal_channels": sorted(failed),
                                    "mode_metadata": mode_path, "selection_information": "solver flags only"})
        progress.append({"example_id": example, "inherited_prediction_sha256": sha256_file(cache / pred_path),
                         "failed_frontal_channels": sorted(failed), "new_vmd_decompositions": 0})
        write_jsonl(output / "cached_review_progress.jsonl", progress)
        print("CACHED_FRONTAL_SAVED", len(progress), len(rows), example, flush=True)
    table = pd.DataFrame(records)
    original = pd.read_csv(cache / "channel_metrics.csv", float_precision="round_trip")
    keys = ["example_id", "arm", "scope", "channel"]
    previous = original.set_index(keys).sort_index()
    repeated = table[table.arm.isin(ARMS)].set_index(keys).sort_index()
    if not previous.index.equals(repeated.index):
        raise ValueError("Reproduced original metric keys differ")
    metric_columns = ["mse", "rrmse_time", "pearson_cc", "snr_energy_db", "snr_improvement_db", "pooled_snr_db"]
    np.testing.assert_allclose(repeated[metric_columns], previous[metric_columns],
                               rtol=1e-12, atol=1e-30, equal_nan=True)
    grouped = save_tables(records, gates, output, arms=ALL_ARMS, candidates=("vmd_anchored", FALLBACK))
    old_intervals = pd.read_csv(cache / "coverage_uncertainty_intervals.csv", float_precision="round_trip")
    new_intervals = pd.read_csv(output / "coverage_uncertainty_intervals.csv", float_precision="round_trip")
    interval_keys = ["scope", "candidate", "comparator", "metric"]
    previous_intervals = old_intervals.set_index(interval_keys).sort_index()
    repeated_intervals = new_intervals[new_intervals.candidate == "vmd_anchored"].set_index(interval_keys).sort_index()
    if not previous_intervals.index.equals(repeated_intervals.index):
        raise ValueError("Original uncertainty comparison keys differ")
    interval_values = ["mean_delta", "ci95_lower", "ci95_upper"]
    np.testing.assert_allclose(repeated_intervals[interval_values], previous_intervals[interval_values],
                               rtol=1e-12, atol=1e-30, equal_nan=True)
    plot_conditions(grouped, output, arms=ALL_ARMS)
    source_checks(gates, output, config)
    write_jsonl(output / "engineering_checks.jsonl", gates)
    atomic_json(output / "fallback_decisions.json", changed_records)
    atomic_json(output / "cached_review_summary.json", {"examples": len(rows), "arms": list(ALL_ARMS),
        "new_vmd_decompositions": 0, "new_calibration_fits": 0, "original_paper_metrics_reproduced": True,
        "original_uncertainty_values_reproduced": True,
        "reporting_failure_contract_repaired": True, "fallback_changed_records": len(changed_records),
        "original_parent_preserved": True, "teacher_qualified": False, "reserved_confirmation_opened": False,
        "runtime_s": time.perf_counter() - started,
        "limitation": "development-exposed controlled mixtures; native and fresh confirmation remain untested"})
