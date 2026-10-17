"""Exploratory regional composition of immutable frontal and posterior evidence.

No VMD fitting/search or neural training. Posterior recipes are inherited from
source-excluded development selections; calibration fitting sees original EEG.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .experiments import corpus_parent
from .frontal_methods import frontal_indices
from .frontal_review import controlled_gates, strict_mean
from .io import atomic_json, read_jsonl, write_jsonl, sha256_file
from .metrics import paired
from .paper_evaluation import save_protocol
from .paper_metrics import paired_channels
from .posterior import fit_ica, fit_mwf
from .posterior_context import context_input, compose_correction, select_examples, PAPER_KEYS
from .reference import correlations, project


FRONTAL_ARMS = ("calibration_raw_gain", "K3_calibration_mode_gain",
                "calibration_raw_event_gain", "K3_calibration_event_gain")


def unique_artifact(input_root, name):
    paths = list(Path(input_root).rglob(name))
    if len(paths) != 1:
        raise ValueError("Exactly one versioned parent required: " + name)
    return paths[0]


def readiness(input_root, config):
    if config.get("schema_version") != 4 or config.get("autovmd_enabled") is not False:
        raise ValueError("Regional exploration requires schema4 and stopped AutoVMD")
    if config.get("scope") != "regional_composition_frontal_vmd_only":
        raise ValueError("Regional VMD scope differs")
    if config.get("reserved_access") != "closed until verified freeze":
        raise ValueError("Reserved access differs")
    ready = unique_artifact(input_root, "research_ready.json")
    verify_parent(ready.parent, (ready.name,))
    value = json.loads(ready.read_text())
    if value.get("passed") is not True or value.get("reserved_confirmation_opened") is not False:
        raise ValueError("Development readiness did not pass")
    if value.get("target_definition") != config["target_definition"]:
        raise ValueError("Readiness targets differ")


def load_frontal(parent, row, data):
    path = parent / "transfer_predictions" / (row["example_id"] + ".npz")
    verify_parent(parent, (str(path.relative_to(parent)),))
    saved = dict(np.load(path, allow_pickle=False))
    ids = frontal_indices(data["eeg"], data["channel_names"], data["regions"], data["mask"])
    if not np.array_equal(saved["names"], data["channel_names"][ids]):
        raise ValueError("Frontal channel identity differs")
    if not np.array_equal(saved["original"], data["eeg"][ids]) or not np.array_equal(saved["target"], data["paired_reference"][ids]):
        raise ValueError("Frontal saved input/target differs from corpus")
    artifacts = {}
    for arm in FRONTAL_ARMS:
        correction = np.zeros_like(data["eeg"], dtype=float)
        correction[ids] = saved["artifact_" + arm]
        if not np.allclose(data["eeg"][ids] - correction[ids], saved[arm], rtol=1e-12, atol=1e-18):
            raise ValueError("Saved frontal artifact/output parity failed")
        artifacts[arm] = correction
    return artifacts, ids


def posterior_artifact(data, calibration, row, selection, fits, config):
    """Reapply an inherited recipe; paired targets are not accepted."""
    _, _, ids, _ = context_input(data, "posterior")
    original = data["eeg"][ids]
    artifact = np.zeros_like(original, dtype=float)
    spec = selection.get("recipe")
    if spec is None or spec["method"] == "identity":
        return ids, artifact, None
    association = np.max(np.abs(correlations(original, data["references"], 20)), axis=1)
    method = spec["method"]
    error = None
    if method == "direct":
        artifact = project(original, data["references"], penalty=spec["penalty"])[0]
    else:
        scoring, out_rows, ids, layout = context_input(data, spec["support"])
        cal_values, cal_rows, cal_ids, cal_layout = context_input(calibration, spec["support"])
        if layout != cal_layout or not np.array_equal(ids, cal_ids) or not np.array_equal(out_rows, cal_rows):
            raise ValueError("Posterior calibration/scoring layout mismatch")
        key = (row["calibration"], method, spec["support"], tuple(spec["lags"]), spec["rank"])
        if key not in fits:
            started = time.perf_counter()
            try:
                if method == "ica":
                    expert = fit_ica(cal_values, calibration["references"], seed=config["seed"], boundaries=calibration["boundaries"])
                else:
                    expert = fit_mwf(cal_values, calibration["trial_types"], calibration["boundaries"], tuple(spec["lags"]), spec["rank"])
                fits[key] = (expert, None, time.perf_counter() - started)
            except Exception as failure:
                fits[key] = (None, f"{type(failure).__name__}: {failure}", time.perf_counter() - started)
        expert, error, _ = fits[key]
        if expert is not None:
            artifact = (expert.artifact(scoring, threshold=spec["threshold"]) if method == "ica" else expert.artifact(scoring))[out_rows]
    artifact *= (association >= spec["threshold"])[:, None]
    return ids, spec["strength"] * artifact, error


def review_oof(parent, output, config):
    """Add full CC reporting without changing the inherited partial-gate recipes."""
    frame = pd.read_csv(parent / "posterior_context_oof_scores.csv")
    keys = ["example_id"]
    direct = frame[frame.method == "direct"][keys + ["pearson_cc"]].rename(columns={"pearson_cc": "direct_cc"})
    frame = frame.merge(direct, on=keys, validate="many_to_one")
    frame["cc_deterioration"] = frame.direct_cc - frame.pearson_cc
    dirty = frame[frame.condition != "clean"]
    fields = list(PAPER_KEYS) + ["cc_deterioration"]
    balanced = dirty.groupby(["method", "support", "recipient", "condition", "input_snr_db"])[fields].agg(strict_mean)
    means = balanced.groupby(["method", "support", "recipient"]).agg(strict_mean).groupby(["method", "support"]).agg(strict_mean)
    means["correlation_gate"] = means.cc_deterioration <= config["preservation"]["pearson_deterioration"]
    means.to_csv(output / "posterior_oof_review.csv")
    return frame


def run(input_root, output, config, profile):
    started = time.perf_counter()
    readiness(input_root, config)
    corpus, corpus_rows = corpus_parent(input_root)
    ready = json.loads(unique_artifact(input_root, "research_ready.json").read_text())
    if ready["corpus_manifest_sha256"] != sha256_file(corpus / "corpus_manifest.jsonl") or ready["split_manifest_sha256"] != sha256_file(corpus / "split_manifest.json"):
        raise ValueError("Readiness source/split identities differ from attached corpus")
    rows = select_examples(corpus_rows, profile)
    posterior = unique_artifact(input_root, "posterior_context_summary.json").parent
    frontal = unique_artifact(input_root, "frontal_transfer_summary.json").parent
    verify_parent(posterior, ("posterior_context_summary.json", "posterior_context_fold_selections.jsonl", "posterior_context_oof_scores.csv"))
    verify_parent(frontal, ("frontal_transfer_summary.json",))
    verify_parent(frontal, ("frontal_transfer_failures.json",))
    source_failures = json.loads((frontal / "frontal_transfer_failures.json").read_text())
    atomic_json(output / "inherited_frontal_failures.json", source_failures)
    info = json.loads((frontal / "frontal_transfer_summary.json").read_text())
    if info.get("K") != 3 or info.get("alpha") != 2000 or info.get("monotonic_events") is not True or info.get("reserved_confirmation_opened") is not False:
        raise ValueError("Wrong fixed frontal parent")
    selections = read_jsonl(posterior / "posterior_context_fold_selections.jsonl")
    mapping = {}
    for item in selections:
        key = (item["fold"], item["method"], item["support"])
        if key in mapping:
            raise ValueError("Duplicate fold selection")
        mapping[key] = item
    write_jsonl(output / "inherited_posterior_selections.jsonl", selections)
    write_jsonl(output / "regional_selection_manifest.jsonl", rows)
    oof = review_oof(posterior, output, config)
    expected = {(r["example_id"], r["partition"]["fold"]) for r in select_examples(corpus_rows, "full")}
    if set(zip(oof.example_id, oof.fold)) != expected:
        raise ValueError("Posterior OOF source/window coverage differs")
    save_protocol(output)
    predictions = output / "regional_predictions"
    predictions.mkdir()
    fits, channel_rows, example_rows, parity_rows = {}, [], [], []
    for number, row in enumerate(rows):
        data = dict(np.load(corpus / row["array_path"], allow_pickle=False))
        cal = dict(np.load(corpus / row["calibration"], allow_pickle=False))
        front, front_ids = load_frontal(frontal, row, data)
        available = np.flatnonzero(data["mask"])
        fold = row["partition"]["fold"]
        choices = [v for (f, _, _), v in mapping.items() if f == fold]
        _, _, post_ids, _ = context_input(data, "posterior")
        if np.intersect1d(front_ids, post_ids).size:
            raise ValueError("Regional output masks overlap")
        meta = {k: row[k] for k in ("example_id", "recipient", "donor", "condition", "input_snr_db")}
        meta["fold"] = fold
        posterior_estimates = {}
        for selection in choices:
            label = selection["method"] + ":" + selection["support"]
            ids, artifact, failure = posterior_artifact(data, cal, row, selection, fits, config)
            posterior_estimates[label] = (artifact, failure)
            metrics = paired_channels(data["eeg"][ids] - artifact, data["paired_reference"][ids], data["eeg"][ids])
            old = oof[(oof.example_id == row["example_id"]) & (oof.method == selection["method"]) & (oof.support == selection["support"])]
            if len(old) != 1:
                raise ValueError("Inherited posterior OOF row missing")
            for metric in PAPER_KEYS:
                current = float(np.mean(metrics[metric]))
                previous = float(old.iloc[0][metric])
                passed = bool(np.isclose(current, previous, rtol=1e-5, atol=1e-10, equal_nan=True))
                parity_rows.append({**meta, "posterior_arm": label, "metric": metric, "current": current, "archived": previous, "passed": passed})
                if not passed:
                    pd.DataFrame(parity_rows).to_csv(output / "posterior_metric_parity.csv", index=False)
                    raise RuntimeError("Posterior metric parity failed; preserve diagnosis: " + label + "/" + metric)
        estimates = {"identity": data["eeg"]}
        failures = {"identity": None}
        for frontal_arm, frontal_artifact in front.items():
            front_failure = [item for item in source_failures if item["example_id"] == row["example_id"] and item["arm"] == frontal_arm]
            front_message = json.dumps(front_failure) if front_failure else None
            estimates[frontal_arm + "+pass"] = data["eeg"] - frontal_artifact
            failures[frontal_arm + "+pass"] = front_message
            for label, (artifact, failure) in posterior_estimates.items():
                name = frontal_arm + "+" + label
                estimates[name], _ = compose_correction(data["eeg"], post_ids, artifact, frontal_artifact)
                failures[name] = "; ".join(value for value in (front_message, failure) if value) or None
        comparator = estimates["calibration_raw_gain+direct:posterior"]
        scopes = {"whole": available, "frontal": front_ids, "posterior": post_ids}
        for arm, corrected in estimates.items():
            untouched = np.setdiff1d(np.arange(len(corrected)), np.concatenate([front_ids, post_ids]))
            if not np.array_equal(corrected[untouched], data["eeg"][untouched]):
                raise RuntimeError("Regional composition altered shared/unknown rows")
            for scope, ids in scopes.items():
                target, original = data["paired_reference"][ids], data["eeg"][ids]
                paper = paired_channels(corrected[ids], target, original)
                comp = paired_channels(comparator[ids], target, original)
                engineering = paired(corrected[ids], target, config["fs"])
                example_rows.append({**meta, "arm": arm, "scope": scope, "failure": failures[arm],
                    "pooled_energy_snr_db": engineering["snr_db"], "clean_change": engineering["relative_error"],
                    "alpha_db": engineering["alpha_db"], "beta_db": engineering["beta_db"], "covariance_error": engineering["covariance"]})
                for local, index in enumerate(ids):
                    channel_rows.append({**meta, "scope": scope, "channel": str(data["channel_names"][index]),
                        "evaluated_family": arm, "failure": failures[arm],
                        **{name: float(value[local]) for name, value in paper.items()},
                        "cc_deterioration": float(comp["pearson_cc"][local] - paper["pearson_cc"][local]),
                        "clean_change": float(paper["rrmse_time"][local]), "alpha_db": engineering["alpha_db"],
                        "beta_db": engineering["beta_db"], "covariance_error": engineering["covariance"]})
        np.savez_compressed(predictions / (row["example_id"] + ".npz"), original=data["eeg"], target=data["paired_reference"],
                            names=data["channel_names"], mask=data["mask"], regions=data["regions"], **estimates)
        atomic_json(output / "regional_progress.json", {"completed_examples": number + 1, "total_examples": len(rows), "last_example": row["example_id"]})
        print("REGIONAL_SAVED", number + 1, len(rows), flush=True)
    frame = pd.DataFrame(channel_rows)
    frame.to_csv(output / "regional_channel_scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame(example_rows).to_csv(output / "regional_example_scores.csv", index=False)
    pd.DataFrame(parity_rows).to_csv(output / "posterior_metric_parity.csv", index=False)
    write_jsonl(output / "regional_calibration_fits.jsonl", [{"calibration": key[0], "method": key[1],
        "support": key[2], "lags": list(key[3]), "rank": key[4], "failure": value[1], "runtime_s": value[2]}
        for key, value in fits.items()])
    fields = list(PAPER_KEYS) + ["cc_deterioration"]
    dirty = frame[frame.condition != "clean"]
    balanced = dirty.groupby(["scope", "evaluated_family", "recipient", "condition", "input_snr_db"])[fields].agg(strict_mean)
    balanced.groupby(["scope", "evaluated_family", "recipient"]).agg(strict_mean).groupby(["scope", "evaluated_family"]).agg(strict_mean).to_csv(output / "regional_method_means.csv")
    balanced.groupby(["scope", "evaluated_family", "condition", "input_snr_db"]).agg(strict_mean).to_csv(output / "regional_condition_means.csv")
    gates = {scope: controlled_gates(part, config) for scope, part in frame.groupby("scope")}
    atomic_json(output / "regional_controlled_gates.json", gates)
    atomic_json(output / "regional_summary.json", {"examples": len(rows), "recipients": int(frame.recipient.nunique()),
        "arms": len(estimates), "posterior_calibration_fits": len(fits), "fit_failures": sum(v[1] is not None for v in fits.values()),
        "runtime_s": time.perf_counter() - started, "posterior_parity_checks": len(parity_rows),
        "source_split_sha256": sha256_file(corpus / "split_manifest.json"), "autovmd_executed": False,
        "VMD_scope": "verified frontal K3/2000 caches only", "weights_trained": False,
        "qualification": False, "reserved_confirmation_opened": False, "native_preservation_unresolved": True,
        "selection": "inherited source-excluded posterior recipes with partial original gates; no retuning",
        "comparator": "calibration_raw_gain+direct:posterior", "scoring": "channel-first participant-balanced primary; pooled energy supplementary"})
    import matplotlib.pyplot as plt
    means = pd.read_csv(output / "regional_method_means.csv")
    for scope in scopes:
        table = means[means.scope == scope].sort_values("snr_energy_db")
        figure, axis = plt.subplots(figsize=(12, max(5, .23 * len(table))))
        axis.barh(table.evaluated_family, table.snr_energy_db)
        axis.axvline(15, linestyle="--", color="black", label="Project15dB target")
        axis.set_xlabel("Mean output SNR (dB); controlled development")
        axis.set_title(scope + ": inherited regional comparisons; native qualification pending")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / ("regional_" + scope + ".png"), dpi=130)
        plt.close(figure)
