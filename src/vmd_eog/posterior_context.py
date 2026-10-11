"""Matched posterior teacher experiments; no VMD or neural-training dependency.

Every expert sees original EEG. Its output occupies posterior channels only.
Calibration EEG/EOG is unscored; paired targets are evaluation-only.
"""
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import canonical_hash
from .experiments import corpus_parent
from .io import atomic_json, write_jsonl, sha256_file
from .metrics import paired
from .paper_evaluation import save_protocol
from .paper_metrics import paired_channels
from .posterior import fit_ica, fit_mwf
from .reference import correlations, project, signed_context


SUPPORTS = ("posterior", "common_context", "signed_context", "raw_frontal", "full")
PAPER_KEYS = ("mse", "rrmse_time", "pearson_cc", "snr_energy_db", "snr_improvement_db")


def context_input(data, support):
    """Stable support layout is chosen by verified metadata, never window energy.

    Signed support deliberately contains redundant common/difference channels.
    ICA handles numerical rank; shrinkage stabilizes MWF. This is documented as
    an adaptation, and conditioning is saved rather than assumed harmless.
    """
    eeg = np.asarray(data["eeg"], dtype=float)
    regions = np.asarray(data["regions"])
    mask = np.asarray(data.get("mask", np.ones(len(eeg))), dtype=bool)
    if eeg.ndim != 2 or mask.shape != (len(eeg),) or regions.shape != mask.shape:
        raise ValueError("Posterior input channel/mask/region axes differ")
    if not np.isfinite(eeg).all():
        raise ValueError("Posterior input is nonfinite")
    posterior = np.flatnonzero((regions == 1) & mask)
    frontal = np.flatnonzero((regions == 0) & mask)
    names = np.asarray(data["channel_names"])
    if not len(posterior):
        raise ValueError("No verified available posterior channels")
    usable_regions = np.where(mask, regions, -1)
    context = signed_context(eeg, names, usable_regions, data["hemispheres"])
    flags = dict(zip(("left", "right", "midline"), context["available"]))
    flags.update(common=any(context["available"]),
                 lateral=bool(context["available"][0] and context["available"][1]))
    if support in ("common_context", "signed_context"):
        keys = ("common",) if support == "common_context" else (
            "left", "right", "midline", "common", "lateral")
        keys = [key for key in keys if flags[key]]
        extra = np.stack([context[key] for key in keys]) if keys else np.empty((0, eeg.shape[-1]))
        values = np.concatenate([eeg[posterior], extra])
        layout = names[posterior].tolist() + ["context:"+key for key in keys]
        out_rows = np.arange(len(posterior))
    else:
        if support == "posterior":
            ids = posterior
        elif support == "raw_frontal":
            ids = np.concatenate([posterior, frontal])
        elif support == "full":
            ids = np.flatnonzero(mask)
        else:
            raise ValueError("Unknown posterior support")
        values = eeg[ids]
        layout = names[ids].tolist()
        out_rows = np.array([np.flatnonzero(ids == index)[0] for index in posterior])
    return values, out_rows, posterior, {"layout": layout, "frontal_available": flags}


def compose_correction(original, posterior_ids, posterior_artifact, frontal_artifact=None):
    """One subtraction with disjoint output masks; all artifacts use raw EEG."""
    original = np.asarray(original, dtype=float)
    ids = np.asarray(posterior_ids, dtype=int)
    artifact = np.asarray(posterior_artifact, dtype=float)
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(original)):
        raise ValueError("Invalid posterior output indices")
    if artifact.shape != original[ids].shape or not np.isfinite(artifact).all():
        raise ValueError("Posterior artifact shape or finiteness failure")
    correction = np.zeros_like(original)
    if frontal_artifact is not None:
        front = np.asarray(frontal_artifact, dtype=float)
        if front.shape != original.shape or not np.isfinite(front).all() or np.any(front[ids] != 0):
            raise ValueError("Frontal and posterior correction masks overlap or shapes differ")
        correction += front
    correction[ids] = artifact
    return original - correction, correction


def select_examples(rows, profile):
    """First window per recipient/condition/input level, preserving source buckets."""
    controlled = [row for row in rows if row["target_kind"] == "controlled_recipient_reference"]
    recipient_folds, donor_folds = {}, {}
    for row in controlled:
        if row["partition"]["role"] != "development":
            raise ValueError("Reserved source entered posterior development")
        fold = row["partition"]["fold"]
        for key, registry in (("recipient", recipient_folds), ("donor", donor_folds)):
            identity = row[key]
            if identity in registry and registry[identity] != fold:
                raise ValueError("Recipient/donor identity crosses frozen source buckets")
            registry[identity] = fold
    if profile == "pilot":
        first_by_fold = {}
        for identity, fold in sorted(recipient_folds.items()):
            first_by_fold.setdefault(fold, identity)
        allowed = set(first_by_fold.values())
    else:
        allowed = set(recipient_folds)
    selected, seen = [], set()
    for row in sorted(controlled, key=lambda item: item["example_id"]):
        if row["recipient"] not in allowed:
            continue
        level = row.get("input_snr_db")
        if level not in (None, 0.) and profile == "pilot":
            continue
        if level not in (None, -5., 0., 5.):
            continue
        key = (row["recipient"], row["condition"], level)
        if key not in seen:
            selected.append(row)
            seen.add(key)
    if not selected or not any(row["condition"] == "clean" for row in selected):
        raise ValueError("Posterior selection lacks examples or clean controls")
    return selected


def candidate_specs(config, profile):
    grid = config["classical_grid"]
    ranks = [rank for rank in grid["mwf_ranks"] if profile != "pilot" or rank in (1, 2)]
    thresholds = [value for value in grid["thresholds"] if profile != "pilot" or value in (.4, .6)]
    strengths = [value for value in grid["strengths"] if profile != "pilot" or value in (.5, 1.)]
    specs = [{"method": "identity", "support": "posterior", "lags": [0],
              "rank": None, "threshold": 0., "strength": 0.}]
    for method, support in itertools.product(("mwf", "ica"), SUPPORTS):
        banks = grid["mwf_lags"] if method == "mwf" else [[0]]
        method_ranks = ranks if method == "mwf" else [None]
        for lags, rank, threshold, strength in itertools.product(banks, method_ranks, thresholds, strengths):
            specs.append(dict(method=method, support=support, lags=lags, rank=rank,
                              threshold=threshold, strength=strength))
    for threshold, strength in itertools.product(thresholds, strengths):
        specs.append(dict(method="direct", support="posterior", lags=[0], rank=None,
                          threshold=threshold, strength=strength, penalty=.01))
    return specs


def summarize(frame, specs, limits):
    """Equal recipient/condition/input weighting; engineering gates are separate."""
    records = []
    for key, group in frame.groupby("candidate_hash"):
        clean = group[group.condition == "clean"]
        dirty = group[group.condition != "clean"]
        balanced = dirty.groupby(["recipient", "condition", "input_snr_db"])[
            list(PAPER_KEYS)].mean().groupby("recipient").mean()
        clean_means = clean.groupby("recipient")[["clean_change", "clean_alpha_db",
                                                  "clean_beta_db", "covariance_error"]].mean()
        gates = {"clean_change": bool(len(clean_means) and clean_means.clean_change.max() <= limits["clean_change"]),
                 "alpha": bool(len(clean_means) and clean_means.clean_alpha_db.max() <= limits["alpha_db"]),
                 "beta": bool(len(clean_means) and clean_means.clean_beta_db.max() <= limits["beta_db"]),
                 "covariance": bool(len(clean_means) and clean_means.covariance_error.max() <= limits["covariance"])}
        record = {"candidate_hash": key, **specs[key], "recipients": len(balanced),
                  "failed_examples": int(group.fit_failure.notna().sum()),
                  "partial_preservation_gate": all(gates.values()),
                  "worst_recipient_mean_clean_change": float(clean_means.clean_change.max()),
                  "mean_apply_runtime_s": float(group.apply_runtime_s.mean()),
                  **{name: float(balanced[name].mean()) for name in PAPER_KEYS},
                  **{"gate_"+name: value for name, value in gates.items()}}
        records.append(record)
    return pd.DataFrame(records)


def grouped_report(frame, specs, output, config):
    """Held-out development bucket selection; both recipient and donor excluded.

    Own unscored calibration is allowed at inference. No calibration operator
    is globally fit across participants. These results do not open confirmation.
    """
    selected, evaluated = [], []
    for fold in sorted(frame.fold.unique()):
        training = frame[frame.fold != fold]
        heldout = frame[frame.fold == fold]
        for identity in ("recipient", "donor"):
            if set(training[identity]) & set(heldout[identity]):
                raise ValueError("Posterior grouped selection leaks "+identity)
        table = summarize(training, specs, config["preservation"])
        for (method, support), group in table.groupby(["method", "support"]):
            feasible = group[group.partial_preservation_gate].sort_values(
                ["snr_energy_db", "mean_apply_runtime_s"], ascending=[False, True])
            if len(feasible):
                key = feasible.iloc[0].candidate_hash
                selection = {"fold": int(fold), "method": method, "support": support,
                             "candidate_hash": key, "status": "selected_partial_gates",
                             "recipe": specs[key]}
                scored = heldout[heldout.candidate_hash == key].copy()
            else:
                key = next(key for key, spec in specs.items() if spec["method"] == "identity")
                selection = {"fold": int(fold), "method": method, "support": support,
                             "candidate_hash": None, "status": "no_feasible_recipe_passthrough"}
                scored = heldout[heldout.candidate_hash == key].copy()
                scored["method"], scored["support"] = method, support
            selected.append(selection)
            evaluated.append(scored)
    write_jsonl(output/"posterior_context_fold_selections.jsonl", selected)
    pd.concat(evaluated).to_csv(output/"posterior_context_oof_scores.csv", index=False)
    return selected


def run_comparison(input_root, output, config, profile, experiment):
    if experiment and experiment.get("purpose") != "matched_posterior_context":
        raise ValueError("Undeclared posterior experiment purpose")
    parent, corpus_rows = corpus_parent(input_root)
    rows = select_examples(corpus_rows, profile)
    specs = {canonical_hash(spec): spec for spec in candidate_specs(config, profile)}
    atomic_json(output/"posterior_context_candidates.json", specs)
    write_jsonl(output/"posterior_context_selection_manifest.jsonl", rows)
    save_protocol(output)
    shard_dir = output/"posterior_shards"
    shard_dir.mkdir(exist_ok=True)
    fits, fit_diagnostics, score_shards, excluded = {}, [], [], []
    started = time.perf_counter()
    for number, row in enumerate(rows):
        print("POSTERIOR_CONTEXT_BEGIN", number+1, len(rows), row["example_id"], flush=True)
        data = dict(np.load(parent/row["array_path"], allow_pickle=False))
        cal = dict(np.load(parent/row["calibration"], allow_pickle=False))
        try:
            _, _, posterior_ids, _ = context_input(data, "posterior")
        except ValueError as error:
            excluded.append({"example_id": row["example_id"], "reason": str(error)})
            continue
        original = data["eeg"][posterior_ids]
        target = data["paired_reference"][posterior_ids]
        # Original posterior-EOG window gate, with identical information access.
        association = np.max(np.abs(correlations(original, data["references"], 20)), axis=1)
        predictions, example_scores, channels, support_cache = {}, [], [], {}
        metadata = {key: row[key] for key in ("example_id", "recipient", "donor", "condition", "input_snr_db")}
        metadata["fold"] = row["partition"]["fold"]
        for key, spec in specs.items():
            apply_start = time.perf_counter()
            method, support = spec["method"], spec["support"]
            fit_error = None
            base_key = (method, support, tuple(spec["lags"]), spec["rank"], spec["threshold"])
            if base_key not in predictions:
                artifact = np.zeros_like(original, dtype=float)
                if method == "direct":
                    artifact = project(original, data["references"], penalty=spec["penalty"])[0]
                elif method != "identity":
                    if support not in support_cache:
                        scoring, out_rows, ids, support_info = context_input(data, support)
                        calibration, cal_out, cal_ids, cal_info = context_input(cal, support)
                        if not np.array_equal(ids, cal_ids) or not np.array_equal(out_rows, cal_out) or support_info["layout"] != cal_info["layout"]:
                            raise ValueError("Calibration and scoring support layouts differ")
                        support_cache[support] = (scoring, calibration, out_rows, cal_info)
                    scoring, calibration, out_rows, cal_info = support_cache[support]
                    fit_key = (row["calibration"], method, support, tuple(spec["lags"]), spec["rank"])
                    if fit_key not in fits:
                        fit_start = time.perf_counter()
                        try:
                            if method == "mwf":
                                expert = fit_mwf(calibration, cal["trial_types"], cal["boundaries"], tuple(spec["lags"]), spec["rank"])
                                details = {"eigenvalues": expert.eigenvalues.tolist(), "effective_output_rank": expert.rank}
                            else:
                                expert = fit_ica(calibration, cal["references"], seed=config["seed"], boundaries=cal["boundaries"])
                                details = {"effective_ica_rank": len(expert.unmixing), "component_associations": expert.associations.tolist()}
                            fits[fit_key] = (expert, None)
                        except Exception as error:
                            fits[fit_key] = (None, f"{type(error).__name__}: {error}")
                            details = {}
                        fit_diagnostics.append({"calibration": row["calibration"], "method": method, "support": support,
                            "lags": spec["lags"], "rank": spec["rank"], **cal_info, **details,
                            "runtime_s": time.perf_counter()-fit_start, "failure": fits[fit_key][1]})
                        write_jsonl(output/"posterior_context_fits.jsonl", fit_diagnostics)
                        print("POSTERIOR_CONTEXT_FIT", method, support, spec["rank"], spec["lags"], fits[fit_key][1], flush=True)
                    expert, fit_error = fits[fit_key]
                    if expert is not None:
                        artifact = (expert.artifact(scoring, threshold=spec["threshold"]) if method == "ica" else expert.artifact(scoring))[out_rows]
                artifact *= (association >= spec["threshold"])[:, None]
                predictions[base_key] = (artifact, fit_error)
            artifact, fit_error = predictions[base_key]
            cleaned, correction = compose_correction(data["eeg"], posterior_ids, spec["strength"]*artifact)
            cleaned = cleaned[posterior_ids]
            apply_runtime = time.perf_counter()-apply_start
            paper = paired_channels(cleaned, target, original)
            engineering = paired(cleaned, target, config["fs"])
            mean_metrics = {name: float(np.nanmean(paper[name])) for name in PAPER_KEYS}
            example_scores.append({**metadata, "candidate_hash": key, "method": method, "support": support,
                **mean_metrics, "fit_failure": fit_error, "posterior_channels": len(posterior_ids),
                "apply_runtime_s": apply_runtime,
                "clean_change": engineering["relative_error"], "clean_alpha_db": engineering["alpha_db"],
                "clean_beta_db": engineering["beta_db"], "covariance_error": engineering["covariance"],
                "correction_norm": float(np.linalg.norm(correction))})
            for channel, index in enumerate(posterior_ids):
                channels.append({**metadata, "candidate_hash": key, "method": method, "support": support,
                    "channel": str(data["channel_names"][index]), "original_channel_index": int(index),
                    **{name: float(value[channel]) for name, value in paper.items()}})
        scores_path = shard_dir/(row["example_id"]+"-scores.csv")
        pd.DataFrame(example_scores).to_csv(scores_path, index=False)
        pd.DataFrame(channels).to_csv(shard_dir/(row["example_id"]+"-channels.csv.gz"), index=False, compression="gzip")
        score_shards.append(scores_path)
        atomic_json(output/"posterior_context_progress.json", {"completed_examples": len(score_shards), "total_examples": len(rows),
            "last_example": row["example_id"], "scores_sha256": sha256_file(scores_path), "runtime_s": time.perf_counter()-started})
        print("POSTERIOR_CONTEXT_SAVED", number+1, len(rows), row["example_id"], flush=True)
    if not score_shards:
        raise ValueError("No eligible posterior examples")
    frame = pd.concat([pd.read_csv(path) for path in score_shards], ignore_index=True)
    frame.to_csv(output/"posterior_context_scores.csv", index=False)
    table = summarize(frame, specs, config["preservation"])
    table.to_csv(output/"posterior_context_search.csv", index=False)
    selected = grouped_report(frame, specs, output, config)
    atomic_json(output/"posterior_context_summary.json", {"profile": profile, "examples": len(score_shards),
        "candidates": len(specs), "recipients": int(frame.recipient.nunique()), "fits": len(fits),
        "fit_failures": sum(error is not None for _, error in fits.values()), "excluded": excluded,
        "partial_gate_fold_selections": len(selected), "teacher_qualified": False, "reserved_confirmation_opened": False,
        "snr_aggregation": "per channel, then condition/input level, then equal recipient",
        "scope": "bounded pilot software/context screen" if profile == "pilot" else "source-grouped controlled development comparison",
        "primary_metrics": list(PAPER_KEYS), "runtime_s": time.perf_counter()-started,
        "limitations": ["LEMON low-ocular reference is not proven artifact-free", "regression-derived ocular fields may favor reference methods",
            "own unscored recipient/donor calibration allowed", "native OSF and clean-correlation comparator qualification remain separate",
            "rest preservation gates do not alone qualify a teacher", "signed virtual channels are redundant; regularization/rank behavior must be inspected"]})
