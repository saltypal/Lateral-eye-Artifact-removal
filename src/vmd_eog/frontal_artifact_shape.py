"""Fixed DC-preserving artifact-shape experiment over archived frontal modes."""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .contracts import canonical_hash
from .coverage_uncertainty import uncertainty_report
from .experiments import corpus_parent
from .frontal_axis_native import CONTROLLED, checked_parent
from .frontal_methods import FixedRecipe, frontal_indices
from .frontal_replication import ArchivedModes
from .frontal_review import controlled_gates, strict_mean
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired
from .paper_metrics import paired_channels
from .posterior_context import PAPER_KEYS, select_examples
from .reference import correlations, lag_matrix


RAW = "calibration_raw_gain"
VMD = "K3_calibration_mode_gain"
SHAPE = "K3_dc_shape"
BASELINES = ("identity", RAW, VMD)
METHODS = (*BASELINES, SHAPE)


def dc_projection(artifact, modes, selected, valid):
    """Project variation, restore original artifact DC, and keep lag edges zero."""
    artifact, modes = np.asarray(artifact, float), np.asarray(modes, float)
    valid, selected = np.asarray(valid), np.asarray(selected)
    if (artifact.ndim != 1 or modes.ndim != 2 or modes.shape[-1] != len(artifact) or
            valid.shape != artifact.shape or valid.dtype != bool or valid.sum() < 2 or
            selected.ndim != 1 or (selected.size and not np.issubdtype(selected.dtype, np.integer)) or
            not np.isfinite(artifact).all() or not np.isfinite(modes).all()):
        raise ValueError("Finite mode/artifact vectors, integer selection and a Boolean valid mask required")
    selected = selected.astype(int)
    if len(set(selected.tolist())) != len(selected) or np.any((selected < 0) | (selected >= len(modes))):
        raise ValueError("Mode selection indices must be unique and in range")
    if np.any(artifact[~valid] != 0):
        raise ValueError("Archived artifact must be zero at invalid lag edges")
    mean = float(artifact[valid].mean())
    observation = artifact[valid] - mean
    vectors = modes[selected][:, valid]
    matrix = (vectors - vectors.mean(-1, keepdims=True)).T if len(selected) else np.empty((valid.sum(), 0))
    rcond = float(np.finfo(float).eps * max(matrix.shape))
    if len(selected):
        coefficients, _, rank, singular = np.linalg.lstsq(matrix, observation, rcond=rcond)
        projected = matrix @ coefficients
    else:
        coefficients, rank, singular, projected = np.empty(0), 0, np.empty(0), np.zeros_like(observation)
    # Preserve the original DC even if roundoff introduces a tiny projected mean.
    projected -= projected.mean()
    result = np.zeros_like(artifact)
    result[valid] = mean + projected
    diagnostics = dict(selected_modes=selected.tolist(), rank=int(rank), rcond=rcond,
        singular_values=singular.tolist(), coefficients=coefficients.tolist(),
        original_artifact_mean=mean, valid_samples=int(valid.sum()),
        original_centered_energy=float(observation @ observation),
        retained_centered_energy=float(projected @ projected),
        discarded_centered_energy=float(np.sum((observation - projected) ** 2)),
        empty_or_degenerate_span=bool(rank == 0), rank_rule="eps64 * max(valid_samples, selected_modes)")
    return result, diagnostics


def shape_correction(original, references, names, regions, mask, bundles, artifact, details):
    """No targets, calibration fitting, gain estimation or scoring labels enter."""
    original, artifact = np.asarray(original, float), np.asarray(artifact, float)
    ids = frontal_indices(original, names, regions, mask)
    if artifact.shape != original.shape or not np.isfinite(artifact).all():
        raise ValueError("Archived artifact must align with original EEG")
    if set(bundles) != set(ids.tolist()) or len(details) != len(ids):
        raise ValueError("Modes and decisions must match exactly the verified frontal rows")
    outside = np.ones(len(original), bool); outside[ids] = False
    if np.any(artifact[outside] != 0):
        raise ValueError("Nonfrontal or masked artifact must remain zero")
    _, valid = lag_matrix(references, (-10, 0, 10))
    result, diagnostics = np.zeros_like(original), []
    for local, index in enumerate(ids):
        detail = details[local]
        modes, residual, info = bundles[int(index)]
        if detail["channel"] != str(names[index]) or detail["lags"] != [-10, 0, 10] or detail["penalty"] != .01 or detail["gain_mode"] != "shared":
            raise ValueError("Archived correction decision settings differ")
        if bool(detail["triggered"]) != bool(detail["field_association"] >= .4):
            raise ValueError("Archived admission does not match the fixed .4 policy")
        failed = bool(info.get("failure") or info.get("hit_iteration_limit"))
        if failed != bool(detail.get("failure")):
            raise ValueError("Archived mode and correction failures disagree")
        if failed:
            if np.any(artifact[index]) or detail["selected_modes"]:
                raise ValueError("Failed mode rows must have zero fields and empty selection")
            diagnostics.append({**detail, "shape_failure": "inherited_scored_passthrough"})
            continue
        modes = np.asarray(modes, float)
        if modes.shape != (3, original.shape[-1]):
            raise ValueError("Shape correction requires the fixed K3 vectors")
        np.testing.assert_allclose(modes.sum(0) + residual, original[index], rtol=1e-5, atol=1e-11)
        selected = np.flatnonzero(np.max(np.abs(correlations(modes, references, 20)), axis=1) >= .2)
        if selected.tolist() != detail["selected_modes"]:
            raise ValueError("Archived fixed .2 mode association selection differs")
        if not detail["triggered"] and np.any(artifact[index]):
            raise ValueError("Unadmitted correction must be zero")
        result[index], projection = dc_projection(artifact[index], modes, selected, valid)
        diagnostics.append({**detail, **projection, "shape_failure": None, "gain_refitted": False})
    return original - result, result, diagnostics


def shape_parent(input_root):
    parent, summary = checked_parent(input_root, "frontal_transfer_summary.json", CONTROLLED)
    if (summary.get("examples") != 320 or summary.get("K") != 3 or summary.get("alpha") != 2000 or
            summary.get("weights_trained") is not False or summary.get("qualification") is not False):
        raise ValueError("Controlled shape precedent scope differs")
    return parent


def reconcile_metrics(frame, previous):
    """Legacy nonfinite numbers serialized as null; preserve new nonfinite values."""
    keys = ["example_id", "channel", "evaluated_family"]
    first = frame[frame.evaluated_family.isin(BASELINES)]
    second = previous[previous.evaluated_family.isin(BASELINES)]
    if len(first) != len(second) or first.duplicated(keys).any() or second.duplicated(keys).any():
        raise ValueError("Baseline metric coverage differs")
    joined = first.merge(second, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    if len(joined) != len(first):
        raise ValueError("Baseline example/channel metric keys differ")
    for field in ("recipient", "donor", "condition", "input_snr_db"):
        if not joined[field + "_new"].fillna("__MISSING__").equals(joined[field + "_old"].fillna("__MISSING__")):
            raise ValueError("Baseline source/condition metadata differs: " + field)
    fields = [*PAPER_KEYS, "alpha_db", "beta_db", "covariance_error", "clean_change"]
    for field in fields:
        new = joined[field + "_new"].to_numpy(float)
        old = joined[field + "_old"].to_numpy(float)
        normalized = np.where(np.isfinite(new), new, np.nan)
        np.testing.assert_allclose(normalized, old, rtol=1e-7, atol=1e-24, equal_nan=True)
    return len(joined) * len(fields)


def run(input_root, output, config, profile):
    if profile != "full" or config.get("require_archived_modes") is not True:
        raise ValueError("The fixed shape test requires the full cohort and archived modes only")
    parent = shape_parent(input_root)
    corpus, manifest = corpus_parent(input_root)
    rows = select_examples(manifest, "full")
    if len(rows) != 320:
        raise ValueError("Controlled shape cohort differs")
    cache = ArchivedModes(input_root, corpus, config)
    predictions, decisions_directory = output / "shape_predictions", output / "shape_decisions"
    predictions.mkdir(); decisions_directory.mkdir()
    write_jsonl(output / "shape_source_rows.jsonl", rows)
    atomic_json(output / "shape_protocol.json", dict(methods=list(METHODS), K=3, alpha=2000,
        cc_comparator=RAW, admission=.4, mode_association=.2, lags=[-10, 0, 10], penalty=.01,
        temporal_support=False, gain_refitted=False, residual_policy="subtract once from original; modes/residual need not be orthogonal",
        target_access="scoring only", source_scope="development-exposed fixed hypothesis; no new selection",
        primary_metrics=list(PAPER_KEYS), legacy_parity_nonfinite="parent nulls compared with normalized nonfinite values; new CSV preserves infinity/NaN"))
    scores, inherited, parity, started = [], [], 0, time.perf_counter()
    for number, row in enumerate(rows):
        with np.load(corpus / row["array_path"], allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        original = np.asarray(data["eeg"], float)
        ids = frontal_indices(original, data["channel_names"], data["regions"], data["mask"])
        bundles = cache.load(row, data, FixedRecipe(3))
        old_path = parent / "transfer_predictions" / (row["example_id"] + ".npz")
        verify_parent(parent, (str(old_path.relative_to(parent)),))
        entries = read_jsonl(parent / "transfer_decisions" / (row["example_id"] + ".jsonl"))
        old_details = {entry["arm"]: entry["decisions"] for entry in entries}
        if len(old_details) != len(entries):
            raise ValueError("Duplicate archived correction arms")
        with np.load(old_path, allow_pickle=False) as old:
            np.testing.assert_array_equal(old["names"], data["channel_names"][ids])
            np.testing.assert_array_equal(old["original"], original[ids])
            artifact = np.zeros_like(original)
            artifact[ids] = old["artifact_" + VMD]
            cleaned, shaped, details = shape_correction(original, data["references"], data["channel_names"],
                data["regions"], data["mask"], bundles, artifact, old_details[VMD])
            estimates = {"identity": original.copy(), SHAPE: cleaned}
            fields = {SHAPE: shaped}
            for method in (RAW, VMD):
                full = np.zeros_like(original); full[ids] = old["artifact_" + method]
                estimates[method] = original - full; fields[method] = full
                np.testing.assert_allclose(estimates[method][ids], old[method], rtol=1e-7, atol=1e-12)
                parity += 1
            np.testing.assert_array_equal(estimates["identity"][ids], old["original"])
            parity += 1
            # Targets are checked/scored only after the target-free correction.
            np.testing.assert_array_equal(data["paired_reference"][ids], old["target"])
        outside = np.ones(len(original), bool); outside[ids] = False
        for estimate in estimates.values():
            np.testing.assert_array_equal(estimate[outside], original[outside])
        raw_metrics = paired_channels(estimates[RAW][ids], data["paired_reference"][ids], original[ids])
        for method, estimate in estimates.items():
            paper = paired_channels(estimate[ids], data["paired_reference"][ids], original[ids])
            engineering = paired(estimate[ids], data["paired_reference"][ids], config["fs"])
            for local, index in enumerate(ids):
                source_details = old_details[VMD if method == SHAPE else method] if method != "identity" else []
                failure = source_details[local].get("failure") if source_details else None
                meta = dict(example_id=row["example_id"], recipient=row["recipient"], donor=row["donor"],
                    fold=row["partition"]["fold"], condition=row["condition"], input_snr_db=row["input_snr_db"],
                    channel=str(data["channel_names"][index]), evaluated_family=method, failure=failure,
                    clean_change=float(paper["rrmse_time"][local]), alpha_db=engineering["alpha_db"],
                    beta_db=engineering["beta_db"], covariance_error=engineering["covariance"],
                    cc_deterioration=float(raw_metrics["pearson_cc"][local] - paper["pearson_cc"][local]))
                scores.append({**meta, **{key: float(values[local]) for key, values in paper.items()}})
                if failure:
                    inherited.append(dict(example_id=row["example_id"], method=method, channel=meta["channel"], failure=failure))
        np.savez_compressed(predictions / (row["example_id"] + ".npz"), original=original,
            target=data["paired_reference"], names=data["channel_names"], regions=data["regions"], mask=data["mask"],
            **estimates, **{"artifact_" + key: value for key, value in fields.items()})
        write_jsonl(decisions_directory / (row["example_id"] + ".jsonl"), details)
        atomic_json(output / "shape_progress.json", dict(completed=number + 1, total=320))
        if (number + 1) % 20 == 0:
            print("FRONTAL_SHAPE_SAVED", number + 1, 320, flush=True)
    if parity != 960:
        raise ValueError("Controlled shape baseline prediction coverage differs")
    frame = pd.DataFrame(scores)
    metric_checks = reconcile_metrics(frame, pd.read_csv(parent / "frontal_transfer_scores.csv"))
    frame.to_csv(output / "shape_channel_scores.csv.gz", index=False, compression="gzip")
    write_jsonl(output / "shape_inherited_failures.jsonl", inherited)
    dirty = frame[frame.condition != "clean"]
    fields = [*PAPER_KEYS, "cc_deterioration"]
    strata = dirty.groupby(["evaluated_family", "recipient", "condition", "input_snr_db"])[fields].agg(strict_mean)
    recipients = strata.groupby(["evaluated_family", "recipient"]).agg(strict_mean)
    recipients.to_csv(output / "shape_recipient_means.csv")
    means = recipients.groupby("evaluated_family").agg(strict_mean)
    means.to_csv(output / "shape_method_means.csv")
    strata.groupby(["evaluated_family", "condition", "input_snr_db"]).agg(strict_mean).to_csv(output / "shape_condition_means.csv")
    reports = controlled_gates(frame, config)
    for report in reports:
        report["mean_cc_deterioration_vs_calibrated_raw"] = report.pop("mean_cc_deterioration_vs_direct")
        report["cc_comparator"] = RAW
    atomic_json(output / "shape_controlled_gates.json", reports)
    uncertainty = dirty.rename(columns={"evaluated_family": "arm"}).assign(scope="frontal")
    uncertainty_report(uncertainty, output, comparators=(RAW, VMD), candidates=(SHAPE,),
                       seed=42, repetitions=2000, scopes=("frontal",))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric, label in zip(axes, ("snr_energy_db", "pearson_cc"), ("Output SNR (dB)", "Ordinary Pearson CC")):
        means.loc[list(METHODS), metric].plot.bar(ax=axis)
        axis.set_ylabel(label); axis.set_xlabel(""); axis.tick_params(axis="x", rotation=25)
    fig.suptitle("Fixed frontal artifact shape: participant-balanced controlled development")
    fig.tight_layout(); fig.savefig(output / "shape_comparison.png", dpi=140); plt.close(fig)
    atomic_json(output / "shape_summary.json", dict(profile=profile, examples=320, methods=list(METHODS),
        baseline_prediction_parity_checks=parity, baseline_metric_parity_checks=metric_checks,
        recipients=int(frame.recipient.nunique()), donors=int(dirty.donor.nunique()), runtime_s=time.perf_counter()-started,
        source_split_sha256=sha256_file(corpus / "split_manifest.json"), parent_manifest_sha256=sha256_file(parent / "artifact_manifest.json"),
        scope="verified_frontal_only", K=3, alpha=2000, autovmd_executed=False, decompositions_computed=0,
        artifact_fields_fitted=False, gains_refitted=False, weights_trained=False, global_parameters_selected=False,
        reserved_confirmation_opened=False, qualification=False, native_validation_required=True,
        calibration_generated_contamination_bias=True, research_recipe_hash=canonical_hash({"K": 3, "alpha": 2000, "operator": "dc_preserving_selected_mode_shape"})))
