"""Original-trial support ablation over frozen frontal artifact fields.

The gate is reconstructed from unscored calibration. This module never fits an
artifact field, decomposes EEG or selects a recipe on scored resting trials.
"""
import json
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .experiments import corpus_parent
from .frontal_axis_diagnostic import diagnostic_parent, original_overlap, trial_identity, window_artifacts
from .frontal_bounded_feasibility import BASELINES, preservation_reviews, reporting_condition
from .frontal_event_support import fit_axis_amplitude_support
from .frontal_methods import frontal_indices
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired
from .native_protocol import ConditionMoments, multi_overlap_add, write_native_reports
from .paper_metrics import resting_spectrum


TRIAL_METHODS = {
    ("raw", "or"): "raw_trial_or",
    ("raw", "separate"): "raw_trial_separate",
    ("vmd", "or"): "vmd_trial_or",
    ("vmd", "separate"): "vmd_trial_separate",
}
METHODS = ("identity", *BASELINES.values(), *TRIAL_METHODS.values())


def support_slice(track, window_number, window=1024, hop=512, padding="reflect"):
    """Align a precomputed original-trial track to an existing EEG window."""
    values = np.asarray(track, float)
    if (values.ndim not in (1, 2) or values.shape[-1] < 2 or
            not np.isfinite(values).all() or np.any((values < -1e-12) | (values > 1 + 1e-12)) or
            not 0 < hop <= window or window_number < 1 or padding not in ("reflect", "constant")):
        raise ValueError("Invalid support track or cached window identity")
    pad = window // 2
    widths = [(0, 0)] * (values.ndim - 1) + [(pad, pad + window)]
    padded = np.pad(values, widths, mode=padding)
    start = (window_number - 1) * hop
    result = padded[..., start:start + window]
    if result.shape[-1] != window:
        raise ValueError("Support window extends beyond the original padding contract")
    return result


def supported_fields(saved, axes, union):
    """Use cached signed contributions; preserve their gain and .4 admission."""
    baseline = window_artifacts(saved)
    original = np.asarray(saved["original"])
    axes, union = np.asarray(axes, float), np.asarray(union, float)
    if (axes.shape != (2, original.shape[-1]) or union.shape != (original.shape[-1],) or
            not np.isfinite(axes).all() or not np.isfinite(union).all() or
            np.any((axes < -1e-12) | (axes > 1 + 1e-12)) or
            np.any((union < -1e-12) | (union > 1 + 1e-12))):
        raise ValueError("Aligned trial supports must be finite and bounded")
    result = dict(baseline)
    for (representation, kind), method in TRIAL_METHODS.items():
        fields = np.asarray(saved[representation + "_axis_fields"], float)
        result[method] = (fields.sum(0) * union[None] if kind == "or" else
                          (fields * axes[:, None]).sum(0))
    return result


def checked_calibration(directory, record):
    """Only exact development-ledger calibration can reconstruct this gate."""
    saved_record = json.loads((directory / "record.json").read_text())
    if saved_record != record:
        raise ValueError("Calibration and native source ledgers differ")
    if record["partition"]["role"] != "development":
        raise ValueError("Reserved calibration source is forbidden")
    if set(record["calibration_trial_ids"]) & set(record["scoring_trial_ids"]):
        raise ValueError("Calibration/scoring trial overlap")
    with np.load(directory / "calibration.npz", allow_pickle=False) as saved:
        calibration = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(calibration["channel_names"], record["channel_names"])
    references, labels = calibration["references"], calibration["trial_types"]
    boundaries = np.asarray(calibration["boundaries"], int)
    if (references.ndim != 2 or references.shape[0] != 2 or labels.shape != (references.shape[-1],) or
            not np.isfinite(references).all() or np.any(boundaries <= 0) or
            np.any(boundaries >= len(labels)) or np.any(np.diff(boundaries) <= 0)):
        raise ValueError("Invalid calibration references, labels or trial boundaries")
    edges = [0, *boundaries.tolist(), len(labels)]
    if len(edges) - 1 != len(record["calibration_trial_ids"]):
        raise ValueError("Calibration boundary count differs from source trials")
    for trial, left, right in zip(record["calibration_trial_ids"], edges[:-1], edges[1:]):
        if not np.all(labels[left:right] == record["trial_labels"][trial]):
            raise ValueError("Calibration sample labels differ from source ledger")
    gate_inputs = dict(references=references, trial_types=labels, boundaries=boundaries)
    return calibration, fit_axis_amplitude_support(gate_inputs)


def check_local_support(saved, gate, entries):
    """A reconstructed gate must reproduce the predecessor before ablation."""
    axes, _ = gate.support_axes(saved["references"])
    union, _ = gate.support(saved["references"])
    np.testing.assert_array_equal(axes, saved["axis_supports"])
    np.testing.assert_array_equal(union, saved["union_support"])
    for method in BASELINES.values():
        details = entries[method]
        if not details:
            raise ValueError("Missing cached support decisions")
        for detail in details:
            np.testing.assert_array_equal(gate.thresholds, detail["axis_thresholds"])
            if detail["support_window"] != gate.window or detail["support_hop"] != gate.hop:
                raise ValueError("Cached detector timing differs")


def plot_gates(reviews, output):
    rows = [row for row in reviews if row["method"] != "identity"]
    labels = [row["method"].replace("calibration_", "").replace("_event_gain", " window OR")
              .replace("_axis window OR", " window separate").replace("_", " ") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, field, limit in zip(axes, ("relative_error", "covariance"), (.01, .02)):
        axis.barh(labels, [row["worst_record"][field] for row in rows])
        axis.axvline(limit, color="black", linestyle="--", label="Unchanged gate")
        axis.set_xlabel("Worst-record mean resting " + ("relative modification" if field == "relative_error" else "covariance error"))
        axis.legend()
    fig.suptitle("Fixed frontal fields: window versus original-trial support (two exposed records)")
    fig.tight_layout()
    fig.savefig(output / "trial_support_preservation.png", dpi=140)
    plt.close(fig)


def run(input_root, output, config, profile):
    if profile != "pilot":
        raise ValueError("Trial-support ablation is restricted to the exposed two-record pilot")
    parent = diagnostic_parent(input_root)
    corpus, _ = corpus_parent(input_root)
    if sha256_file(corpus / "artifact_manifest.json") != config["imported_parents"]["corpus-002"]:
        raise ValueError("Trial-support calibration corpus manifest differs")
    started = time.perf_counter()
    sources = read_jsonl(parent / "native_frontal_sources.jsonl")
    source_map = {record["record_id"]: record for record in sources}
    if len(sources) != 2 or set(source_map) != {"study01_p02_prep", "study01_p03_prep"}:
        raise ValueError("Native source cohort differs")
    gates, calibrations, cached_decisions = {}, {}, {}
    for record in sources:
        directory = corpus / "calibration" / record["record_id"]
        verify_parent(corpus, [str((directory / name).relative_to(corpus)) for name in ("record.json", "calibration.npz")])
        calibration, gate = checked_calibration(directory, record)
        calibrations[record["record_id"]], gates[record["record_id"]] = calibration, gate
    for path in sorted((parent / "native_frontal_decisions").glob("*.jsonl")):
        entries = read_jsonl(path)
        keys = [(row["window"], row["method"]) for row in entries]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate cached window decisions")
        cached_decisions[path.stem] = {key: row["decisions"] for key, row in zip(keys, entries)}
    support_checks = 0
    # This entire numerical preflight completes before any new prediction.
    for path in sorted((parent / "native_frontal_axis_windows").glob("*.npz")):
        stem, number = path.stem.rsplit("-window-", 1)
        record, _ = trial_identity(stem)
        with np.load(path, allow_pickle=False) as saved:
            entries = {method: cached_decisions[stem][(int(number), method)] for method in BASELINES.values()}
            check_local_support(saved, gates[record], entries)
        support_checks += 1
    if support_checks != 306:
        raise ValueError("Cached detector parity coverage differs")
    atomic_json(output / "trial_support_calibration.json", dict(
        cached_windows_verified=support_checks, corpus_manifest_sha256=sha256_file(corpus / "artifact_manifest.json"),
        records={record: dict(center=gate.center.tolist(), thresholds=gate.thresholds.tolist(),
            calibration_counts=gate.calibration_counts, scoring_data_fitted=False,
            calibration_sha256=sha256_file(corpus / "calibration" / record / "calibration.npz")) for record, gate in gates.items()}))
    rest_frame = pd.read_csv(parent / "native_frontal_preservation_trials.csv")
    rest_keys = set(zip(rest_frame.record_id, rest_frame.trial.astype(int)))
    inherited_failures = read_jsonl(parent / "native_frontal_failures.jsonl")
    write_jsonl(output / "inherited_native_failures.jsonl", inherited_failures)
    atomic_json(output / "trial_support_evaluation_protocol.json", dict(
        primary="complete-condition ordinary Pearson: HEOG lateral, VEOG blink/vertical; resting RMSE and relative spectra",
        units="source array units; microvolt scaling unverified", native_reconstruction_snr=False,
        spectral_settings=dict(fs=config["fs"], segment_seconds=2, overlap_seconds=1,
            window="Hann", detrend="constant", normalization_band_hz=[1, 40],
            unspecified_choices="declared paper adaptations"),
        engineering_gates=config["preservation"], exposed_participants=2,
        inference_labels=False, population_superiority_claim=False))
    predictions, supports_directory, decisions_directory = [output / name for name in
        ("trial_support_predictions", "trial_support_tracks", "trial_support_decisions")]
    for directory in (predictions, supports_directory, decisions_directory):
        directory.mkdir()
    moments, record_channels, seen_trials = {}, {}, set()
    preservation, spectra, disagreements = [], [], []
    trials, parity, padding_checks = 0, 0, 0
    for path in sorted((parent / "native_frontal_predictions").glob("*.npz")):
        record, trial = trial_identity(path.stem)
        gate, calibration = gates[record], calibrations[record]
        decisions = cached_decisions[path.stem]
        with np.load(path, allow_pickle=False) as archived:
            original, refs = archived["original"], archived["inference_references"]
            evaluation, names, regions = archived["evaluation_references"], archived["channel_names"], archived["regions"]
            np.testing.assert_array_equal(names, calibration["channel_names"])
            np.testing.assert_array_equal(regions, calibration["regions"])
            ids = frontal_indices(original, names, regions, calibration.get("mask", np.ones(len(names), bool)))
            with np.load(parent / "native_frontal_mode_examples" / path.name, allow_pickle=False) as example:
                np.testing.assert_array_equal(names[ids], example["channel_names"])
            axes, _ = gate.support_axes(refs)
            union, _ = gate.support(refs)
            np.savez_compressed(supports_directory / path.name, inference_references=refs,
                                original_axis_supports=axes, original_union_support=union)
            window_number, new_decisions = 0, []

            def estimate(values, references):
                nonlocal window_number
                window_number += 1
                cached = parent / "native_frontal_axis_windows" / f"{path.stem}-window-{window_number:04d}.npz"
                with np.load(cached, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(values, saved["original"])
                    np.testing.assert_array_equal(references, saved["references"])
                    aligned_axes = support_slice(axes, window_number, config["window"], config["hop"])
                    aligned_union = support_slice(union, window_number, config["window"], config["hop"])
                    result = supported_fields(saved, aligned_axes, aligned_union)
                    zero = supported_fields(saved,
                        support_slice(axes, window_number, config["window"], config["hop"], "constant"),
                        support_slice(union, window_number, config["window"], config["hop"], "constant"))
                    result.update({method + "_zero_padding": zero[method] for method in TRIAL_METHODS.values()})
                    start, overlap = original_overlap(window_number, config["window"], config["hop"], original.shape[-1])
                    left = max(0, config["window"] // 2 - start)
                    right = min(config["window"], config["window"] // 2 + original.shape[-1] - start)
                    differences = dict(record_id=record, trial=trial, window=window_number,
                        original_overlap_samples=overlap,
                        union_mean_abs_change=float(np.mean(np.abs(aligned_union[left:right] - saved["union_support"][left:right]))) if overlap else np.nan)
                    for axis, label in enumerate(("heog", "veog")):
                        differences[label + "_mean_abs_change"] = float(np.mean(np.abs(aligned_axes[axis, left:right] - saved["axis_supports"][axis, left:right]))) if overlap else np.nan
                    disagreements.append(differences)
                    new_decisions.append(dict(window=window_number,
                        padded_window_start=start, original_overlap_samples=overlap,
                        raw=decisions[(window_number, BASELINES[("raw", "or")])],
                        vmd=decisions[(window_number, BASELINES[("vmd", "or")])],
                        trial_support_policy="detector before padding; original-sample support invariant across overlaps",
                        inference_target_access=False, scoring_condition_access=False))
                    return result

            internal_methods = (*METHODS, *[method + "_zero_padding" for method in TRIAL_METHODS.values()])
            cleaned = multi_overlap_add(original, refs, estimate, methods=internal_methods,
                                        window=config["window"], hop=config["hop"])
            if len(decisions) != window_number * 5:
                raise ValueError("Trial cached decision coverage differs")
            for method in TRIAL_METHODS.values():
                np.testing.assert_array_equal(cleaned[method], cleaned.pop(method + "_zero_padding"))
                padding_checks += 1
            for method in ("identity", *BASELINES.values()):
                np.testing.assert_allclose(cleaned[method], archived[method], rtol=1e-7, atol=1e-12)
                parity += 1
            condition = reporting_condition(source_map[record], trial, rest_keys)
            seen_trials.add((record, trial))
            if record in record_channels:
                np.testing.assert_array_equal(record_channels[record], names[ids])
            record_channels[record] = names[ids].copy()
            outside = np.ones(len(names), bool); outside[ids] = False
            for method, values in cleaned.items():
                np.testing.assert_array_equal(values[outside], original[outside])
                reference = (original[ids] if condition == "rest" else
                    np.broadcast_to(evaluation[0 if condition == "lateral" else 1], original[ids].shape))
                moments.setdefault((record, condition, method), ConditionMoments(len(ids))).update(values[ids], reference)
                if condition == "rest":
                    scores = paired(values[ids], original[ids], config["fs"])
                    preservation.append(dict(record_id=record, trial=trial, method=method, rest_rmse=scores["rmse"],
                        **{key: scores[key] for key in ("relative_error", "alpha_db", "beta_db", "covariance")}))
                    spectra.append(dict(record_id=record, method=method,
                        segments=1 + (original.shape[-1] - 2 * config["fs"]) // config["fs"],
                        spectrum=resting_spectrum(values[ids], original[ids], config["fs"], normalization_band=(1., 40.))))
            np.savez_compressed(predictions / path.name, original=original, inference_references=refs,
                                evaluation_references=evaluation, channel_names=names, regions=regions, **cleaned)
        write_jsonl(decisions_directory / (path.stem + ".jsonl"), new_decisions)
        trials += 1
        pd.DataFrame(preservation).to_csv(output / "trial_support_rest_trials.csv", index=False)
        pd.DataFrame(disagreements).to_csv(output / "trial_support_disagreements.csv", index=False)
        print("TRIAL_SUPPORT_SAVED", record, trial, trials, flush=True)
    expected = {(record["record_id"], trial) for record in sources for trial in record["scoring_trial_ids"]}
    if trials != 51 or parity != 255 or padding_checks != 204 or seen_trials != expected:
        raise ValueError("Trial-support coverage or parity count differs")
    rows = []
    for (record, condition, method), summary in moments.items():
        before = moments[(record, condition, "identity")]
        for local, name in enumerate(record_channels[record]):
            rows.append(dict(source=source_map[record]["participant"], record_id=record, condition=condition,
                method=method, channel=str(name), samples=summary.count,
                rest_rmse=summary.rmse()[local] if condition == "rest" else np.nan,
                eeg_eog_signed_r_after=summary.correlation()[local] if condition != "rest" else np.nan,
                eeg_eog_abs_r_after=abs(summary.correlation()[local]) if condition != "rest" else np.nan,
                eeg_eog_abs_r_before=abs(before.correlation()[local]) if condition != "rest" else np.nan,
                units="original source array units; microvolt scale unverified",
                scoring_unit="complete condition concatenation; independent trial processing"))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "native_condition_channels.csv", index=False)
    write_native_reports(frame, spectra, output)
    records = pd.read_csv(output / "native_condition_participants.csv")
    previous = pd.read_csv(parent / "native_condition_participants.csv")
    columns, metric_parity = ["rest_rmse", "eeg_eog_abs_r_after", "eeg_eog_abs_r_before"], 0
    for method in ("identity", *BASELINES.values()):
        first = records[records.method == method].set_index(["source", "condition"])
        second = previous[previous.method == method].set_index(["source", "condition"])
        if set(first.index) != set(second.index) or len(first) != 8:
            raise ValueError("Baseline condition metric coverage differs")
        np.testing.assert_allclose(first[columns].sort_index(), second[columns].sort_index(),
                                   rtol=1e-7, atol=1e-12, equal_nan=True)
        metric_parity += len(first)
    rest_records = pd.DataFrame(preservation).groupby(["record_id", "method"])[
        ["rest_rmse", "relative_error", "alpha_db", "beta_db", "covariance"]].agg(
            lambda values: np.asarray(values, float).mean()).reset_index()
    rest_records.to_csv(output / "trial_support_rest_records.csv", index=False)
    bank = [dict(method=method, representation=representation, support=kind, support_position="original_trial")
            for (representation, kind), method in TRIAL_METHODS.items()]
    reviews = preservation_reviews(rest_records, bank, config["preservation"])
    atomic_json(output / "trial_support_gates.json", reviews)
    plot_gates(reviews, output)
    copies = output / "unchanged_parent_reports"; copies.mkdir()
    for name in ("native_condition_participants.csv", "native_frontal_preservation_records.csv", "native_frontal_preservation_gates.json"):
        shutil.copyfile(parent / name, copies / name)
    atomic_json(output / "trial_support_summary.json", dict(
        trials=trials, methods=len(METHODS), cached_detector_window_parity=support_checks,
        baseline_prediction_parity_checks=parity, baseline_condition_metric_parity_rows=metric_parity,
        reflected_zero_padding_parity_checks=padding_checks,
        inherited_failed_window_channel_entries=len(inherited_failures), failed_cases_retained=True,
        trial_support_preservation_passes=sum(row["method"] in TRIAL_METHODS.values() and row["native_preservation_gates_pass"] for row in reviews),
        runtime_s=time.perf_counter()-started, qualification=False, teacher_qualified=False,
        scope="development-exposed two-record support-boundary ablation", nonfrontal_changed=False,
        autovmd_executed=False, weights_trained=False, reserved_confirmation_opened=False,
        global_parameters_selected=False, artifact_fields_fitted=False, detector_reconstructed_from_unscored_calibration=True,
        native_reconstruction_snr_claim=False, detector_before_padding=True,
        parent_manifest_sha256=sha256_file(parent / "artifact_manifest.json"),
        interpretation="boundary sensitivity does not prove true ocular contamination or neural loss; suppression and preservation both required"))
