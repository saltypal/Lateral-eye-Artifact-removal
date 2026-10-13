"""Explain saved native interventions without fitting or changing predictions."""
import re
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .frontal_axis_native import checked_parent
from .frontal_methods import frontal_indices
from .io import atomic_json, read_jsonl, sha256_file
from .native_protocol import multi_overlap_add


PARENT = {
    "run_id": "frontal-native-separate-axis-001",
    "git_sha": "ce6f31d672f5061cfaae44a22c2318e4e4afc351",
    "manifest": "63055a6fa7ed3c17b767e6ba64e0f7ad845b5d40af0afe0b78d59da7c83ca049",
}
METHODS = ("identity", "calibration_raw_event_gain", "K3_calibration_event_gain",
           "calibration_raw_axis_event_gain", "K3_calibration_axis_event_gain")


def diagnostic_parent(input_root):
    parent, summary = checked_parent(input_root, "native_frontal_summary.json", PARENT)
    if (summary.get("axis_support_experiment") is not True or
            summary.get("predecessor_prediction_parity_checks") != 153 or
            summary.get("scoring_trials") != 51 or summary.get("recordings") != 2 or
            summary.get("teacher_qualified") is not False or
            summary.get("nonfrontal_changed") is not False):
        raise ValueError("Native diagnostic precedent or qualification scope differs")
    return parent


def trial_identity(stem):
    match = re.fullmatch(r"(.+)-trial-(\d+)", stem)
    if match is None:
        raise ValueError("Invalid saved trial identity")
    return match.group(1), int(match.group(2))


def original_overlap(window_number, window, hop, trial_length):
    if window_number < 1 or not 0 < hop <= window or trial_length < 2:
        raise ValueError("Invalid overlap identity")
    start = (window_number - 1) * hop
    valid_start = window // 2
    valid_end = valid_start + trial_length
    samples = max(0, min(start + window, valid_end) - max(start, valid_start))
    return start, samples


def window_artifacts(saved):
    """Assemble the exact cached intervention; no EEG target is an input."""
    original = np.asarray(saved["original"], float)
    if original.ndim != 2 or not np.isfinite(original).all():
        raise ValueError("Invalid cached original EEG")
    supports = np.asarray(saved["axis_supports"], float)
    union = np.asarray(saved["union_support"], float)
    if (supports.shape != (2, original.shape[-1]) or union.shape != (original.shape[-1],) or
            not np.isfinite(supports).all() or not np.isfinite(union).all()):
        raise ValueError("Cached supports differ from the sample contract")
    result = {"identity": np.zeros_like(original)}
    for representation, prefix in (("raw", "calibration_raw"), ("vmd", "K3_calibration")):
        fields = np.asarray(saved[representation + "_axis_fields"], float)
        if fields.shape != (2, *original.shape) or not np.isfinite(fields).all():
            raise ValueError("Invalid cached signed axis fields")
        result[prefix + "_event_gain"] = fields.sum(0) * union[None]
        result[prefix + "_axis_event_gain"] = (fields * supports[:, None]).sum(0)
    return result


def window_measures(saved, indices, details, representation):
    """Engineering mechanism diagnostics, separate from primary paper metrics."""
    original = np.asarray(saved["original"], float)[indices]
    fields = np.asarray(saved[representation + "_axis_fields"], float)[:, indices]
    supports = np.asarray(saved["axis_supports"], float)
    union = np.asarray(saved["union_support"], float)
    joint = fields.sum(0)
    original_energy = float(np.sum(original ** 2))
    axis_energy = np.sum(fields ** 2, axis=(1, 2))
    cross = float(2 * np.sum(fields[0] * fields[1]))
    separate = (fields * supports[:, None]).sum(0)
    combined = joint * union[None]
    if not details:
        raise ValueError("Missing saved shared-gain decisions")
    first = details[0]
    for row in details:
        if any(row.get(key) != first.get(key) for key in ("gain", "field_association", "triggered")):
            raise ValueError("Saved montage-wide gain or association differs across channels")
    def relative(values):
        if original_energy <= 0:
            return np.nan
        return float(np.sqrt(np.sum(values ** 2) / original_energy))
    return dict(association=first["field_association"], shared_gain=first["gain"],
                triggered=bool(first["triggered"]), failed_channels=sum(bool(row.get("failure")) for row in details),
                heog_support_mean=float(supports[0].mean()), veog_support_mean=float(supports[1].mean()),
                union_support_mean=float(union.mean()),
                heog_only=bool(np.any(supports[0]) and not np.any(supports[1])),
                veog_only=bool(np.any(supports[1]) and not np.any(supports[0])),
                nonzero_field=bool(axis_energy.sum() > 0),
                heog_only_intervention=bool(first["triggered"] and np.any(separate) and np.any(supports[0]) and not np.any(supports[1])),
                veog_only_intervention=bool(first["triggered"] and np.any(separate) and np.any(supports[1]) and not np.any(supports[0])),
                separate_nonzero_intervention=bool(np.any(separate)),
                heog_field_energy=float(axis_energy[0]), veog_field_energy=float(axis_energy[1]),
                field_cross_term=cross, joint_field_energy=float(np.sum(joint ** 2)),
                cancellation_fraction=-cross / float(axis_energy.sum()) if axis_energy.sum() > 0 else np.nan,
                or_correction_fraction=relative(combined), separate_correction_fraction=relative(separate))


def plot_window(saved, indices, destination, title, fs):
    times = np.arange(saved["original"].shape[-1]) / fs
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(times, np.asarray(saved["original"])[indices].T, alpha=.35)
    axes[0].set_ylabel("Source EEG units")
    for axis, label in enumerate(("HEOG", "VEOG")):
        axes[1].plot(times, saved["axis_supports"][axis], label=label)
    axes[1].plot(times, saved["union_support"], linestyle="--", label="OR support")
    axes[1].set_ylabel("Support")
    axes[1].legend()
    fields = saved["vmd_axis_fields"][:, indices]
    or_lines = axes[2].plot(times, (fields.sum(0) * saved["union_support"]).T, alpha=.3, color="tab:orange")
    separate_lines = axes[2].plot(times, (fields * saved["axis_supports"][:, None]).sum(0).T, alpha=.3, color="tab:blue")
    or_lines[0].set_label("OR-support correction")
    separate_lines[0].set_label("Separate-axis correction")
    axes[2].legend()
    axes[2].set_ylabel("Estimated correction")
    axes[2].set_xlabel("Padded window time (seconds)")
    fig.suptitle(title + " — rest preservation proxy; not paired clean truth")
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def run(input_root, output, config, profile):
    if profile != "pilot":
        raise ValueError("Saved two-record diagnostic has pilot scope only")
    parent = diagnostic_parent(input_root)
    started = time.perf_counter()
    rest = pd.read_csv(parent / "native_frontal_preservation_trials.csv")
    rest_keys = set(zip(rest.record_id, rest.trial.astype(int)))
    rows, parity, trials = [], 0, 0
    for path in sorted((parent / "native_frontal_predictions").glob("*.npz")):
        record, trial = trial_identity(path.stem)
        decisions = read_jsonl(parent / "native_frontal_decisions" / (path.stem + ".jsonl"))
        keys = [(row["window"], row["method"]) for row in decisions]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate cached window/method decisions")
        decisions = {key: row["decisions"] for key, row in zip(keys, decisions)}
        with np.load(path, allow_pickle=False) as saved_trial:
            original, references = saved_trial["original"], saved_trial["inference_references"]
            names, regions = saved_trial["channel_names"], saved_trial["regions"]
            with np.load(parent / "native_frontal_mode_examples" / (path.stem + ".npz"), allow_pickle=False) as mode_example:
                verified_names = mode_example["channel_names"]
                indices = frontal_indices(original, names, regions, np.isin(names, verified_names))
                if not np.array_equal(names[indices], verified_names):
                    raise ValueError("Saved verified frontal row identities differ")
            window_number = 0

            def estimate(values, refs):
                nonlocal window_number
                window_number += 1
                window_path = parent / "native_frontal_axis_windows" / f"{path.stem}-window-{window_number:04d}.npz"
                with np.load(window_path, allow_pickle=False) as saved_window:
                    np.testing.assert_array_equal(values, saved_window["original"])
                    np.testing.assert_array_equal(refs, saved_window["references"])
                    estimates = window_artifacts(saved_window)
                    for representation, method in (("raw", "calibration_raw_event_gain"), ("vmd", "K3_calibration_event_gain")):
                        details = decisions[(window_number, method)]
                        row = window_measures(saved_window, indices, details, representation)
                        window_start, overlap_samples = original_overlap(
                            window_number, config["window"], config["hop"], original.shape[-1])
                        row.update(record_id=record, trial=trial, window=window_number,
                                   window_start=window_start, original_overlap_samples=overlap_samples,
                                   representation=representation, rest_proxy=(record, trial) in rest_keys)
                        rows.append(row)
                return estimates

            reconstructed = multi_overlap_add(original, references, estimate, methods=METHODS,
                                               window=config["window"], hop=config["hop"])
            if len(decisions) != window_number * len(METHODS):
                raise ValueError("Cached decision coverage differs from reconstructed windows")
            for method, values in reconstructed.items():
                np.testing.assert_allclose(values, saved_trial[method], rtol=1e-7, atol=1e-12)
                parity += 1
            trials += 1
        pd.DataFrame(rows).to_csv(output / "axis_diagnostic_windows.csv", index=False)
        print("AXIS_DIAGNOSTIC_TRIAL", record, trial, window_number, flush=True)
    if trials != 51 or parity != 255:
        raise ValueError("Saved native diagnostic cohort coverage differs")
    frame = pd.DataFrame(rows)
    contributing = frame[frame.original_overlap_samples > 0]
    summaries = contributing.groupby(["record_id", "representation", "rest_proxy"]).agg(
        windows=("window", "size"), triggered_windows=("triggered", "sum"),
        heog_only_detector_windows=("heog_only", "sum"), veog_only_detector_windows=("veog_only", "sum"),
        heog_only_intervention_windows=("heog_only_intervention", "sum"),
        veog_only_intervention_windows=("veog_only_intervention", "sum"),
        nonzero_field_windows=("nonzero_field", "sum"),
        separate_nonzero_intervention_windows=("separate_nonzero_intervention", "sum"),
        association_mean=("association", "mean"), shared_gain_mean=("shared_gain", "mean"),
        union_support_mean=("union_support_mean", "mean"),
        or_fraction_mean=("or_correction_fraction", "mean"),
        separate_fraction_mean=("separate_correction_fraction", "mean"),
        cancellation_mean=("cancellation_fraction", "mean"),
        cancellation_available_windows=("cancellation_fraction", "count"),
        correction_fraction_available_windows=("separate_correction_fraction", "count"),
    ).reset_index()
    summaries.to_csv(output / "axis_diagnostic_record_summaries.csv", index=False)
    resting = contributing[contributing.rest_proxy].sort_values("separate_correction_fraction", ascending=False)
    resting.to_csv(output / "axis_diagnostic_rest_ranked.csv", index=False)
    for record, group in resting[resting.representation == "vmd"].groupby("record_id"):
        row = group.iloc[0]
        stem = f"{record}-trial-{int(row.trial):04d}"
        with np.load(parent / "native_frontal_axis_windows" / f"{stem}-window-{int(row.window):04d}.npz", allow_pickle=False) as cached:
            with np.load(parent / "native_frontal_predictions" / (stem + ".npz"), allow_pickle=False) as trial:
                with np.load(parent / "native_frontal_mode_examples" / (stem + ".npz"), allow_pickle=False) as mode_example:
                    mask = np.isin(trial["channel_names"], mode_example["channel_names"])
                    indices = frontal_indices(trial["original"], trial["channel_names"], trial["regions"], mask)
                plot_window(cached, indices, output / f"axis_diagnostic_{record}.png",
                            f"{record}, trial {int(row.trial)}, window {int(row.window)}; largest contributing-window VMD separate-support resting fraction", config["fs"])
    copies = output / "unchanged_parent_reports"
    copies.mkdir()
    for name in ("native_condition_participants.csv", "native_frontal_preservation_records.csv", "native_frontal_preservation_gates.json"):
        shutil.copyfile(parent / name, copies / name)
    atomic_json(output / "axis_diagnostic_summary.json", dict(
        profile=profile, trials=trials, window_rows=len(frame), contributing_window_rows=len(contributing),
        zero_overlap_window_rows=int((frame.original_overlap_samples == 0).sum()), reconstruction_parity_checks=parity,
        parent_manifest_sha256=sha256_file(parent / "artifact_manifest.json"), runtime_s=time.perf_counter()-started,
        parameters_fitted=False, predictions_changed=False, autovmd_executed=False,
        weights_trained=False, reserved_confirmation_opened=False, qualification=False,
        scoring_labels_use="rest labels only for saved-output stratification; no inference/recipe selection",
        window_dependence="overlapping padded windows are descriptive, not independent observations",
        overlap_policy="all cached windows reconstructed; means/ranking exclude zero-original-overlap windows; energy fractions are full padded-window diagnostics, not overlap-weighted trial modification",
        primary_metrics="unchanged parent paper/native reports; new energy quantities are mechanism diagnostics",
        mean_denominators="cancellation means exclude zero-field windows and correction fractions exclude zero-input energy; explicit available-window counts saved",
        support_counts="detector-only support counts distinguished from triggered nonzero-field interventions",
        causal_limit="cached fields describe decisions/cancellation; unpaired rest cannot establish true artifact, field-estimation error or neural loss",
        mask_note="exact verified frontal identities from saved mode-example metadata; failed fields remain zero",
    ))
