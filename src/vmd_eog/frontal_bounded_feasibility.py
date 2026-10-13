"""Replay a declared correction bank; never select a recipe on native rest."""
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .frontal_axis_diagnostic import diagnostic_parent, trial_identity, window_artifacts
from .frontal_methods import frontal_indices
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired
from .native_protocol import CONDITIONS, ConditionMoments, multi_overlap_add
from .paper_metrics import pearson


THRESHOLDS = (.4, .6, .8)
STRENGTHS = (.25, .5, .75, 1.)
BASELINES = {
    ("raw", "or"): "calibration_raw_event_gain",
    ("raw", "separate"): "calibration_raw_axis_event_gain",
    ("vmd", "or"): "K3_calibration_event_gain",
    ("vmd", "separate"): "K3_calibration_axis_event_gain",
}


def candidate_bank():
    return [dict(method=f"{representation}_{support}_a{int(threshold*100):02d}_s{int(strength*100):03d}",
                 representation=representation, support=support,
                 threshold=threshold, strength=strength)
            for representation, support in BASELINES
            for threshold in THRESHOLDS for strength in STRENGTHS]


def cached_admission(details, threshold):
    """Higher-threshold replay is valid only above the original .4 gate."""
    if threshold not in THRESHOLDS or not details:
        raise ValueError("Replay requires the declared threshold bank and saved decisions")
    first = details[0]
    association = first.get("field_association")
    gain = first.get("gain")
    if (association is None or gain is None or not np.isfinite(association) or
            not np.isfinite(gain) or not -1 <= association <= 1 or not 0 <= gain <= 8):
        raise ValueError("Invalid saved shared association/gain")
    if any(any(row.get(key) != first.get(key) for key in
               ("field_association", "gain", "triggered")) for row in details):
        raise ValueError("Cached montage-wide decisions differ")
    if first.get("triggered") != bool(association >= .4):
        raise ValueError("Cached admission does not match the original .4 gate")
    return bool(association >= threshold)


def replay_window(saved, decisions, bank):
    baseline = window_artifacts(saved)
    results = {"identity": baseline["identity"]}
    for candidate in bank:
        key = BASELINES[(candidate["representation"], candidate["support"])]
        strength = candidate["strength"]
        if strength not in STRENGTHS:
            raise ValueError("Replay requires the declared correction-strength bank")
        admitted = cached_admission(decisions[candidate["representation"]], candidate["threshold"])
        results[candidate["method"]] = baseline[key] * (strength if admitted else 0.)
    return results


def reporting_condition(record, trial, rest_keys):
    """Verified source labels enter reporting only, after inference."""
    if record.get("partition", {}).get("role") != "development":
        raise ValueError("Feasibility reporting requires development sources")
    if (trial not in record["scoring_trial_ids"] or
            set(record["scoring_trial_ids"]) & set(record["calibration_trial_ids"])):
        raise ValueError("Source scoring/calibration trial identities differ")
    condition = CONDITIONS.get(record["trial_labels"][trial])
    if condition is None or ((record["record_id"], trial) in rest_keys) != (condition == "rest"):
        raise ValueError("Saved condition labels disagree with resting reports")
    return condition


def preservation_reviews(records, bank, limits):
    fields = ["relative_error", "alpha_db", "beta_db", "covariance"]
    candidates = {row["method"]: row for row in bank}
    reviews = []
    for method, group in records.groupby("method"):
        finite = bool(np.isfinite(group[fields].to_numpy()).all())
        complete = set(group.record_id) == {"study01_p02_prep", "study01_p03_prep"}
        worst = {field: float(group[field].max()) if finite else None for field in fields}
        gates = dict(coverage=complete and finite,
                     clean_modification=finite and worst["relative_error"] <= limits["clean_change"],
                     clean_alpha=finite and worst["alpha_db"] <= limits["alpha_db"],
                     clean_beta=finite and worst["beta_db"] <= limits["beta_db"],
                     clean_covariance=finite and worst["covariance"] <= limits["covariance"])
        reviews.append(dict(**{**candidates.get(method, {}), "method": method},
                            worst_record=worst, gates=gates,
                            native_preservation_gates_pass=all(gates.values()),
                            qualification=False, global_recipe_selected=False))
    return reviews


def plot_preservation(reviews, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for representation, color in (("raw", "tab:orange"), ("vmd", "tab:blue")):
        for support, marker in (("or", "o"), ("separate", "x")):
            rows = [row for row in reviews if row.get("representation") == representation and row.get("support") == support]
            for axis, field in zip(axes, ("relative_error", "covariance")):
                axis.scatter([row["strength"] for row in rows],
                             [row["worst_record"][field] for row in rows],
                             label=f"{representation}/{support}", marker=marker, color=color, alpha=.7)
    for axis, threshold, label in zip(axes, (.01, .02), ("Relative resting modification", "Relative resting covariance error")):
        axis.axhline(threshold, color="black", linestyle="--", label="Unchanged gate")
        axis.set(xlabel="Correction multiplier (three association settings each)", ylabel=label)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Two exposed OSF records: worst-record preservation feasibility")
    fig.tight_layout()
    fig.savefig(output / "bounded_native_preservation.png", dpi=140)
    plt.close(fig)


def run(input_root, output, config, profile):
    if profile != "pilot":
        raise ValueError("Two exposed native records permit pilot feasibility scope only")
    parent = diagnostic_parent(input_root)
    started = time.perf_counter()
    bank = candidate_bank()
    methods = ["identity", *[row["method"] for row in bank]]
    rest = pd.read_csv(parent / "native_frontal_preservation_trials.csv")
    rest_keys = set(zip(rest.record_id, rest.trial.astype(int)))
    atomic_json(output / "bounded_candidates.json", bank)
    predictions = output / "bounded_native_predictions"
    decisions_directory = output / "bounded_native_decisions"
    predictions.mkdir(); decisions_directory.mkdir()
    sources = read_jsonl(parent / "native_frontal_sources.jsonl")
    source_map = {row["record_id"]: row for row in sources}
    if len(sources) != 2 or set(source_map) != {"study01_p02_prep", "study01_p03_prep"}:
        raise ValueError("Cached feasibility source cohort differs")
    preservation, correlations, parity, trials = [], [], 0, 0
    moments, record_channels, seen_trials = {}, {}, set()
    for path in sorted((parent / "native_frontal_predictions").glob("*.npz")):
        record, trial_id = trial_identity(path.stem)
        entries = read_jsonl(parent / "native_frontal_decisions" / (path.stem + ".jsonl"))
        keys = [(row["window"], row["method"]) for row in entries]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate cached window/method decisions")
        decisions = {key: row["decisions"] for key, row in zip(keys, entries)}
        with np.load(path, allow_pickle=False) as archived:
            original, refs = archived["original"], archived["inference_references"]
            evaluation_refs, names, regions = archived["evaluation_references"], archived["channel_names"], archived["regions"]
            with np.load(parent / "native_frontal_mode_examples" / (path.stem + ".npz"), allow_pickle=False) as modes:
                ids = frontal_indices(original, names, regions, np.isin(names, modes["channel_names"]))
                np.testing.assert_array_equal(names[ids], modes["channel_names"])
            window_number, trial_decisions = 0, []

            def estimate(values, references):
                nonlocal window_number
                window_number += 1
                cached = parent / "native_frontal_axis_windows" / f"{path.stem}-window-{window_number:04d}.npz"
                with np.load(cached, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(values, saved["original"])
                    np.testing.assert_array_equal(references, saved["references"])
                    raw_details = {representation: decisions[(window_number, BASELINES[(representation, "or")])]
                                   for representation in ("raw", "vmd")}
                    artifacts = replay_window(saved, raw_details, bank)
                    trial_decisions.append(dict(window=window_number,
                        raw=raw_details["raw"], vmd=raw_details["vmd"],
                        candidate_bank="bounded_candidates.json", inference_target_access=False,
                        scoring_condition_access=False))
                    return artifacts

            cleaned = multi_overlap_add(original, refs, estimate, methods=methods,
                                        window=config["window"], hop=config["hop"])
            if len(decisions) != window_number * 5:
                raise ValueError("Cached decision coverage differs")
            # Metadata is deliberately read after prediction, never by estimate().
            condition = reporting_condition(source_map[record], trial_id, rest_keys)
            is_rest = condition == "rest"
            seen_trials.add((record, trial_id))
            if record in record_channels:
                np.testing.assert_array_equal(record_channels[record], names[ids])
            record_channels[record] = names[ids].copy()
            np.testing.assert_allclose(cleaned["identity"], archived["identity"], rtol=1e-7, atol=1e-12)
            parity += 1
            for row in bank:
                if row["threshold"] == .4 and row["strength"] == 1.:
                    np.testing.assert_allclose(cleaned[row["method"]], archived[BASELINES[(row["representation"], row["support"])]], rtol=1e-7, atol=1e-12)
                    parity += 1
            outside = np.ones(len(names), bool); outside[ids] = False
            for method, values in cleaned.items():
                np.testing.assert_array_equal(values[outside], original[outside])
                reference = (original[ids] if is_rest else
                             np.broadcast_to(evaluation_refs[0 if condition == "lateral" else 1], original[ids].shape))
                moments.setdefault((record, condition, method), ConditionMoments(len(ids))).update(values[ids], reference)
                if is_rest:
                    scores = paired(values[ids], original[ids], config["fs"])
                    preservation.append(dict(record_id=record, trial=trial_id, method=method,
                        rest_rmse=scores["rmse"], **{key: scores[key] for key in ("relative_error", "alpha_db", "beta_db", "covariance")}))
                for axis, label in enumerate(("HEOG", "VEOG")):
                    reference = np.broadcast_to(evaluation_refs[axis], original[ids].shape)
                    before, after = pearson(original[ids], reference), pearson(values[ids], reference)
                    for local, index in enumerate(ids):
                        correlations.append(dict(record_id=record, trial=trial_id, method=method,
                            channel=str(names[index]), reference=label, rest_proxy=is_rest,
                            before_signed_r=before[local], after_signed_r=after[local],
                            before_abs_r=abs(before[local]), after_abs_r=abs(after[local]),
                            condition_label=condition,
                            scoring_unit="individual trial; exploratory dual-reference correlation"))
            np.savez_compressed(predictions / path.name, original=original, evaluation_references=evaluation_refs,
                                inference_references=refs, channel_names=names, regions=regions, **cleaned)
        write_jsonl(decisions_directory / (path.stem + ".jsonl"), trial_decisions)
        trials += 1
        pd.DataFrame(preservation).to_csv(output / "bounded_native_rest_trials.csv", index=False)
        pd.DataFrame(correlations).to_csv(output / "bounded_native_trial_correlations.csv", index=False)
        print("BOUNDED_NATIVE_TRIAL", record, trial_id, trials, flush=True)
    if trials != 51 or parity != 255:
        raise ValueError("Bounded native baseline coverage differs")
    expected_trials = {(row["record_id"], trial) for row in sources for trial in row["scoring_trial_ids"]}
    if seen_trials != expected_trials:
        raise ValueError("Source-ledger scoring trial coverage differs")
    condition_rows = []
    for (record, condition, method), summary in moments.items():
        before = moments[(record, condition, "identity")]
        for local, name in enumerate(record_channels[record]):
            condition_rows.append(dict(source=source_map[record]["participant"], record_id=record,
                condition=condition, method=method, channel=str(name), samples=summary.count,
                rest_rmse=summary.rmse()[local] if condition == "rest" else np.nan,
                eeg_eog_signed_r_after=summary.correlation()[local] if condition != "rest" else np.nan,
                eeg_eog_abs_r_after=abs(summary.correlation()[local]) if condition != "rest" else np.nan,
                eeg_eog_abs_r_before=abs(before.correlation()[local]) if condition != "rest" else np.nan,
                scoring_unit="complete condition concatenation, independent trial processing"))
    condition_frame = pd.DataFrame(condition_rows)
    condition_frame.to_csv(output / "bounded_native_condition_channels.csv", index=False)
    columns = ["rest_rmse", "eeg_eog_abs_r_after", "eeg_eog_abs_r_before"]
    condition_records = condition_frame.groupby(["source", "record_id", "condition", "method"])[columns].agg(
        lambda values: np.asarray(values, float).mean()).reset_index()
    condition_records.to_csv(output / "bounded_native_condition_records.csv", index=False)
    previous_conditions = pd.read_csv(parent / "native_condition_participants.csv")
    metric_parity = 0
    for method, previous in [("identity", "identity"), *[
            (row["method"], BASELINES[(row["representation"], row["support"])])
            for row in bank if row["threshold"] == .4 and row["strength"] == 1.]]:
        first = condition_records[condition_records.method == method].set_index(["source", "condition"])
        second = previous_conditions[previous_conditions.method == previous].set_index(["source", "condition"])
        if set(first.index) != set(second.index) or len(first) != 8:
            raise ValueError("Cached condition metric coverage differs")
        np.testing.assert_allclose(first[columns].sort_index(), second[columns].sort_index(),
                                   rtol=1e-7, atol=1e-12, equal_nan=True)
        metric_parity += len(first)
    frame = pd.DataFrame(preservation)
    fields = ["rest_rmse", "relative_error", "alpha_db", "beta_db", "covariance"]
    records = frame.groupby(["record_id", "method"])[fields].agg(lambda values: np.asarray(values, float).mean()).reset_index()
    records.to_csv(output / "bounded_native_rest_records.csv", index=False)
    reviews = preservation_reviews(records, bank, config["preservation"])
    atomic_json(output / "bounded_native_gates.json", reviews)
    plot_preservation(reviews, output)
    copies = output / "unchanged_parent_reports"; copies.mkdir()
    for name in ("native_condition_participants.csv", "native_frontal_preservation_records.csv", "native_frontal_preservation_gates.json"):
        shutil.copyfile(parent / name, copies / name)
    atomic_json(output / "bounded_native_summary.json", dict(
        trials=trials, active_arms=len(bank), settings=12, baseline_parity_checks=parity,
        baseline_condition_metric_parity_rows=metric_parity,
        active_preservation_passes=sum(row["method"] != "identity" and row["native_preservation_gates_pass"] for row in reviews),
        parent_manifest_sha256=sha256_file(parent / "artifact_manifest.json"), runtime_s=time.perf_counter()-started,
        qualification=False, global_recipe_selected=False, parameters_fitted=False,
        autovmd_executed=False, weights_trained=False, reserved_confirmation_opened=False,
        native_reconstruction_snr_claim=False, nonfrontal_changed=False,
        scope="exhaustive development-exposed pilot feasibility; not grouped selection",
        condition_reporting="verified cached source-ledger labels; complete-condition ordinary correlation; labels never enter inference",
        permutation_status="no significance/winner claim for the two-record candidate feasibility screen",
        interpretation="native preservation necessary but insufficient; identity pass is not cleaning qualification"))
