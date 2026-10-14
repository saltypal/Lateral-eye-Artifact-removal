"""Recoverable fixed-recipe posterior/central comparisons on Kaggle."""
import json
import time

import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .experiments import corpus_parent
from .frontal_review import controlled_gates, strict_mean
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired
from .nonfrontal_challengers import METHODS, estimate_fields, fit_experts, replace_regions
from .paper_evaluation import save_protocol
from .paper_metrics import paired_channels
from .posterior_context import PAPER_KEYS
from .regional_coverage import select_rows
from .regional_diagnostic import GROUPS
from .regional_integration import unique_artifact


BASELINE = "central_frontal_vmd_context"
RAW_CONTEXT = "central_frontal_raw_context"
RAW_FRONTAL = "raw_frontal_comparator"
COMPOSITIONS = {"posterior": (1,), "central": (2,), "both": (1, 2)}


def validate_family(config):
    scope, family = config.get("scope"), config.get("method_family")
    valid = ((scope == "fixed_nonfrontal_component_challengers" and family in (None, "component_challengers"))
             or (scope == "fixed_nonfrontal_mwf_controls" and family == "mwf_controls")
             or (scope == "fixed_nonfrontal_cca_adaptation" and family == "cca_adaptation"))
    if not valid:
        raise ValueError("Nonfrontal scope/method-family pairing differs")
    return family


def readiness(input_root, config):
    validate_family(config)
    if (config.get("scope") not in ("fixed_nonfrontal_component_challengers", "fixed_nonfrontal_mwf_controls", "fixed_nonfrontal_cca_adaptation")
            or config.get("autovmd_enabled") is not False
            or config.get("reserved_access") != "closed until verified freeze"):
        raise ValueError("Nonfrontal experiment policy differs")
    corpus, rows = corpus_parent(input_root)
    if sha256_file(corpus / "split_manifest.json") != config["source_split_sha256"]:
        raise ValueError("Frozen source split differs")
    parent = unique_artifact(input_root, "coverage_summary.json").parent
    expected = config["context_parent"]
    if sha256_file(parent / "artifact_manifest.json") != expected["manifest_sha256"]:
        raise ValueError("Frozen context manifest differs")
    verify_parent(parent, ("coverage_summary.json", "run_spec.json", "stage_tests_summary.json",
                           "coverage_selection_manifest.jsonl", "coverage_channel_scores.csv.gz"))
    spec = json.loads((parent / "run_spec.json").read_text())
    if any(spec.get(key) != expected[key] for key in ("run_id", "git_sha", "campaign_id")):
        raise ValueError("Frozen context source identity differs")
    summary = json.loads((parent / "coverage_summary.json").read_text())
    if (summary.get("examples") != 320 or summary.get("profile") != "full"
            or summary.get("reserved_confirmation_opened") is not False
            or summary.get("source_split_sha256") != config["source_split_sha256"]
            or json.loads((parent / "stage_tests_summary.json").read_text()).get("passed") is not True):
        raise ValueError("Frozen context completion/contract differs")
    selected = read_jsonl(parent / "coverage_selection_manifest.jsonl")
    mapping = {row["example_id"]: row for row in rows}
    if any(mapping.get(row["example_id"]) != row for row in selected):
        raise ValueError("Frozen context selection differs from corpus")
    # Same grouped split checks as the posterior campaign; no source can change fold.
    for source in ("recipient", "donor"):
        buckets = {}
        for row in selected:
            fold = row["partition"]["fold"]
            if row["partition"]["role"] != "development":
                raise ValueError("Reserved source entered experiment")
            if row[source] in buckets and buckets[row[source]] != fold:
                raise ValueError("Source identity crosses frozen folds")
            buckets[row[source]] = fold
    return corpus, parent, selected


def save_experts(directory, fitted):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {"ids": fitted.ids, "field_coefficients": fitted.field_model.coefficients,
              "reference_mean": fitted.field_model.reference_mean,
              "reference_scale": fitted.field_model.reference_scale,
              "active_columns": fitted.field_model.active_columns,
              "lags": np.asarray(fitted.field_model.lags)}
    if fitted.ica is not None:
        arrays.update(ica_mean=fitted.ica.mean, unmixing=fitted.ica.unmixing,
                      mixing=fitted.ica.mixing, selected=fitted.ica.selected,
                      associations=fitted.ica.associations, rest_center=fitted.rest_center,
                      rest_scale=fitted.rest_scale, component_coefficients=fitted.component_model.coefficients,
                      component_reference_mean=fitted.component_model.reference_mean,
                      component_reference_scale=fitted.component_model.reference_scale,
                      component_active_columns=fitted.component_model.active_columns)
    for rank, (projection, spectrum, retained) in fitted.projections.items():
        arrays["rank" + str(rank) + "_projection"] = projection
        arrays["rank" + str(rank) + "_spectrum"] = spectrum
    np.savez_compressed(directory / "fitted_arrays.npz", **arrays)
    atomic_json(directory / "fit_status.json", {"failures": fitted.failures,
        "channel_count": len(fitted.ids),
        "component_count": fitted.ica.unmixing.shape[0] if fitted.ica is not None else None,
        "selected_components": fitted.ica.selected.tolist() if fitted.ica is not None else [],
        "field_metadata": fitted.field_model.metadata,
        "component_field_metadata": fitted.component_model.metadata if fitted.component_model is not None else None,
        "scoring_target_access": False})


def load_experts(directory):
    """Load saved numerical fits without refitting, with explicit model metadata."""
    from .nonfrontal_challengers import NonfrontalExperts
    from .nonfrontal_ridge import FrozenRidgeModel
    from .posterior import ICAExpert
    arrays = dict(np.load(directory / "fitted_arrays.npz", allow_pickle=False))
    status = json.loads((directory / "fit_status.json").read_text())
    def ridge(prefix, metadata):
        return FrozenRidgeModel(tuple(int(x) for x in arrays["lags"]), metadata["penalty"],
            arrays[prefix + "reference_mean"], arrays[prefix + "reference_scale"],
            arrays[prefix + "active_columns"], arrays[prefix + "coefficients"],
            metadata["calibration_valid_samples"], metadata["calibration_channel_count"],
            metadata["predictor_count"])
    # Field coefficients use the archival name field_coefficients.
    arrays["coefficients"] = arrays["field_coefficients"]
    field_model = ridge("", status["field_metadata"])
    expert = component = center = scale = None
    if "unmixing" in arrays:
        expert = ICAExpert(arrays["ica_mean"], arrays["unmixing"], arrays["mixing"],
                           arrays["selected"], arrays["associations"], True)
        component = ridge("component_", status["component_field_metadata"])
        center, scale = arrays["rest_center"], arrays["rest_scale"]
    projections = {}
    for rank in (1, 2):
        key = "rank" + str(rank)
        if key + "_projection" in arrays:
            projections[rank] = (arrays[key + "_projection"], arrays[key + "_spectrum"], min(rank, len(arrays["ids"])))
    return NonfrontalExperts(arrays["ids"], field_model, expert, component, center, scale,
                            projections, status["failures"])


def labelled_gates(scores, config):
    """Comparator identities are explicit; do not mislabel a context loss as direct."""
    output = {}
    for scope, group in scores.groupby("scope"):
        comparisons = {BASELINE: "cc_deterioration_vs_vmd_context", RAW_CONTEXT: "cc_deterioration_vs_raw_context"}
        if scope == "frontal":
            comparisons[RAW_FRONTAL] = "cc_deterioration"
        output[scope] = {}
        for comparator, column in comparisons.items():
            reports = controlled_gates(group.assign(cc_deterioration=group[column]), config)
            for report in reports:
                report["comparator"] = comparator
                report["mean_cc_deterioration_vs_comparator"] = report.pop("mean_cc_deterioration_vs_direct")
            output[scope][comparator] = reports
    return output


def reconcile_baselines(scores, old_scores):
    """Exact cohort/metadata matching before tolerant saved paper-metric parity."""
    keys = ["example_id", "scope", "channel", "evaluated_family"]
    columns = keys + ["recipient", "donor", "condition", "input_snr_db"] + list(PAPER_KEYS)
    current = scores[scores.evaluated_family.isin(("identity", BASELINE, RAW_CONTEXT))][columns]
    parent = old_scores[old_scores.example_id.isin(current.example_id.unique()) &
                        old_scores.evaluated_family.isin(("identity", BASELINE, RAW_CONTEXT))][columns]
    if current.duplicated(keys).any() or parent.duplicated(keys).any():
        raise ValueError("Baseline metric keys are not unique")
    left = current.set_index(keys).sort_index()
    right = parent.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Baseline metric cohort differs")
    for name in ("recipient", "donor", "condition", "input_snr_db"):
        if not left[name].equals(right[name]):
            raise ValueError("Baseline metric source metadata differs: " + name)
    for name in PAPER_KEYS:
        if not np.allclose(left[name], right[name], rtol=1e-10, atol=1e-18, equal_nan=True):
            raise ValueError("Baseline paper metric parity failed: " + name)
    return len(left) * len(PAPER_KEYS)


def plot_results(output):
    import matplotlib.pyplot as plt
    table = pd.read_csv(output / "nonfrontal_method_means.csv")
    for scope in ("whole", "posterior", "central_shared"):
        selected = table[(table.scope == scope) &
                         (table.evaluated_family.str.endswith(":both") |
                          table.evaluated_family.isin((BASELINE, RAW_CONTEXT, "identity")))]
        figure, axis = plt.subplots(figsize=(11, 5))
        axis.barh(selected.evaluated_family, selected.snr_energy_db)
        axis.axvline(15., color="black", linestyle="--", label="15 dB target")
        axis.set_xlabel("Recipient-balanced channel-first output SNR (dB)")
        axis.set_title(scope + ": fixed-recipe development comparisons; qualification false")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / ("nonfrontal_" + scope + ".png"), dpi=130)
        plt.close(figure)


def run(input_root, output, config, profile):
    started = time.perf_counter()
    family = validate_family(config)
    if family == "cca_adaptation":
        from .nonfrontal_cca import METHODS as methods, fit_experts as fit_function
        from .nonfrontal_cca import estimate_fields as estimate_function
        from .nonfrontal_cca import save_experts as save_function, load_experts as load_function
    elif family == "mwf_controls":
        from .nonfrontal_mwf import METHODS as methods, fit_experts as fit_function
        from .nonfrontal_mwf import estimate_fields as estimate_function
        from .nonfrontal_mwf import save_experts as save_function, load_experts as load_function
    else:
        methods, fit_function, estimate_function = METHODS, fit_experts, estimate_fields
        save_function, load_function = save_experts, load_experts
    corpus, parent, all_rows = readiness(input_root, config)
    rows = select_rows(all_rows, profile)
    write_jsonl(output / "nonfrontal_selection_manifest.jsonl", rows)
    save_protocol(output)
    predictions = output / "nonfrontal_predictions"
    shards = output / "nonfrontal_metric_shards"
    predictions.mkdir()
    shards.mkdir()
    old_scores = pd.read_csv(parent / "coverage_channel_scores.csv.gz")
    inherited = old_scores[(old_scores.scope == "whole") &
                           (old_scores.evaluated_family == BASELINE)]
    prior_failures = inherited.groupby("example_id").failure.first().to_dict()
    fitted_by_calibration, records, fit_records, decisions = {}, [], [], []
    coverage_rows = []
    parity_checks = fit_replay_checks = 0
    for number, row in enumerate(rows):
        print("NONFRONTAL_START", number + 1, len(rows), row["example_id"], flush=True)
        verify_parent(corpus, (row["array_path"], row["calibration"]))
        relative = "coverage_predictions/" + row["example_id"] + ".npz"
        verify_parent(parent, (relative,))
        data = dict(np.load(corpus / row["array_path"], allow_pickle=False))
        calibration = dict(np.load(corpus / row["calibration"], allow_pickle=False))
        saved = dict(np.load(parent / relative, allow_pickle=False))
        original = data["eeg"]
        for key, stored in (("eeg", "original"), ("paired_reference", "target"),
                            ("channel_names", "names"), ("regions", "regions"), ("mask", "mask")):
            if not np.array_equal(data[key], saved[stored]):
                raise ValueError("Frozen parent waveform/metadata differs: " + key)
        for key in ("channel_names", "regions", "mask", "hemispheres"):
            if not np.array_equal(data[key], calibration[key]):
                raise ValueError("Calibration montage metadata differs: " + key)
        key = row["calibration"]
        if key not in fitted_by_calibration:
            fit_started = time.perf_counter()
            # Producer targets/true fields are never forwarded to fitting.
            cal_input = {name: calibration[name] for name in
                         ("eeg", "references", "mask", "trial_types", "boundaries")}
            fitted = fit_function(cal_input, config["challenger_policy"])
            fitted_by_calibration[key] = fitted
            identity = sha256_file(corpus / key)
            fit_directory = output / "nonfrontal_fits" / identity
            save_function(fit_directory, fitted)
            reloaded = load_function(fit_directory)
            probe = (original, data["references"], data["regions"], data["mask"])
            live_fields, live_details, live_failures = estimate_function(*probe, fitted, config["challenger_policy"])
            saved_fields, saved_details, saved_failures = estimate_function(*probe, reloaded, config["challenger_policy"])
            for method in methods:
                if not np.array_equal(live_fields[method], saved_fields[method]):
                    raise RuntimeError("Saved-fit replay differs: " + method)
                fit_replay_checks += 1
            if live_details != saved_details or live_failures != saved_failures:
                raise RuntimeError("Saved-fit decisions differ")
            fit_records.append({"calibration": key, "sha256": identity,
                "runtime_s": time.perf_counter() - fit_started, "failures": fitted.failures,
                "paired_target_access": False, "reserved_source_access": False})
        apply_started = time.perf_counter()
        artifacts, detail, failures = estimate_function(original, data["references"], data["regions"],
            data["mask"], fitted_by_calibration[key], config["challenger_policy"])
        apply_runtime = time.perf_counter() - apply_started
        estimates = {"identity": original, BASELINE: saved[BASELINE], RAW_CONTEXT: saved[RAW_CONTEXT]}
        for method, artifact in artifacts.items():
            for composition, regions in COMPOSITIONS.items():
                name = method + ":" + composition
                estimates[name] = replace_regions(original, saved[BASELINE], artifact,
                    data["regions"], data["mask"], regions)
                frontal = data["mask"] & (data["regions"] == 0)
                if not np.array_equal(estimates[name][frontal], saved[BASELINE][frontal]):
                    raise RuntimeError("Frozen frontal output changed")
                parity_checks += 1
        available = np.flatnonzero(data["mask"])
        scopes = {"whole": available, **{label: np.flatnonzero(data["mask"] & (data["regions"] == index))
                  for index, label in GROUPS.items()}}
        scopes = {name: ids for name, ids in scopes.items() if len(ids)}
        meta = {key: row[key] for key in ("example_id", "recipient", "donor", "condition", "input_snr_db")}
        meta["fold"] = row["partition"]["fold"]
        example_records = []
        prior = prior_failures.get(row["example_id"])
        prior = prior if isinstance(prior, str) and prior else None
        for arm, cleaned in estimates.items():
            method = arm.split(":")[0]
            failure = "; ".join(x for x in (prior if arm != "identity" else None,
                                           failures.get(method)) if x) or None
            for scope, ids in scopes.items():
                # Clean target access starts only after every estimate is fixed.
                target, raw = data["paired_reference"][ids], original[ids]
                paper = paired_channels(cleaned[ids], target, raw)
                comparator_name = RAW_FRONTAL if scope == "frontal" else RAW_CONTEXT
                comparison = paired_channels(saved[comparator_name][ids], target, raw)
                vmd_comparison = paired_channels(saved[BASELINE][ids], target, raw)
                raw_comparison = paired_channels(saved[RAW_CONTEXT][ids], target, raw)
                engineering = paired(cleaned[ids], target, config["fs"])
                for local, index in enumerate(ids):
                    example_records.append({**meta, "scope": scope, "channel": str(data["channel_names"][index]),
                        "evaluated_family": arm, "failure": failure, "comparator": comparator_name,
                        **{name: float(value[local]) for name, value in paper.items()},
                        "cc_deterioration": float(comparison["pearson_cc"][local] - paper["pearson_cc"][local]),
                        "cc_deterioration_vs_vmd_context": float(vmd_comparison["pearson_cc"][local] - paper["pearson_cc"][local]),
                        "cc_deterioration_vs_raw_context": float(raw_comparison["pearson_cc"][local] - paper["pearson_cc"][local]),
                        "clean_change": float(paper["rrmse_time"][local]),
                        "alpha_db": engineering["alpha_db"], "beta_db": engineering["beta_db"],
                        "covariance_error": engineering["covariance"], "apply_runtime_s": apply_runtime})
        records.extend(example_records)
        np.savez_compressed(predictions / (row["example_id"] + ".npz"),
            original=original, target=data["paired_reference"], names=data["channel_names"],
            regions=data["regions"], mask=data["mask"], **estimates)
        decisions.append({**meta, "details": detail, "failures": failures,
                          "inherited_failure": prior, "apply_runtime_s": apply_runtime})
        if family == "cca_adaptation":
            # Evaluator-only coverage uses paired truth after every estimate exists.
            # It cannot change admission, direction selection, gain or prediction.
            for method in methods:
                for region, label in ((1, "posterior"), (2, "central_shared")):
                    ids = np.flatnonzero(data["mask"] & (data["regions"] == region))
                    if not len(ids):
                        continue
                    truth = original[ids] - data["paired_reference"][ids]
                    info = detail[method]["regions"].get(str(region), {})
                    coverage_rows.append({**meta, "method": method, "scope": label,
                        "triggered": bool(info.get("triggered", False)), "method_failure": failures.get(method),
                        "true_artifact_energy": float(np.sum(truth ** 2)),
                        "estimated_artifact_energy": float(np.sum(artifacts[method][ids] ** 2)),
                        "association": info.get("association"), "gain": info.get("gain"),
                        "generation_multiplier": row["recipe"]["amplitude"],
                        "legacy_reference_scaled_with_field": False, "evaluation_only": True})
        atomic_json(predictions / (row["example_id"] + ".json"), decisions[-1])
        pd.DataFrame(example_records).to_csv(shards / (row["example_id"] + ".csv.gz"),
                                            index=False, compression="gzip")
        atomic_json(output / "nonfrontal_progress.json", {"completed_examples": number + 1,
                    "total_examples": len(rows), "calibrations": len(fitted_by_calibration)})
        print("NONFRONTAL_SAVED", number + 1, len(rows), flush=True)
    scores = pd.DataFrame(records)
    if family == "cca_adaptation":
        from .nonfrontal_review import MIDDLE, TEMPORAL
        central = scores[scores.scope == "central_shared"]
        middle = central[central.channel.isin(MIDDLE)].assign(scope="middle_verified")
        temporal = central[central.channel.isin(TEMPORAL)].assign(scope="temporal_shared")
        if len(middle) + len(temporal) != len(central):
            raise ValueError("CCA central anatomy is not covered by verified subgroups")
        scores = pd.concat((scores, middle, temporal), ignore_index=True)
    scores.to_csv(output / "nonfrontal_channel_scores.csv.gz", index=False, compression="gzip")
    dirty = scores[scores.condition != "clean"]
    fields = list(PAPER_KEYS) + ["cc_deterioration"]
    grouped = dirty.groupby(["scope", "evaluated_family", "recipient", "condition", "input_snr_db"])[fields].agg(strict_mean)
    grouped.groupby(["scope", "evaluated_family", "recipient"]).agg(strict_mean).groupby(
        ["scope", "evaluated_family"]).agg(strict_mean).reset_index().to_csv(output / "nonfrontal_method_means.csv", index=False)
    grouped.groupby(["scope", "evaluated_family", "condition", "input_snr_db"]).agg(strict_mean).reset_index().to_csv(
        output / "nonfrontal_condition_means.csv", index=False)
    baseline_metric_checks = reconcile_baselines(scores[scores.scope.isin(("whole", "frontal", "posterior", "central_shared"))], old_scores)
    gates = labelled_gates(scores, config)
    atomic_json(output / "nonfrontal_controlled_gates.json", gates)
    write_jsonl(output / "nonfrontal_fit_records.jsonl", fit_records)
    write_jsonl(output / "nonfrontal_decisions.jsonl", decisions)
    if family == "cca_adaptation":
        pd.DataFrame(coverage_rows).to_csv(output / "cca_coverage_audit.csv", index=False)
        atomic_json(output / "cca_coupling_audit.json", {
            "legacy_reference_scaled_with_field": False,
            "verified_producer": "corpus.py: calibration/scoring injected fields scaled; reference arrays unscaled",
            "new_generator_applied": False, "fresh_qualification_allowed": False,
            "policy": "matched legacy engineering comparison; current user requested next approach; RC repair remains pending",
            "reference_ranks": [1, 2], "reference_predictors": "HEOG/VEOG x three lags",
            "author_code_parity": False, "paired_truth_access": "coverage/metrics only after predictions"})
    from .coverage_uncertainty import uncertainty_report
    uncertainty_scores = scores[scores.scope.isin(("whole", "posterior", "central_shared"))].rename(
        columns={"evaluated_family": "arm"})
    uncertainty_report(uncertainty_scores, output, comparators=(BASELINE, RAW_CONTEXT),
        candidates=tuple(method + ":both" for method in methods), repetitions=500,
        scopes=("whole", "posterior", "central_shared"))
    atomic_json(output / "nonfrontal_summary.json", {"profile": profile, "examples": len(rows),
        "recipients": len({row["recipient"] for row in rows}), "donors": len({row["donor"] for row in rows}),
        "calibrations": len(fitted_by_calibration), "arms": list(estimates),
        "frozen_frontal_parity_checks": parity_checks, "source_split_sha256": config["source_split_sha256"],
        "saved_fit_replay_checks": fit_replay_checks, "baseline_paper_metric_checks": baseline_metric_checks,
        "new_decompositions": {"mwf_controls": "calibration-only GEVD-MWF", "cca_adaptation": "calibration-only regularized CCA"}.get(family, "calibration-only ICA") + "; no VMD solver executed",
        "runtime_s": time.perf_counter() - started, "weights_trained": False,
        "autovmd_executed": False, "scoring_target_fit": False, "reserved_confirmation_opened": False,
        "qualification": False, "native_validation_required": True,
        "reference_amplitude_mismatch": True, "field_generator_bias": "regression-derived controlled field",
        "uncertainty": "supplemental recipient/donor Poisson approximation, 500 draws; not paper-native permutation"})
    plot_results(output)
