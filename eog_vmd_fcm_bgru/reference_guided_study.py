"""Kaggle development gate for EOG-guided modes and posterior ICA support."""
from itertools import product
from pathlib import Path
import json
import pickle
import time
import numpy as np
import pandas as pd
from scipy import signal
from threadpoolctl import threadpool_limits
from .campaign_report import unique_source
from .channel_regions import REGION_NAMES, region_ids
from .dataset_io import klados_arrays, osf_trials, trial_condition, annotated_score_slice
from .evaluation import paired_metrics, modification_metrics, ocular_proxies
from .experiment import finite_json
from .provenance import save_json, sha256_file
from .reference_guided import (ReferenceModeSelector, aligned_correlations,
    reference_projection, soft_correlation_gate, posterior_ica_indices, routed_ica_residual)
from .spatial_expert import ICAExpert
from .vmd_rolling import rolling_decompose

STARTS = [2000, 3024, 4048]
WINDOW = 1024
K, ALPHA = 3, 2000
THRESHOLDS = [0.2, 0.4, 0.6, 0.8]
STRENGTHS = [0.25, 0.5, 0.75, 1.0]


def decompose_rows(values, max_iterations=500):
    vectors, diagnostics = [], []
    for row in values:
        modes, _, detail = rolling_decompose(row, K, ALPHA, max_iterations=max_iterations)
        vectors.append(modes)
        diagnostics.append(detail)
    return np.stack(vectors), diagnostics


def ordered_references(trial):
    eyes = {name.strip().upper(): values for name, values in trial["eog"].items()}
    if "HEOG" not in eyes or "VEOG" not in eyes:
        raise ValueError("This declared experiment requires both verified HEOG and VEOG")
    return np.stack([eyes["HEOG"], eyes["VEOG"]])


def window_evidence(vectors, references, selectors):
    correlations = np.stack([aligned_correlations(row, references) for row in vectors])
    cluster_weights = {"correlation": np.ones(vectors.shape[:2])}
    for name, selector in selectors.items():
        cluster_weights[name] = np.stack([selector.evidence(row, references)["cluster_weight"] for row in vectors])
    projected = reference_projection(vectors.reshape(-1, vectors.shape[-1]), references).reshape(vectors.shape)
    return {"vectors": vectors, "projected": projected, "correlations": correlations,
            "cluster_weights": cluster_weights}


def artifact_from_evidence(evidence, selector, threshold, projection):
    correlations = evidence["correlations"]
    weights = soft_correlation_gate(correlations.reshape(-1, correlations.shape[-1]), threshold)
    weights = weights.reshape(correlations.shape[:2]) * evidence["cluster_weights"][selector]
    selected = evidence["projected"] if projection else evidence["vectors"]
    return np.sum(selected * weights[..., None], axis=1)


def klados_development(root, output, split, max_iterations=500):
    dirty, clean, eog = klados_arrays(root)
    training_modes, training_references, training_keys, diagnostics = [], [], [], []
    for record in split["train"]:
        segment = slice(STARTS[0], STARTS[0] + WINDOW)
        modes, details = decompose_rows(dirty[record, :, segment], max_iterations)
        for channel, (vectors, detail) in enumerate(zip(modes, details)):
            training_modes.append(vectors)
            training_references.append(eog[record, :, segment])
            training_keys.append((record, channel))
            diagnostics.append({"split": "train", "record": record, "channel": channel,
                                "input": "dirty", "start": STARTS[0], **detail})
        print("Reference-guided training modes", record, flush=True)
    selectors = {f"fcm_{clusters}": ReferenceModeSelector.fit(training_modes, training_references, clusters=clusters)
                 for clusters in [2, 3]}
    with (output / "reference_selectors.pkl").open("wb") as handle:
        pickle.dump(selectors, handle)
    save_json(output / "cluster_evidence.json", {
        name: {"clusters": len(selector.centers), "training_modes": selector.training_modes,
               "reference_names": selector.reference_names,
               "cluster_heog_veog_evidence": selector.cluster_reference_evidence.tolist(),
               "centers_standardized": selector.centers.tolist(),
               "features": ["centroid", "low_fraction", "relative_energy", "kurtosis",
                            "spectral_entropy", "temporal_concentration", "abs_heog_corr", "abs_veog_corr"]}
        for name, selector in selectors.items()})
    np.savez_compressed(output / "training_mode_cache.npz", modes=np.stack(training_modes),
        references=np.stack(training_references), keys=np.asarray(training_keys))
    rows = []
    settings = list(product(["correlation", "fcm_2", "fcm_3"], [False, True], THRESHOLDS, STRENGTHS))
    for record in split["val"]:
        for start in STARTS:
            segment = slice(start, start + WINDOW)
            raw, target, references = dirty[record, :, segment], clean[record, :, segment], eog[record, :, segment]
            raw_vectors, raw_detail = decompose_rows(raw, max_iterations)
            clean_vectors, clean_detail = decompose_rows(target, max_iterations)
            for kind, details in [("dirty", raw_detail), ("clean", clean_detail)]:
                diagnostics.extend({"split": "val", "record": record, "channel": channel,
                                    "input": kind, "start": start, **detail}
                                   for channel, detail in enumerate(details))
            raw_evidence = window_evidence(raw_vectors, references, selectors)
            clean_evidence = window_evidence(clean_vectors, references, selectors)
            baseline = paired_metrics(raw.astype(np.float64), target.astype(np.float64))
            converged = not any(detail["hit_iteration_limit"] for detail in raw_detail + clean_detail)
            raw_artifacts, clean_artifacts = {}, {}
            for selector, projection, threshold, strength in settings:
                key = (selector, projection, threshold)
                if key not in raw_artifacts:
                    raw_artifacts[key] = artifact_from_evidence(raw_evidence, selector, threshold, projection)
                    clean_artifacts[key] = artifact_from_evidence(clean_evidence, selector, threshold, projection)
                corrected = raw.astype(np.float64) - strength * raw_artifacts[key]
                clean_corrected = target.astype(np.float64) - strength * clean_artifacts[key]
                quality = paired_metrics(corrected, target.astype(np.float64))
                preservation = paired_metrics(clean_corrected, target.astype(np.float64))
                change = modification_metrics(clean_corrected, target.astype(np.float64))
                rows.append({"record": record, "start": start, "selector": selector,
                    "projection": projection, "threshold": threshold, "strength": strength,
                    **quality, "raw_rmse": baseline["rmse"], "raw_snr_db": baseline["snr_db"],
                    "rmse_improvement_fraction": 1 - quality["rmse"] / baseline["rmse"],
                    "clean_relative_change": change["relative_change"],
                    "clean_alpha_error_db": preservation["alpha_error_db"],
                    "clean_beta_error_db": preservation["beta_error_db"],
                    "convergence_pass": converged})
            print("Reference-guided validation saved", record, start, flush=True)
        pd.DataFrame(rows).to_csv(output / "reference_validation_windows.csv", index=False)
        save_json(output / "reference_vmd_diagnostics.json", finite_json(diagnostics))
    columns = ["selector", "projection", "threshold", "strength"]
    frame = pd.DataFrame(rows)
    record_means = frame.groupby(columns + ["record"], as_index=False).mean(numeric_only=True)
    record_means.to_csv(output / "reference_validation_records.csv", index=False)
    summary = record_means.groupby(columns, as_index=False).mean(numeric_only=True)
    worst = record_means.groupby(columns).clean_relative_change.max().reset_index(name="worst_record_clean_change")
    summary = summary.merge(worst, on=columns)
    all_training_converged = not any(detail["hit_iteration_limit"] for detail in diagnostics if detail["split"] == "train")
    summary["feasible"] = ((summary.worst_record_clean_change <= 0.01)
        & (summary.clean_alpha_error_db <= 0.5) & (summary.clean_beta_error_db <= 0.5)
        & (summary.convergence_pass == 1) & all_training_converged & (summary.rmse_improvement_fraction > 0))
    summary.to_csv(output / "reference_validation_summary.csv", index=False)
    feasible = summary[summary.feasible]
    selected = None if feasible.empty else feasible.sort_values(["rmse", "selector", "threshold"]).iloc[0].to_dict()
    # A single overall development winner is frozen before the OSF diagnostic.
    save_json(output / "reference_selected.json", finite_json({"selected": selected,
        "K": K, "alpha": ALPHA, "rms_normalized": True, "tolerance": 1e-6,
        "max_iterations": max_iterations,
        "candidate_count": len(settings), "training_record_count": len(split["train"]),
        "validation_records": split["val"], "validation_channels": dirty.shape[1],
        "starts": STARTS, "all_training_decompositions_converged": all_training_converged,
        "heldout_test_scored": False, "requires_runtime_heog_veog": True,
        "anatomical_claim_supported_on_klados": False, "full_validation": False, "victory": False}))
    return selectors, selected


def fit_routed_ica(calibration, fit_indices):
    if any(trial["names"] != calibration[0]["names"] for trial in calibration):
        raise ValueError("ICA calibration channels change between independent trials")
    raw = np.concatenate([trial["eeg"][fit_indices] for trial in calibration], axis=1)
    eyes = np.concatenate([ordered_references(trial) for trial in calibration], axis=1)
    sos = signal.butter(4, 1, fs=200, btype="highpass", output="sos")
    filtered = np.concatenate([signal.sosfiltfilt(sos, trial["eeg"][fit_indices], axis=-1)
                               for trial in calibration], axis=1)
    attempts = []
    for method in ["picard", "infomax"]:
        try:
            expert = ICAExpert.fit(raw, eyes, method, calibration_highpass=filtered)
            attempts.append({"method": method, "converged": True, "rank": expert.rank,
                             "iterations": int(expert.decomposition.n_iter_)})
            return expert, attempts
        except Exception as error:
            attempts.append({"method": method, "converged": False, "error": repr(error)})
    return None, attempts


def osf_support_diagnostic(root, output, selectors, selected, max_iterations=500):
    sessions = [sorted((root / "Dataset1_OSF" / study).glob("*_prep.set"))[0]
                for study in ["study01", "study02", "study03", "study04"]]
    rows, calibration_rows, failures, runtime_rows = [], [], [], []
    for path in sessions:
        trials = list(osf_trials(path))
        calibration = trials[:5]
        if len(calibration) < 5:
            raise RuntimeError("Insufficient independent calibration trials")
        names = calibration[0]["names"]
        posterior_only, targets, _ = posterior_ica_indices(names, 0)
        supported, _, support = posterior_ica_indices(names, 4)
        fits = {"posterior_only_ica": posterior_only, "frontal_supported_ica": supported,
                "whole_montage_ica_posterior_output": list(range(len(names)))}
        states = {}
        for variant, fit_indices in fits.items():
            started = time.perf_counter()
            expert, attempts = fit_routed_ica(calibration, fit_indices)
            states[variant] = expert
            calibration_rows.append({"session": path.stem, "variant": variant,
                "fit_electrodes": [names[index] for index in fit_indices],
                "corrected_electrodes": [names[index] for index in targets],
                "support_electrodes": [names[index] for index in support] if variant == "frontal_supported_ica" else [],
                "unscored_trial_ids": [trial["trial"] for trial in calibration],
                "attempts": attempts, "fit_runtime_s": time.perf_counter() - started,
                "converged": expert is not None})
            save_json(output / "posterior_ica_calibration.json", finite_json(calibration_rows))
            print("Posterior support ICA fit", path.stem, variant, expert is not None, flush=True)
        seen = set()
        for trial in trials[5:]:
            kind = trial_condition(trial)
            segment = annotated_score_slice(trial, calibration=0)
            if kind is None or kind in seen or segment is None:
                continue
            if trial["names"] != names:
                failures.append({"session": path.stem, "trial": trial["trial"],
                                 "reason": "valid montage differs from calibration"})
                continue
            seen.add(kind)
            raw, eyes = trial["eeg"][:, segment], ordered_references(trial)[:, segment]
            variants = {"raw": raw.copy()}
            for variant, fit_indices in fits.items():
                if states[variant] is None:
                    failures.append({"session": path.stem, "condition": kind,
                                     "method": variant, "reason": "ICA calibration failed"})
                    continue
                started = time.perf_counter()
                residual = routed_ica_residual(raw, fit_indices, targets, states[variant])
                variants[variant] = raw - residual
                runtime_rows.append({"session": path.stem, "condition": kind,
                    "method": variant, "correction_runtime_s": time.perf_counter() - started})
            frontal = np.flatnonzero(region_ids(names) == 0)
            frontal_residual = np.zeros_like(raw)
            if selected is not None and len(frontal):
                started = time.perf_counter()
                vectors, details = decompose_rows(raw[frontal], max_iterations)
                evidence = window_evidence(vectors, eyes, selectors)
                candidate = artifact_from_evidence(evidence, selected["selector"],
                    selected["threshold"], selected["projection"]) * selected["strength"]
                for position, detail in enumerate(details):
                    if detail["hit_iteration_limit"]:
                        candidate[position] = 0
                        failures.append({"session": path.stem, "condition": kind,
                            "method": "frontal_vmd", "electrode": names[frontal[position]],
                            "reason": "VMD iteration limit; explicit channel passthrough"})
                frontal_residual[frontal] = candidate
                variants["frontal_vmd_only"] = raw - frontal_residual
                runtime_rows.append({"session": path.stem, "condition": kind,
                    "method": "frontal_vmd_only", "correction_runtime_s": time.perf_counter() - started})
                if "frontal_supported_ica" in variants:
                    # Disjoint target rows: posterior correction and frontal correction once each.
                    variants["regional_vmd_supported_ica"] = variants["frontal_supported_ica"] - frontal_residual
            for method, corrected in variants.items():
                groups = region_ids(names)
                for region, region_name in enumerate(REGION_NAMES):
                    keep = groups == region
                    if keep.any():
                        rows.append({"study": trial["study"], "participant": trial["participant"],
                            "session": trial["session"], "trial": trial["trial"], "condition": kind,
                            "method": method, "region": region_name, "channels": int(keep.sum()),
                            **ocular_proxies(raw[keep], corrected[keep], {"HEOG": eyes[0], "VEOG": eyes[1]},
                                             trial["labels"][segment])})
            pd.DataFrame(rows).to_csv(output / "posterior_support_region_proxies.csv", index=False)
            pd.DataFrame(runtime_rows).to_csv(output / "posterior_support_runtimes.csv", index=False)
            save_json(output / "posterior_support_exclusions.json", failures)
            print("Posterior support comparison", path.stem, kind, flush=True)
            if len(seen) == 4:
                break
    frame = pd.DataFrame(rows)
    frame.groupby(["method", "region", "condition"], as_index=False).mean(numeric_only=True).to_csv(
        output / "posterior_support_diagnostic_summary.csv", index=False)
    return {"sessions": [path.stem for path in sessions], "scope": "four development sessions, one per study",
        "rows": len(rows), "ica_calibrations_requested": len(calibration_rows),
        "ica_calibrations_converged": sum(row["converged"] for row in calibration_rows),
        "threshold": 0.8, "strength": 0.25, "support_count": 4,
        "osf_used_for_parameter_selection": False, "osf_snr_available": False,
        "winner_selected_from_osf": False, "all_session_validation": False}


def run_reference_guided(root, output, max_iterations=500):
    gate_path = unique_source("vmd_engine_gate.json")
    gate = json.loads(gate_path.read_text())
    if not gate["parity_pass"]:
        raise RuntimeError("VMD rolling parity gate failed")
    split_path = unique_source("record_split.json")
    split = json.loads(split_path.read_text())
    save_json(output / "reference_guided_protocol.json", {
        "split": split, "heldout_test_used": False, "reference_order": ["HEOG", "VEOG"],
        "runtime_references_required": True, "K": K, "alpha": ALPHA,
        "max_iterations": max_iterations, "stopping_tolerance": 1e-6,
        "reference_ridge_penalty": 0.01, "correlation_thresholds": THRESHOLDS,
        "correction_strengths": STRENGTHS, "window_starts": STARTS, "seed": 42,
        "source_ledger": [{"path": str(path), "sha256": sha256_file(path)} for path in [gate_path, split_path]],
        "numerical_execution": "Kaggle only", "full_validation": False})
    with threadpool_limits(limits=2):
        selectors, selected = klados_development(root, output, split, max_iterations)
        diagnostic = osf_support_diagnostic(root, output, selectors, selected, max_iterations)
    save_json(output / "reference_guided_summary.json", finite_json({
        "development_vmd_selected": selected, "posterior_support": diagnostic,
        "heldout_quality_measured": False, "complete_removal_established": False,
        "neural_deployment_implemented": False, "victory": False}))
