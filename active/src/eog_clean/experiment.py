"""One recoverable Klados/OSF experiment, with explicit limits and no mixture code."""
import json
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from .data import (save_json, sha256, klados, preprocess, osf_trials,
                   osf_sessions, calibration_split)
from .frontal import estimate, combine
from .methods import fit_ridge, window_regression, fit_ica, fit_mwf
from .metrics import paired, correlations, preservation


CONDITIONS = {1: "rest", 2: "lateral", 3: "vertical", 4: "blink"}


def _safe_fit(name, function, failures, record):
    try:
        return function()
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        failures.append({"record": record, "method": name, "phase": "calibration",
                         "error": type(error).__name__+": "+str(error), "fallback": "identity"})
        return None


def _fit(values, eyes, ocular, rest, config, failures, record):
    return {
        "direct_calibrated": _safe_fit("direct_calibrated", lambda: fit_ridge(values, eyes, config["ridge"]), failures, record),
        "ica_component_regression": _safe_fit("ica_component_regression", lambda: fit_ica(values, eyes,
            config["ica_reference_threshold"], config["ridge"], config["seed"], config["ica_components_max"]), failures, record),
        "mwf": _safe_fit("mwf", lambda: fit_mwf(ocular, rest, config["mwf_rank"], config["mwf_strength"]), failures, record),
    }


def _artifacts(values, eyes, fits, config):
    result = {"identity": np.zeros_like(values), "direct_window": window_regression(values, eyes, config["ridge"])}
    for method, model in fits.items():
        result[method] = model.artifact(values, eyes) if model is not None else np.zeros_like(values)
    ica = fits["ica_component_regression"]
    result["ica_component_rejection"] = ica.artifact(values, eyes, whole_components=True) if ica is not None else np.zeros_like(values)
    return result


def _paired_rows(prediction, target, raw, eyes, record, start, method, channel_names):
    scores = paired(prediction, target)
    baseline = paired(raw, target)
    association = correlations(prediction, eyes)
    rows = []
    for channel, name in enumerate(channel_names):
        rows.append({"dataset": "Klados", "record": record, "start": start, "channel": name,
            "region": "unknown", "method": method, "mse": float(scores["mse"][channel]),
            "rmse": float(scores["rmse"][channel]), "rrmse_t": float(scores["rrmse_t"][channel]),
            "pearson_cc": float(scores["pearson_cc"][channel]), "snr_out_db": float(scores["snr_out_db"][channel]),
            "snr_improvement_db": float(scores["snr_out_db"][channel]-baseline["snr_out_db"][channel]),
            "pooled_window_snr_db": scores["pooled_window_snr_db"],
            "heog_cc": float(association[channel, 0]), "veog_cc": float(association[channel, 1])})
    return rows


def run_klados(root, output, config):
    dirty, clean, references, metadata = klados(root)
    save_json(output/"klados_sources.json", metadata)
    rows, preservation_rows, failures, timing = [], [], [], []
    channel_names = [f"row_{i:02d}" for i in range(dirty.shape[1])]
    limit = config.get("klados_record_limit", len(dirty))
    for record in range(min(limit, len(dirty))):
        directory = output/"klados"/f"record_{record:02d}"
        directory.mkdir(parents=True)
        began = time.perf_counter()
        stop = config["klados_calibration_samples"]
        # Separate filtering prevents zero-phase leakage across the fit/score boundary.
        calibration = preprocess(dirty[record, :, :stop])
        cal_eyes = preprocess(references[record, :, :stop])
        scoring = preprocess(dirty[record, :, stop:])
        target_scoring = preprocess(clean[record, :, stop:])
        score_eyes = preprocess(references[record, :, stop:])
        eye_scale = cal_eyes.std(axis=-1, keepdims=True)
        activity = np.mean((cal_eyes/np.maximum(eye_scale, 1e-12))**2, axis=0)
        lower, upper = np.quantile(activity, [.3, .7])
        fits = _fit(calibration, cal_eyes, calibration[:, activity >= upper],
                    calibration[:, activity <= lower], config, failures, record)
        save_json(directory/"calibration.json", {"samples": stop, "scoring_starts_at": stop,
            "paired_target_used_for_fitting": False,
            "mwf_labels": "EOG activity bottom/top 30% within calibration; approximate rest/ocular masks",
            "anatomy": "unknown; no regional VMD computed", "scoring_condition_labels": "unavailable"})
        record_rows, predictions, controls = [], {}, {}
        for start in config["klados_starts"]:
            local = start-stop
            segment = slice(local, local+config["window"])
            raw, target, eyes = scoring[:, segment], target_scoring[:, segment], score_eyes[:, segment]
            if raw.shape[-1] != config["window"]:
                raise ValueError("Klados scoring window exceeds source record")
            artifacts = _artifacts(raw, eyes, fits, config)
            clean_artifacts = _artifacts(target, eyes, fits, config)
            for method, artifact in artifacts.items():
                result = raw-artifact
                predictions[f"{method}_{start}"] = result.astype(np.float32)
                controls[f"{method}_{start}"] = (target-clean_artifacts[method]).astype(np.float32)
                record_rows.extend(_paired_rows(result, target, raw, eyes, record, start, method, channel_names))
                preservation_rows.append({"record": record, "start": start, "method": method,
                    **preservation(target-clean_artifacts[method], target)})
            predictions[f"target_{start}"] = target
            predictions[f"raw_{start}"] = raw
            predictions[f"eyes_{start}"] = eyes
        pd.DataFrame(record_rows).to_csv(directory/"paired_metrics.csv", index=False)
        np.savez_compressed(directory/"predictions.npz", **predictions)
        np.savez_compressed(directory/"clean_input_predictions.npz", **controls)
        save_json(directory/"completion.json", {"complete": True, "record": record, "source_preprocessing": metadata["additional_filter"]})
        rows.extend(record_rows)
        timing.append({"dataset": "Klados", "record": record, "runtime_s": time.perf_counter()-began})
        pd.DataFrame(failures).to_csv(output/"calibration_failures.csv", index=False)
        print("KLADOS_RECORD_SAVED", record, "seconds", round(timing[-1]["runtime_s"], 2), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"klados_per_channel_window.csv", index=False)
    pd.DataFrame(preservation_rows).to_csv(output/"klados_clean_preservation.csv", index=False)
    records = frame.groupby(["method", "record"], as_index=False).mean(numeric_only=True)
    records.to_csv(output/"klados_per_record.csv", index=False)
    summary = records.groupby("method", as_index=False).mean(numeric_only=True)
    summary.to_csv(output/"klados_record_macro.csv", index=False)
    pd.DataFrame(timing).to_csv(output/"klados_runtime.csv", index=False)
    return summary, failures


def _regional_fits(calibration, names, groups, config, failures, record):
    fits = {}
    for group in ("frontal", "posterior", "central"):
        indices = np.flatnonzero(np.asarray(groups) == group)
        if not len(indices):
            continue
        values = np.concatenate([trial["eeg"][indices] for trial in calibration], axis=-1)
        eyes = np.concatenate([trial["eyes"] for trial in calibration], axis=-1)
        ocular = np.concatenate([trial["eeg"][indices] for trial in calibration if trial["condition"] in (2, 3, 4)], axis=-1)
        rest = np.concatenate([trial["eeg"][indices] for trial in calibration if trial["condition"] == 1], axis=-1)
        fits[group] = (indices, _fit(values, eyes, ocular, rest, config, failures, f"{record}:{group}"))
    return fits


def run_osf(root, output, config, partition):
    failures, rows, rest_rows, all_diagnostics, timing = [], [], [], [], []
    for path in osf_sessions(root, partition, config["osf_max_sessions"]):
        began = time.perf_counter()
        trials, metadata = osf_trials(path)
        record = metadata["record"]
        directory = output/"osf"/record
        directory.mkdir(parents=True)
        calibration, scoring, policy = calibration_split(trials)
        save_json(directory/"source_and_split.json", {**metadata, "calibration_policy": policy,
            "calibration_trials": [trial["index"] for trial in calibration],
            "scoring_trials": [trial["index"] for trial in scoring], "paired_clean_target": False})
        names, groups = metadata["channel_names"], np.asarray(metadata["regions"])
        fits = _regional_fits(calibration, names, groups, config, failures, record)
        frontal_indices = np.flatnonzero(groups == "frontal")
        counts = {condition: 0 for condition in CONDITIONS}
        for trial in scoring:
            condition = trial["condition"]
            if counts[condition] >= config["osf_max_scoring_trials_per_condition"]:
                continue
            counts[condition] += 1
            values, eyes, eval_eyes = trial["eeg"], trial["eyes"], trial["evaluation_eyes"]
            artifacts = {name: np.zeros_like(values) for name in (
                "identity", "direct_window", "direct_calibrated", "ica_component_regression", "ica_component_rejection", "mwf")}
            for group, (indices, group_fits) in fits.items():
                for name, artifact in _artifacts(values[indices], eyes, group_fits, config).items():
                    artifacts[name][indices] = artifact
            frontal_artifact = np.zeros_like(values)
            if len(frontal_indices):
                estimate_values, cache, diagnostics = estimate(values[frontal_indices], eyes, config,
                    [names[index] for index in frontal_indices])
                frontal_artifact[frontal_indices] = estimate_values
                np.savez_compressed(directory/f"trial_{trial['index']:03d}_frontal_modes.npz", **cache)
                for detail in diagnostics:
                    all_diagnostics.append({"record": record, "trial": trial["index"],
                        "condition": CONDITIONS[condition], **detail})
            artifacts["frontal_vmd_only"] = frontal_artifact
            for name in ("direct_calibrated", "ica_component_regression", "mwf"):
                _, artifact = combine(values, frontal_artifact, artifacts[name], groups)
                artifacts["hybrid_vmd_"+name] = artifact
            predictions = {"raw": values, "eyes": eyes, "evaluation_eyes": eval_eyes}
            for method, artifact in artifacts.items():
                cleaned = values-artifact
                predictions[method] = cleaned.astype(np.float32)
                association = correlations(cleaned, eval_eyes)
                raw_association = correlations(values, eval_eyes)
                for channel, name in enumerate(names):
                    rows.append({"dataset": "OSF", "record": record, "participant": metadata["participant"],
                        "trial": trial["index"], "condition": CONDITIONS[condition], "method": method,
                        "channel": name, "region": groups[channel], "heog_cc": association[channel, 0],
                        "veog_cc": association[channel, 1], "absolute_heog_cc": abs(association[channel, 0]),
                        "absolute_veog_cc": abs(association[channel, 1]),
                        "raw_absolute_heog_cc": abs(raw_association[channel, 0]),
                        "raw_absolute_veog_cc": abs(raw_association[channel, 1])})
                if condition == 1:
                    for group in ("frontal", "posterior", "central"):
                        keep = groups == group
                        if keep.any():
                            rest_rows.append({"record": record, "participant": metadata["participant"],
                                "trial": trial["index"], "method": method, "region": group,
                                **preservation(cleaned[keep], values[keep])})
            np.savez_compressed(directory/f"trial_{trial['index']:03d}_predictions.npz", **predictions)
            pd.DataFrame(all_diagnostics).to_csv(output/"osf_mode_reference_evidence.csv", index=False)
            pd.DataFrame(rows).to_csv(output/"osf_trial_channel_correlations.csv", index=False)
            pd.DataFrame(rest_rows).to_csv(output/"osf_rest_preservation.csv", index=False)
            print("OSF_TRIAL_SAVED", record, trial["index"], CONDITIONS[condition], flush=True)
        save_json(directory/"completion.json", {"complete": True, "record": record,
            "scored_trials": counts, "regional_anatomy_verified_from_labels": True})
        timing.append({"record": record, "runtime_s": time.perf_counter()-began})
        print("OSF_RECORD_SAVED", record, flush=True)
    pd.DataFrame(failures).to_csv(output/"osf_calibration_failures.csv", index=False)
    pd.DataFrame(timing).to_csv(output/"osf_runtime.csv", index=False)
    frame = pd.DataFrame(rows)
    if len(frame):
        # First within a participant/condition/region, then participants equally.
        participants = frame.groupby(["method", "participant", "condition", "region"], as_index=False).mean(numeric_only=True)
        participants.to_csv(output/"osf_participant_summary.csv", index=False)
        participants.groupby(["method", "condition", "region"], as_index=False).mean(numeric_only=True).to_csv(
            output/"osf_participant_macro.csv", index=False)
    return failures


def replay_historical(input_root, output, config):
    """Frozen archived-cache inference, clearly separate from the new benchmark."""
    import torch
    from .historical_model import EOGContextGainStudent
    parents = list(Path(input_root).rglob("snr_target_protocol.json"))
    if len(parents) != 1:
        raise ValueError("Historical replay needs exactly one versioned context parent")
    parent = parents[0].parent
    checkpoint = parent/"eog_vmd_context_student/best_safe.pt"
    if sha256(checkpoint) != config["historical_checkpoint_sha256"]:
        raise ValueError("Historical checkpoint checksum differs")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = EOGContextGainStudent().eval()
    model.load_state_dict(state["model"])
    model.inference_gate_mode, model.inference_gate_offset = "hard", 0.
    with np.load(parent/"test_vmd_cache.npz", allow_pickle=False) as cache:
        arrays = {name: cache[name].copy() for name in ("raw", "clean", "eyes", "modes", "valid", "keys")}
    with torch.no_grad():
        raw = torch.from_numpy(arrays["raw"])
        result = model(raw, torch.from_numpy(arrays["modes"]), torch.from_numpy(arrays["eyes"]),
                       torch.ones(raw.shape[:2]), torch.full(raw.shape[:2], 3))
        prediction = (raw-result["artifact"]*torch.from_numpy(arrays["valid"])[..., None]).numpy()
    rows = []
    for index, (record, start) in enumerate(arrays["keys"]):
        scores = paired(prediction[index], arrays["clean"][index])
        rows.append({"record": int(record), "start": int(start), "method": "historical_eog_vmd_model",
            "snr_out_db": scores["pooled_window_snr_db"], "rmse": float(np.sqrt(scores["mse"].mean())),
            "pearson_cc": float(np.mean(scores["pearson_cc"]))})
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"historical_window_metrics.csv", index=False)
    record_scores = frame.groupby("record").mean(numeric_only=True)
    record_scores.to_csv(output/"historical_record_metrics.csv")
    mean_snr = float(record_scores.snr_out_db.mean())
    save_json(output/"historical_replay.json", {"measured_record_macro_snr_db": mean_snr,
        "previously_reported_snr_db": 17.927211250454842,
        "reproduced_within_0_001_db": abs(mean_snr-17.927211250454842) < .001,
        "records": len(record_scores), "windows": len(frame), "checkpoint_sha256": sha256(checkpoint),
        "cache_sha256": sha256(parent/"test_vmd_cache.npz"), "parameters": 2499,
        "runtime_eog_required": True, "runtime_vmd_required": True,
        "development_exposed": True, "verified_frontal_claim": False,
        "current_preprocessed_originals_claimed_equal_to_cache": False,
        "aggregation": "pooled-energy SNR per window; average windows within record, then records in dB"})
    np.savez_compressed(output/"historical_predictions.npz", prediction=prediction, **arrays)
    print("HISTORICAL_REPLAY_SAVED", mean_snr, flush=True)


def run(root, input_root, output, config, partition):
    if not Path("/kaggle/input").exists():
        raise RuntimeError("Numerical experiment execution is restricted to Kaggle")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/"execution_state.json", {"state": "running", "config": config})
    try:
        replay_historical(input_root, output, config)
        run_klados(root, output, config)
        run_osf(root, output, config, partition)
        save_json(output/"execution_state.json", {"state": "complete", "config": config,
            "research_qualified": False, "reason": "Candidate preservation and independent validation require review"})
    except Exception as error:
        save_json(output/"execution_state.json", {"state": "failed", "config": config,
            "error": type(error).__name__+": "+str(error), "traceback": traceback.format_exc()})
        raise
