"""Bounded classical feasibility study. Full-study victory is a separate gate."""
from itertools import product
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
import pandas as pd
from .channel_regions import REGION_NAMES, region_ids, fuse_residuals
from .dataset_io import klados_arrays, split_records, osf_trials
from .evaluation import paired_metrics, modification_metrics, ocular_proxies
from .provenance import save_json
from .spatial_expert import ICAExpert, ridge_residual, asr_correction, fit_asr, armbr_correction
from .vmd_expert import ModeExpert
from .research_plots import grid_figures, example_modes

K_GRID = list(range(3, 11))
ALPHA_GRID = [250, 500, 1000, 2000, 4000]
STRENGTH_GRID = [0.25, 0.5, 0.75, 1.0]
CALIBRATION = 2000
WINDOW = 1024


def csv(rows, destination):
    pd.DataFrame(rows).to_csv(destination, index=False)


def finite_json(value):
    """Nonfinite measurements are unavailable, never fabricated zero scores."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def vmd_search(dirty, clean, eog, split, output):
    # Coverage is explicit and identical across all 40 configurations. This
    # bounded engineering search does not replace five-fold final validation.
    training_records = split["train"][:2]
    validation_records = split["val"][:2]
    channels = sorted(set([0, dirty.shape[1] // 2]))
    segment = slice(CALIBRATION, CALIBRATION + WINDOW)
    training = [dirty[r, c, segment] for r in training_records for c in channels]
    references = [eog[r, :, segment] for r in training_records for _ in channels]
    save_json(output / "grid_coverage.json", {"train_records": training_records,
              "validation_records": validation_records, "channels": channels,
              "samples": [CALIBRATION, CALIBRATION + WINDOW], "K": K_GRID,
              "alpha": ALPHA_GRID, "strength": STRENGTH_GRID, "test_used": False})
    rows, centers, failures = [], [], []
    for modes, alpha in product(K_GRID, ALPHA_GRID):
        started = time.perf_counter()
        try:
            expert = ModeExpert.fit(training, references, modes, alpha)
            for record in validation_records:
                raw = dirty[record, channels, segment]
                target = clean[record, channels, segment]
                artifact, diagnostics = expert.artifact(raw)
                clean_artifact, _ = expert.artifact(target)
                baseline = paired_metrics(raw, target)
                for channel, detail in zip(channels, diagnostics):
                    centers.append({"record": record, "channel": channel, "K": modes,
                                    "alpha": alpha, **detail})
                for strength in STRENGTH_GRID:
                    corrected = raw - strength * artifact
                    clean_corrected = target - strength * clean_artifact
                    quality = paired_metrics(corrected, target)
                    preservation = paired_metrics(clean_corrected, target)
                    change = modification_metrics(clean_corrected, target)
                    rows.append({"record": record, "K": modes, "alpha": alpha,
                                 "strength": strength, **quality,
                                 "raw_rmse": baseline["rmse"],
                                 "rmse_improvement_fraction": 1 - quality["rmse"] / baseline["rmse"],
                                 "clean_relative_change": change["relative_change"],
                                 "clean_alpha_error_db": preservation["alpha_error_db"],
                                 "clean_beta_error_db": preservation["beta_error_db"],
                                 "iteration_limit_fraction": float(np.mean([d["hit_iteration_limit"] for d in diagnostics])),
                                 "minimum_center_spacing_hz": float(np.mean([d["nearest_center_hz"] for d in diagnostics])),
                                 "mean_adjacent_psd_overlap": float(np.mean([np.mean(d["adjacent_psd_overlap"]) for d in diagnostics])),
                                 "mean_decomposition_residual_ratio": float(np.mean([d["residual_ratio"] for d in diagnostics])),
                                 "runtime_s": time.perf_counter() - started})
        except Exception as error:
            failures.append({"K": modes, "alpha": alpha, "error": repr(error),
                             "traceback": traceback.format_exc()})
        csv(rows, output / "vmd_grid_records.csv")
        save_json(output / "vmd_centers.json", finite_json(centers))
        save_json(output / "vmd_grid_failures.json", failures)
        print("VMD grid", modes, alpha, "saved", len(rows), "rows", flush=True)
    if not rows:
        raise RuntimeError("All VMD candidates failed; inspect vmd_grid_failures.json")
    aggregate = pd.DataFrame(rows).groupby(["K", "alpha", "strength"], as_index=False).mean(numeric_only=True)
    # These are predeclared feasibility guardrails, not a noninferiority test.
    aggregate["preservation_guard_pass"] = ((aggregate.clean_relative_change <= 0.20)
                                            & (aggregate.clean_alpha_error_db <= 1.0)
                                            & (aggregate.clean_beta_error_db <= 1.0))
    aggregate["convergence_guard_pass"] = aggregate.iteration_limit_fraction == 0
    aggregate.to_csv(output / "vmd_grid_summary.csv", index=False)
    grid_figures(output)
    feasible = aggregate[aggregate.preservation_guard_pass & aggregate.convergence_guard_pass & (aggregate.rmse_improvement_fraction > 0)]
    if feasible.empty:
        save_json(output / "classical_gate.json", {"vmd_feasible": False,
                  "reason": "No development candidate improved RMSE within clean-preservation guardrails",
                  "victory": False})
        return None
    selected = feasible.sort_values(["rmse", "K", "alpha", "strength"]).iloc[0]
    configuration = {"K": int(selected.K), "alpha": int(selected.alpha),
                     "strength": float(selected.strength), "validation_records": validation_records,
                     "selection": "minimum development RMSE subject to predeclared preservation guardrails",
                     "full_validation": False}
    save_json(output / "selected_vmd.json", configuration)
    expert = ModeExpert.fit(training, references, configuration["K"], configuration["alpha"])
    with (output / "vmd_expert.pkl").open("wb") as handle:
        pickle.dump(expert, handle)
    from .vmd_expert import decompose
    raw = dirty[validation_records[0], channels[0], segment]
    target = clean[validation_records[0], channels[0], segment]
    vectors, residual, detail = decompose(raw, configuration["K"], configuration["alpha"])
    np.savez_compressed(output / "vmd_example.npz", raw=raw, target=target, modes=vectors, residual=residual)
    save_json(output / "vmd_example_metadata.json", {"record": validation_records[0], "channel": channels[0],
              "fs": 200, "sample_interval": [CALIBRATION, CALIBRATION + WINDOW], **detail})
    example_modes(raw, target, vectors, residual, detail["centers_hz"], output)
    return expert, configuration


def spatial_search(dirty, clean, eog, split, output):
    rows, failures = [], []
    segment = slice(CALIBRATION, CALIBRATION + WINDOW)
    def preservation(corrected_clean, target):
        change = modification_metrics(corrected_clean[:, segment], target[:, segment])
        quality = paired_metrics(corrected_clean[:, segment], target[:, segment])
        return {"clean_relative_change": change["relative_change"], "clean_alpha_error_db": quality["alpha_error_db"],
                "clean_beta_error_db": quality["beta_error_db"], "clean_covariance_error": quality["covariance_error"]}
    for record in split["val"][:2]:
        raw, target = dirty[record], clean[record]
        for penalty in [0.1, 1.0, 10.0]:
            residual = ridge_residual(raw, eog[record], CALIBRATION, penalty)
            rows.append({"method": "ridge_eog", "parameter": str(penalty), "record": record,
                         **paired_metrics((raw - residual)[:, segment], target[:, segment]),
                         **preservation(target - residual, target)})
        for method in ["picard", "infomax"]:
            try:
                expert = ICAExpert.fit(raw[:, :CALIBRATION], eog[record, :, :CALIBRATION], method)
                for threshold in [0.2, 0.3, 0.4]:
                    residual = expert.residual(raw, threshold)
                    clean_residual = expert.residual(target, threshold)
                    rows.append({"method": "ica", "parameter": f"{method}:{threshold}", "record": record,
                                 **paired_metrics((raw - residual)[:, segment], target[:, segment]),
                                 **preservation(target - clean_residual, target)})
            except Exception as error:
                failures.append({"method": method, "record": record, "error": repr(error)})
        for cutoff in [10, 20, 30]:
            try:
                estimator = fit_asr(raw, CALIBRATION, cutoff)
                corrected = np.asarray(estimator.transform(raw))
                clean_corrected = np.asarray(estimator.transform(target))
                rows.append({"method": "asr", "parameter": str(cutoff), "record": record,
                             **paired_metrics(corrected[:, segment], target[:, segment]),
                             **preservation(clean_corrected, target)})
            except Exception as error:
                failures.append({"method": "asr", "cutoff": cutoff, "record": record, "error": repr(error)})
        csv(rows, output / "spatial_grid_records.csv")
        save_json(output / "spatial_failures.json", failures)
    summary = pd.DataFrame(rows).groupby(["method", "parameter"], as_index=False).mean(numeric_only=True)
    summary["preservation_guard_pass"] = ((summary.clean_relative_change <= 0.20)
                                          & (summary.clean_alpha_error_db <= 1)
                                          & (summary.clean_beta_error_db <= 1))
    summary.to_csv(output / "spatial_grid_summary.csv", index=False)
    selected = {}
    for method, group in summary.groupby("method"):
        if method == "ridge_eog":
            # EOG-reference baseline remains explicit even when the clean-EEG
            # counterfactual with the original EOG violates preservation.
            selected[method] = str(group.sort_values("rmse").iloc[0]["parameter"])
        else:
            feasible = group[group.preservation_guard_pass]
            if not feasible.empty:
                selected[method] = str(feasible.sort_values("rmse").iloc[0]["parameter"])
    save_json(output / "selected_spatial.json", selected)
    return selected


def frozen_methods(raw, references, names, expert, vmd_config, spatial_config):
    """Both hybrid residuals refer to the same raw signal; ICA is global."""
    start = time.perf_counter()
    yield "raw", raw, {"runtime_s": 0}
    artifact = np.zeros_like(raw)
    if expert is not None:
        estimate, _ = expert.artifact(raw[:, CALIBRATION:CALIBRATION + WINDOW], strength=vmd_config["strength"])
        artifact[:, CALIBRATION:CALIBRATION + WINDOW] = estimate
        yield "vmd_fcm", raw - artifact, {"runtime_s": time.perf_counter() - start}
    if "asr" in spatial_config:
        try:
            yield "asr", asr_correction(raw, CALIBRATION, int(spatial_config["asr"])), {"eog_required": False}
        except Exception as error:
            yield "asr_unavailable", None, {"reason": repr(error)}
    try:
        armbr, metadata = armbr_correction(raw, names, CALIBRATION)
        yield "armbr", armbr, metadata
        if expert is not None:
            yield "vmd_armbr_hybrid", fuse_residuals(raw, artifact, raw - armbr, names, 0.8, 0.2, 0.5), metadata
    except Exception as error:
        yield "armbr_unavailable", None, {"reason": repr(error)}
    if references.shape[0] == 0:
        return
    penalty = float(spatial_config["ridge_eog"])
    yield "ridge_eog", raw - ridge_residual(raw, references, CALIBRATION, penalty), {"eog_required": True}
    if "ica" in spatial_config:
        try:
            method, threshold = spatial_config["ica"].split(":")
            ica = ICAExpert.fit(raw[:, :CALIBRATION], references[:, :CALIBRATION], method)
            residual = ica.residual(raw, float(threshold))
            yield "ica", raw - residual, {"rank": ica.rank, "eog_required": True}
            if expert is None:
                return
            # VMD is evaluated on an explicitly bounded segment; no trial join.
            refined_segment = ica.residual(raw[:, CALIBRATION:CALIBRATION + WINDOW], float(threshold), True,
                                           vmd_config["K"], vmd_config["alpha"])
            refined = np.zeros_like(raw)
            refined[:, CALIBRATION:CALIBRATION + WINDOW] = refined_segment
            yield "ica_source_vmd", raw - refined, {"rank": ica.rank, "eog_required": True}
            yield "shared_hybrid", fuse_residuals(raw, artifact, refined, names, 0.5, 0.5, 0.5), {}
            yield "regional_hybrid", fuse_residuals(raw, artifact, refined, names, 0.8, 0.2, 0.5), {
                "frontal_vmd_weight": 0.8, "posterior_vmd_weight": 0.2,
                "weight_source": "fixed hypothesis; unknown Klados anatomy prevents paired regional tuning"}
        except Exception as error:
            yield "ica_unavailable", None, {"reason": repr(error)}


def benchmark(root: Path, output: Path, profile="kaggle_smoke"):
    if profile != "kaggle_smoke":
        raise ValueError("Full five-fold study is a separate explicit protocol")
    dirty, clean, references = klados_arrays(root)
    split = split_records(clean)
    save_json(output / "record_split.json", split)
    outcome = vmd_search(dirty, clean, references, split, output)
    spatial = spatial_search(dirty, clean, references, split, output)
    if outcome is None:
        print("VMD feasibility gate failed; evaluating spatial alternatives without a VMD teacher", flush=True)
        expert, config = None, None
    else:
        expert, config = outcome
    rows, exclusions = [], []
    scoring = slice(CALIBRATION, CALIBRATION + WINDOW)
    for record in split["test"][:2]:
        names = ["unknown"] * dirty.shape[1]
        for method, corrected, metadata in frozen_methods(dirty[record], references[record], names, expert, config, spatial):
            if corrected is None:
                exclusions.append({"record": record, "method": method, **metadata})
                continue
            rows.append({"dataset": "klados", "record": record, "method": method,
                         **paired_metrics(corrected[:, scoring], clean[record, :, scoring])})
        csv(rows, output / "benchmark_records.csv")
    # Protocol Z: frozen Klados choices. Select first horizontal and blink trial
    # from at most two sessions for explicit smoke coverage; no OSF tuning.
    sessions = sorted((root / "Dataset1_OSF").rglob("*_prep.set"))[:2]
    osf_rows = []
    for path in sessions:
        seen = set()
        for trial in osf_trials(path):
            labels = trial["labels"]
            if labels is None or trial["eeg"].shape[-1] < CALIBRATION + WINDOW:
                continue
            present = "horizontal" if np.isin(labels, [1, 2]).any() else "blink" if (labels == 5).any() else None
            if present is None or present in seen:
                continue
            seen.add(present)
            refs = np.stack(list(trial["eog"].values())) if trial["eog"] else np.empty((0, trial["eeg"].shape[-1]))
            for method, corrected, metadata in frozen_methods(trial["eeg"], refs, trial["names"], expert, config, spatial):
                if corrected is None:
                    exclusions.append({"session": path.stem, "trial": trial["trial"], "method": method, **metadata})
                    continue
                groups = region_ids(trial["names"])
                for region, region_name in enumerate(REGION_NAMES):
                    members = groups == region
                    if not members.any():
                        continue
                    raw = trial["eeg"][members, scoring]
                    cleaned = corrected[members, scoring]
                    eog = {name: value[scoring] for name, value in trial["eog"].items()}
                    osf_rows.append({"study": trial["study"], "participant": trial["participant"],
                                     "session": trial["session"], "trial": trial["trial"], "coverage_type": present,
                                     "method": method, "region": region_name, "channels": int(members.sum()),
                                     **ocular_proxies(raw, cleaned, eog, labels[scoring])})
            csv(osf_rows, output / "osf_region_proxies.csv")
            if len(seen) == 2:
                break
    save_json(output / "benchmark_exclusions.json", exclusions)
    pd.DataFrame(rows).groupby("method").mean(numeric_only=True).to_csv(output / "benchmark_summary.csv")
    save_json(output / "classical_gate.json", {"vmd_feasible": outcome is not None, "profile": profile,
              "paired_only_training_allowed": True,
              "test_records_scored": split["test"][:2], "osf_sessions_requested": [p.stem for p in sessions],
              "claim": "bounded feasibility only; no complete-removal or superiority claim", "victory": False})
    print("Classical smoke complete; full validation remains required", flush=True)
