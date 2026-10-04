"""Development-only unit/convergence audit, independent of neural test output."""
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from .dataset_io import klados_arrays
from .evaluation import paired_metrics, modification_metrics
from .experiment import CALIBRATION, WINDOW, finite_json
from .provenance import save_json
from .vmd_expert import ModeExpert, decompose


def check_vmd_convergence(root, output, full_grid=False):
    sources = list(Path("/kaggle/input").rglob("selected_vmd.json"))
    if len(sources) != 1:
        raise RuntimeError("Attach one completed VMD development search")
    source = sources[0].parent
    chosen = json.loads(sources[0].read_text())
    split = json.loads((source / "record_split.json").read_text())
    dirty, clean, eog = klados_arrays(root)
    segment = slice(CALIBRATION, CALIBRATION + WINDOW)
    channels = [0, 9]
    configurations = ([{"K": modes, "alpha": alpha, "normalized": True, "tolerance": 1e-6}
                       for modes in range(3, 11) for alpha in [250, 500, 1000, 2000, 4000]] if full_grid else
                      [{"K": chosen["K"], "alpha": chosen["alpha"], "normalized": normalized, "tolerance": tolerance}
                       for normalized in [False, True] for tolerance in [1e-7, 1e-6, 1e-5]])
    fit_records = split["train"][:2]
    fit_eeg = [dirty[record, channel, segment] for record in fit_records for channel in channels]
    fit_eog = [eog[record, :, segment] for record in fit_records for channel in channels]
    save_json(output / "convergence_protocol.json", {"K": chosen["K"], "alpha": chosen["alpha"],
        "fit_records": fit_records, "validation_records": split["val"], "training_diagnostic_records": split["train"][:6],
        "channels": channels, "start_sample": CALIBRATION, "samples": WINDOW,
        "test_records_used": [], "rms_normalized_options": sorted({item["normalized"] for item in configurations}),
        "tolerances": sorted({item["tolerance"] for item in configurations}),
        "preservation_bound": "development mean clean modification <=1%, alpha/beta <=0.5 dB",
        "configurations": configurations,
        "scope": "expanded normalized K/alpha development search" if full_grid else "numerical VMD development audit; no hidden update of the running neural teacher"})
    rows, diagnostics, scalings, experts = [], [], [], {}
    for configuration in configurations:
        modes, alpha = configuration["K"], configuration["alpha"]
        normalized, tolerance = configuration["normalized"], configuration["tolerance"]
        tag = f"K-{modes}-alpha-{alpha}-rms-{normalized}-tol-{tolerance}"
        expert = ModeExpert.fit(fit_eeg, fit_eog, modes, alpha,
                                relative_tolerance=normalized, tolerance=tolerance)
        experts[tag] = expert
        for partition, records in [("train", split["train"][:6]), ("val", split["val"])]:
            for record in records:
                raw, target = dirty[record, channels, segment], clean[record, channels, segment]
                residual, detail = expert.artifact(raw)
                clean_residual, clean_detail = expert.artifact(target)
                limits = np.mean([entry["hit_iteration_limit"] for entry in detail + clean_detail])
                diagnostics.extend({"configuration": tag, "partition": partition, "record": record,
                                    "channel": channels[index], "input": kind, **entry}
                                   for kind, entries in [("dirty", detail), ("clean", clean_detail)]
                                   for index, entry in enumerate(entries))
                if partition == "val":
                    baseline = paired_metrics(raw, target)["rmse"]
                    for strength in [0.25, 0.5, 0.75, 1.0]:
                        corrected, corrected_clean = raw - strength * residual, target - strength * clean_residual
                        quality = paired_metrics(corrected, target)
                        preservation = paired_metrics(corrected_clean, target)
                        rows.append({"configuration": tag, "record": record, "strength": strength, "K": modes, "alpha": alpha,
                            "rmse": quality["rmse"], "raw_rmse": baseline,
                            "clean_relative_change": modification_metrics(corrected_clean, target)["relative_change"],
                            "clean_alpha_error_db": preservation["alpha_error_db"], "clean_beta_error_db": preservation["beta_error_db"],
                            "iteration_limit_fraction": float(limits)})
        # A solver whose threshold depends on amplitude can stop too early
        # after changing units. Restore the scale before comparing vectors.
        values = fit_eeg[0]
        reference, _, reference_detail = decompose(values, modes, alpha,
                                                    relative_tolerance=normalized, tolerance=tolerance)
        for amplitude in [1e-6, 1e6]:
            candidate, _, candidate_detail = decompose(values * amplitude, modes, alpha,
                                                       relative_tolerance=normalized, tolerance=tolerance)
            error = np.linalg.norm(candidate / amplitude - reference) / np.linalg.norm(reference)
            scalings.append({"configuration": tag, "input_scale": amplitude, "restored_vector_relative_difference": float(error),
                "reference_iterations": reference_detail["iterations"], "scaled_iterations": candidate_detail["iterations"],
                "maximum_center_shift_hz": float(np.max(np.abs(np.asarray(candidate_detail["centers_hz"]) - reference_detail["centers_hz"])))})
        pd.DataFrame(rows).to_csv(output / "convergence_validation_rows.csv", index=False)
        pd.DataFrame(scalings).to_csv(output / "vmd_amplitude_scaling.csv", index=False)
        save_json(output / "convergence_diagnostics.json", finite_json(diagnostics))
        print("VMD convergence setting completed", tag, flush=True)
    summary = pd.DataFrame(rows).drop(columns=["record"]).groupby(["configuration", "strength"], as_index=False).mean(numeric_only=True)
    summary["rmse_improvement_fraction"] = 1 - summary.rmse / summary.raw_rmse
    summary["feasible"] = ((summary.clean_relative_change <= 0.01) & (summary.clean_alpha_error_db <= 0.5)
                            & (summary.clean_beta_error_db <= 0.5) & (summary.iteration_limit_fraction == 0)
                            & (summary.rmse_improvement_fraction > 0))
    summary.to_csv(output / "convergence_grid_summary.csv", index=False)
    feasible = summary[summary.feasible].sort_values("rmse")
    selection = feasible.iloc[0].to_dict() if len(feasible) else None
    if selection is not None:
        with (output / "development_vmd_expert.pkl").open("wb") as handle:
            pickle.dump(experts[selection["configuration"]], handle)
    save_json(output / "convergence_check_summary.json", finite_json({"selected_development_candidate": selection,
        "candidate_count": len(summary), "heldout_test_scored": False, "full_validation": False, "victory": False}))
