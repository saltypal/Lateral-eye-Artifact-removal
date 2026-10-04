"""Diagnose full-montage ICA convergence with input-only Picard fallback."""
import json
from pathlib import Path
import pickle
import pandas as pd
from .dataset_io import osf_trials, trial_condition, annotated_score_slice, reference_matrix
from .spatial_expert import session_spatial_calibration
from .experiment import frozen_methods, finite_json
from .evaluation import ocular_proxies
from .channel_regions import region_ids, REGION_NAMES
from .provenance import save_json


def check_calibration(root, output):
    gates = list(Path("/kaggle/input").rglob("classical_gate.json"))
    if len(gates) != 1:
        raise RuntimeError("Attach exactly one completed development benchmark")
    source = gates[0].parent
    config = json.loads((source / "selected_spatial.json").read_text())
    if "ica" not in config:
        raise RuntimeError("No preservation-feasible Klados ICA configuration to diagnose")
    preferred = config["ica"].split(":")[0]
    config["ica_fallback"] = "picard" if preferred != "picard" else "infomax"
    with (source / "vmd_expert.pkl").open("rb") as handle:
        expert = pickle.load(handle)
    vmd_config = json.loads((source / "selected_vmd.json").read_text())
    sessions = [sorted((root / "Dataset1_OSF" / study).glob("*_prep.set"))[0]
                for study in ["study01", "study02", "study03", "study04"]]
    states, proxies, failures = [], [], []
    for path in sessions:
        trials = list(osf_trials(path))
        state = session_spatial_calibration(trials[:5], config)
        states.append({"session": path.stem, "channels": len(state["names"]),
                       "unscored_trial_ids": state["trial_ids"], "errors": state["errors"],
                       "ica_method": state.get("ica_method"), "ica_attempts": state.get("ica_attempts", []),
                       "rank": state["ica"].rank if state["ica"] is not None else None})
        save_json(output / "calibration_diagnostics.json", finite_json(states))
        seen = set()
        for trial in trials[5:]:
            kind = trial_condition(trial)
            segment = annotated_score_slice(trial, calibration=0)
            if kind is None or kind in seen or segment is None:
                continue
            seen.add(kind)
            refs, _ = reference_matrix(trial)
            groups = region_ids(trial["names"])
            for method, corrected, metadata in frozen_methods(trial["eeg"], refs, trial["names"],
                                                               expert, vmd_config, config, segment.start, state):
                if corrected is None:
                    failures.append({"session": path.stem, "trial": trial["trial"], "method": method, **metadata})
                    continue
                for region, region_name in enumerate(REGION_NAMES):
                    keep = groups == region
                    if keep.any():
                        eyes = {name: values[segment] for name, values in trial["eog"].items()}
                        proxies.append({"study": trial["study"], "participant": trial["participant"],
                                        "session": trial["session"], "trial": trial["trial"], "condition": kind,
                                        "method": method, "region": region_name, "channels": int(keep.sum()),
                                        "ica_fit_method": state.get("ica_method"),
                                        **ocular_proxies(trial["eeg"][keep, segment], corrected[keep, segment],
                                                         eyes, trial["labels"][segment])})
            pd.DataFrame(proxies).to_csv(output / "calibration_region_proxies.csv", index=False)
            save_json(output / "calibration_exclusions.json", failures)
            print("Calibration comparison saved", path.stem, kind, flush=True)
            if len(seen) == 4:
                break
    save_json(output / "calibration_check_summary.json", {"scope": "one session per audited study; development diagnostic",
              "preferred_ica": preferred, "input_only_fallback": config["ica_fallback"],
              "calibration_trials": 5, "sessions_requested": [path.stem for path in sessions],
              "converged_sessions": sum(state["ica_method"] is not None for state in states),
              "selection": "convergence only; no OSF labels/clean target tune any threshold",
              "benchmark_git_sha": json.loads((source / "run_config.json").read_text())["git_sha"], "victory": False})
