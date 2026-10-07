"""Validation-only selection of conservative deployment gating; Kaggle only."""
import json
import time
import numpy as np
import pandas as pd
import torch
from .dataset_io import klados_arrays
from .experiment import finite_json
from .provenance import save_json, sha256_file
from .snr_student import EOGContextGainStudent
from .snr_target import (unique_input, load_verified_cache, prediction, validation_score,
    spectral_preservation, checkpoint, evaluate, transfer_diagnostic, STRENGTHS)


def run_snr_guard(root, output, device):
    torch.set_num_threads(2)
    source = unique_input("snr_target_protocol.json").parent
    protocol = json.loads((source / "snr_target_protocol.json").read_text())
    if "eog_vmd_context_student" not in protocol["model_inputs"]:
        raise ValueError("Attach the completed shared-support experiment")
    source_checkpoint = source / "eog_vmd_context_student" / "best_safe.pt"
    state = torch.load(source_checkpoint, map_location=device, weights_only=True)
    split = protocol["split"]
    raw, clean, eyes = klados_arrays(root)
    validation, _ = load_verified_cache(source, "val", raw, clean, eyes, split["val"], output)
    name = "eog_vmd_context_student"
    folder = output / name
    folder.mkdir(exist_ok=True)
    network = EOGContextGainStudent().to(device).eval()
    network.load_state_dict(state["model"])
    protocol["guard_refinement"] = {"rules": [["soft", 0.0], ["hard", 0.0], ["hard", 0.025], ["hard", 0.05]],
        "validation_clean_change_margin": 0.0025, "reported_test_margin": 0.01,
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_git_sha": json.loads((source / "run_config.json").read_text())["git_sha"],
        "training": "no retraining; preserve the validation-selected neural weights",
        "reason": "soft gate met SNR target but reused-test worst-record preservation exceeded 1%; test is development-exposed",
        "rule_selection": "validation only; no test-derived threshold or strength"}
    protocol["device"] = device
    save_json(output / "snr_guard_protocol.json", protocol)
    rows, selected = [], {}
    best_safe, best_free = -float("inf"), -float("inf")
    for rule, offset in protocol["guard_refinement"]["rules"]:
        network.inference_gate_mode, network.inference_gate_offset = rule, offset
        artifact, identity = prediction(network, validation, device)
        for strength in STRENGTHS:
            metrics = validation_score(validation, artifact, identity, split["val"], strength)
            preservation = spectral_preservation(validation[1], identity, strength)
            row = {**metrics, **preservation, "strength": strength, "inference_gate_mode": rule,
                   "inference_gate_offset": offset, "epoch": state["selection"]["epoch"],
                   "identity_weight": state["selection"]["identity_weight"],
                   "learning_rate": state["selection"]["learning_rate"]}
            feasible = metrics["clean_relative_change_worst_record"] <= 0.0025
            feasible = feasible and preservation["clean_alpha_error_db"] <= 0.5 and preservation["clean_beta_error_db"] <= 0.5
            rows.append({**row, "preservation_feasible": feasible})
            if metrics["validation_snr_db"] > best_free:
                best_free = metrics["validation_snr_db"]
                selected["unconstrained"] = row
                checkpoint(network, row, protocol, folder / "best_unconstrained.pt")
            if feasible and metrics["validation_snr_db"] > best_safe:
                best_safe = metrics["validation_snr_db"]
                selected["safe"] = row
                checkpoint(network, row, protocol, folder / "best_safe.pt")
    if "safe" not in selected:
        raise RuntimeError("No finite guarded candidate, including identity")
    pd.DataFrame(rows).to_csv(folder / "validation_grid.csv", index=False)
    save_json(folder / "selected.json", finite_json(selected))
    print("SNR GUARD validation", json.dumps(selected), flush=True)
    test, keys = load_verified_cache(source, "test", raw, clean, eyes, split["test"], output)
    rows = evaluate(name, test, keys, output, device)
    # Original raw/VMD baseline predictions were frozen before this refinement.
    baselines = pd.read_csv(source / "test_window_metrics.csv")
    baselines = baselines[baselines.method.isin(["raw", "frozen_reference_vmd"])]
    frame = pd.concat([pd.DataFrame(rows), baselines], ignore_index=True)
    frame.to_csv(output / "test_window_metrics.csv", index=False)
    records = frame.drop(columns=["start"]).groupby(["method", "record"], as_index=False).mean(numeric_only=True)
    records.to_csv(output / "test_record_metrics.csv", index=False)
    summary = records.drop(columns=["record"]).groupby("method", as_index=False).mean(numeric_only=True)
    summary["worst_record_clean_change"] = summary.method.map(records.groupby("method").clean_relative_change.max())
    summary["heldout_preservation_pass"] = ((summary.worst_record_clean_change <= 0.01)
        & (summary.clean_alpha_error_db <= 0.5) & (summary.clean_beta_error_db <= 0.5))
    summary.loc[summary.method.isin(["raw", "frozen_reference_vmd"]), "heldout_preservation_pass"] = None
    summary.to_csv(output / "test_summary.csv", index=False)
    target = summary[summary.method == name + "_safe"].iloc[0]
    result = {"selections": selected, "test": summary.replace({np.nan: None}).to_dict(orient="records"),
        "target_15db_met_with_preservation": bool(target.snr_db >= 15 and target.heldout_preservation_pass),
        "parameters": sum(p.numel() for p in network.parameters()), "device": device,
        "snr_definition": "record macro of paired amplitude-sensitive window SNR; all19 rows,8records,3windows",
        "fresh_test": False, "test_informed_guard_motivation": True,
        "individual_record_15db_guaranteed": False, "complete_removal": False, "full_validation": False, "victory": False}
    save_json(output / "snr_guard_summary.json", finite_json(result))
    print("FINAL SNR GUARD", summary[["method", "snr_db", "clean_relative_change", "worst_record_clean_change"]].to_string(index=False), flush=True)
    transfer_diagnostic(root, output, device, [name])
    from .snr_target_plots import plot_snr_campaign
    plot_snr_campaign(output, [name])
