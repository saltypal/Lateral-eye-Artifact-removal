"""Validation-constrained EEG-only search; all numeric work runs on Kaggle."""
import json
import pickle
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from .dataset_io import klados_arrays
from .evaluation import paired_metrics, modification_metrics
from .experiment import CALIBRATION, WINDOW, finite_json
from .provenance import save_json
from .spatial_expert import ICAExpert
from .student import SharedChannelStudent, reconstruction_loss, paired_gate_loss
from .training import seed_everything, evaluate_osf_student, latency_scaling


STARTS = [CALIBRATION + index * WINDOW for index in range(3)]
STRENGTHS = [0.0, 0.1, 0.25, 0.5, 1.0]


def window_arrays(dirty, clean, records):
    keys = [(record, start) for record in records for start in STARTS]
    inputs = torch.from_numpy(np.stack([dirty[record, :, start:start + WINDOW] for record, start in keys]))
    targets = torch.from_numpy(np.stack([clean[record, :, start:start + WINDOW] for record, start in keys]))
    return inputs, targets, keys


def training_teachers(dirty, clean, eog, records, benchmark, output):
    """Fixed first six training records, first window; no validation/test truth."""
    settings = json.loads((benchmark / "selected_vmd.json").read_text())
    spatial_settings = json.loads((benchmark / "selected_spatial.json").read_text())
    with (benchmark / "vmd_expert.pkl").open("rb") as handle:
        expert = pickle.load(handle)
    teachers, diagnostics = {}, []
    segment = slice(STARTS[0], STARTS[0] + WINDOW)
    for record in records[:6]:
        raw, target = dirty[record, :, segment], clean[record, :, segment]
        residual, details = expert.artifact(raw, strength=settings["strength"])
        clean_residual, clean_details = expert.artifact(target, strength=settings["strength"])
        converged = not any(row["hit_iteration_limit"] for row in details + clean_details)
        error = None
        if "ica" in spatial_settings:
            method, threshold, strength = spatial_settings["ica"].split(":")
            try:
                spatial = ICAExpert.fit(dirty[record, :, :CALIBRATION], eog[record, :, :CALIBRATION], method)
                residual = 0.5 * residual + 0.5 * float(strength) * spatial.residual(raw, float(threshold), True, settings["K"], settings["alpha"])
                clean_residual = 0.5 * clean_residual + 0.5 * float(strength) * spatial.residual(target, float(threshold), True, settings["K"], settings["alpha"])
            except Exception as exception:
                error = repr(exception)
        preservation = paired_metrics(target - clean_residual, target)
        change = modification_metrics(target - clean_residual, target)["relative_change"]
        improved = paired_metrics(raw - residual, target)["rmse"] < paired_metrics(raw, target)["rmse"]
        accepted = improved and converged and change <= 0.20 and preservation["alpha_error_db"] <= 1 and preservation["beta_error_db"] <= 1
        if accepted:
            teachers[(record, STARTS[0])] = residual
        diagnostics.append({"record": record, "start_sample": STARTS[0], "accepted": bool(accepted),
                            "clean_relative_change": change, "convergence_pass": converged, "ica_error": error})
        save_json(output / "search_teacher_diagnostics.json", finite_json(diagnostics))
        print("Search teacher", record, "accepted", accepted, flush=True)
    return teachers


def predict_residuals(network, values, device, include_gates=False):
    artifacts, gates = [], []
    network.eval()
    with torch.no_grad():
        for batch in values.split(8):
            inputs = batch.to(device)
            mask = torch.ones(inputs.shape[:2], device=device)
            regions = torch.full(inputs.shape[:2], 3, dtype=torch.long, device=device)
            result = network(inputs, mask, regions)
            artifacts.append(result["artifact"].cpu())
            if include_gates:
                gates.append(result["gate"].cpu())
    return torch.cat(artifacts), torch.cat(gates) if include_gates else None


def record_scores(error, clean_artifact, clean, records):
    windows = len(STARTS)
    record_rmse = error.square().mean(dim=(1, 2)).sqrt().reshape(len(records), windows).mean(dim=1)
    clean_ratio = (clean_artifact.square().sum(dim=(1, 2)).sqrt() /
                   clean.square().sum(dim=(1, 2)).sqrt().clamp_min(1e-6)).reshape(len(records), windows).mean(dim=1)
    return float(record_rmse.mean()), float(clean_ratio.mean()), float(clean_ratio.max())


def search_student(root, output, device):
    sources = list(Path("/kaggle/input").rglob("classical_gate.json"))
    if len(sources) != 1:
        raise RuntimeError("Attach one frozen classical benchmark")
    benchmark = sources[0].parent
    classical_gate = json.loads(sources[0].read_text())
    if not classical_gate.get("vmd_feasible"):
        raise RuntimeError("This hybrid search requires the verified VMD development teacher")
    split = json.loads((benchmark / "record_split.json").read_text())
    dirty, clean, eog = klados_arrays(root)
    if max(STARTS) + WINDOW > dirty.shape[-1]:
        raise ValueError("Declared scoring windows exceed the source recording")
    torch.set_num_threads(2)
    arrays = {part: window_arrays(dirty, clean, split[part]) for part in ["train", "val", "test"]}
    x, y, keys = arrays["train"]
    teachers = training_teachers(dirty, clean, eog, split["train"], benchmark, output)
    teacher_values = torch.stack([torch.from_numpy(teachers[key]) if key in teachers else torch.zeros_like(x[index])
                                 for index, key in enumerate(keys)])
    teacher_enabled = torch.tensor([key in teachers for key in keys])
    vx, vy, _ = arrays["val"]
    raw_validation, _, _ = record_scores(vx - vy, torch.zeros_like(vy), vy, split["val"])
    configurations = [{"identity_weight": identity, "learning_rate": rate, "gate_weight": 0.05,
                       "distillation_weight": 0.10, "seed": 42, "epochs": 60}
                      for identity in [0.25, 1.0, 4.0, 16.0] for rate in [0.001, 0.0003]]
    save_json(output / "search_protocol.json", {"configurations": configurations, "strengths": STRENGTHS,
              "split": split, "starts": STARTS, "samples_per_window": WINDOW,
              "selection": "minimum record-macro validation RMSE among <=1% worst-record mean clean modification",
              "earlier_smoke_test_records": [5, 14], "gate_supervision": "paired -10/-20 dB burden; gray ignored",
              "teacher_scope": "first six training records, first scoring window only",
              "subject_independence": "unverified; record groups only", "full_contract_validation": False})
    best = raw_validation + 1e-9
    selected, history, candidate_rows = None, [], []
    started = time.perf_counter()
    for config_id, config in enumerate(configurations):
        seed_everything(config["seed"])
        network = SharedChannelStudent().to(device)
        # Begin at identity. Supervised gate can learn while residual heads grow.
        for head in [network.temporal_residual, network.context_residual]:
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
        torch.nn.init.constant_(network.gate.bias, -2.0)
        optimizer = torch.optim.AdamW(network.parameters(), lr=config["learning_rate"], weight_decay=0.0001)
        loader = DataLoader(TensorDataset(x, y, teacher_values, teacher_enabled), batch_size=8, shuffle=True,
                            generator=torch.Generator().manual_seed(config["seed"]), num_workers=0)
        for epoch in range(config["epochs"]):
            network.train()
            losses = []
            for values, targets, privileged, enabled in loader:
                values, targets, privileged, enabled = [item.to(device) for item in (values, targets, privileged, enabled)]
                mask = (torch.rand(values.shape[:2], device=device) > 0.15).float()
                mask[:, 0] = 1
                regions = torch.full(values.shape[:2], 3, dtype=torch.long, device=device)
                dirty_scale = values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
                clean_scale = targets.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
                dirty_result, clean_result = network(values, mask, regions), network(targets, mask, regions)
                loss = reconstruction_loss(dirty_result["cleaned"] / dirty_scale, targets / dirty_scale, mask)
                loss = loss + config["identity_weight"] * reconstruction_loss(clean_result["cleaned"] / clean_scale, targets / clean_scale, mask)
                gate_dirty, _, _ = paired_gate_loss(dirty_result["gate"], values, targets, mask)
                gate_clean, _, _ = paired_gate_loss(clean_result["gate"], targets, targets, mask)
                loss = loss + config["gate_weight"] * (gate_dirty + gate_clean)
                if enabled.any():
                    loss = loss + config["distillation_weight"] * reconstruction_loss(
                        dirty_result["artifact"][enabled] / dirty_scale[enabled], privileged[enabled] / dirty_scale[enabled], mask[enabled])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite search loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1)
                optimizer.step()
                losses.append(float(loss.detach()))
            val_artifact, _ = predict_residuals(network, vx, device)
            clean_artifact, _ = predict_residuals(network, vy, device)
            for strength in STRENGTHS:
                rmse, change, worst_change = record_scores(vx - strength * val_artifact - vy,
                                                         strength * clean_artifact, vy, split["val"])
                feasible = worst_change <= 0.01
                row = {"config_id": config_id, "epoch": epoch + 1, "strength": strength, "validation_rmse": rmse,
                       "clean_relative_change_mean": change, "clean_relative_change_worst_record": worst_change,
                       "preservation_feasible": feasible, "rmse_improvement_fraction": 1 - rmse / raw_validation, **config}
                candidate_rows.append(row)
                if feasible and rmse < best:
                    best, selected = rmse, row.copy()
                    state = {"model": {name: value.detach().cpu().clone() for name, value in network.state_dict().items()},
                             "selection": selected, "teacher_git_sha": json.loads((benchmark / "run_config.json").read_text())["git_sha"]}
                    torch.save(state, output / "selected_unscaled.pt")
                    save_json(output / "selected_neural.json", selected)
            history.append({"config_id": config_id, "epoch": epoch + 1, "loss": float(np.mean(losses)), "best_feasible_validation_rmse": best})
            pd.DataFrame(history).to_csv(output / "search_history.csv", index=False)
            if (epoch + 1) % 10 == 0:
                print("Neural search", config_id, "epoch", epoch + 1, "best feasible validation RMSE", best, flush=True)
        torch.save({"model": network.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                    "epoch": config["epochs"]}, output / f"search_{config_id}_last.pt")
        pd.DataFrame(candidate_rows).to_csv(output / "neural_grid.csv", index=False)
    if selected is None:
        raise RuntimeError("No finite validation candidate, including the identity control")
    network = SharedChannelStudent().to(device)
    state = torch.load(output / "selected_unscaled.pt", map_location=device, weights_only=True)
    network.load_state_dict(state["model"])
    with torch.no_grad():
        for head in [network.temporal_residual, network.context_residual]:
            head.weight.mul_(selected["strength"])
            head.bias.mul_(selected["strength"])
    torch.save({"model": network.state_dict(), "selection": selected}, output / "best.pt")
    summary = {"profile": "validation_constrained_8_configuration_search", "coverage": {key: split[key] for key in arrays},
               "starts": STARTS, "seed": 42, "parameters": sum(p.numel() for p in network.parameters()), "device": device,
               "input": "EEG only; no runtime EOG/VMD/ICA", "offline": True, "selected": selected,
               "runtime_s": time.perf_counter() - started, "full_validation": False, "victory": False,
               "regions": "unknown Klados mapping; OSF named priors are fixed transfer ablations"}
    save_json(output / "training_summary.json", summary)
    val_artifact, _ = predict_residuals(network, vx, device)
    val_clean_artifact, _ = predict_residuals(network, vy, device)
    validation_preservation = []
    for index, (record, start) in enumerate(arrays["val"][2]):
        target = vy[index].numpy()
        corrected_clean = target - val_clean_artifact[index].numpy()
        validation_preservation.append({"record": record, "start_sample": start,
            **modification_metrics(corrected_clean, target), **paired_metrics(corrected_clean, target)})
    pd.DataFrame(validation_preservation).to_csv(output / "validation_clean_metrics.csv", index=False)
    summary["validation_preservation_measured"] = True
    summary["validation_alpha_error_mean_db"] = float(np.mean([row["alpha_error_db"] for row in validation_preservation]))
    summary["validation_beta_error_mean_db"] = float(np.mean([row["beta_error_db"] for row in validation_preservation]))
    summary["development_guard_pass"] = bool(selected["clean_relative_change_worst_record"] <= 0.01
        and summary["validation_alpha_error_mean_db"] <= 0.5 and summary["validation_beta_error_mean_db"] <= 0.5)
    summary["development_removal_target_pass"] = bool(selected["rmse_improvement_fraction"] >= 0.10)
    save_json(output / "training_summary.json", summary)
    evaluate_selected(network, arrays["test"], split["test"], output, device)
    evaluate_osf_student(network, root, output, device, session_limit=None)
    latency_scaling(network, output, device)
    from .export_model import export_and_verify
    tx, _, _ = arrays["test"]
    artifact, _ = predict_residuals(network, tx, device)
    export_and_verify(network, output, tx.numpy(), (tx - artifact).numpy())
    from .research_plots import neural_search_figures
    neural_search_figures(output)


def evaluate_selected(network, test_arrays, records, output, device):
    inputs, targets, keys = test_arrays
    artifact, gates = predict_residuals(network, inputs, device, include_gates=True)
    clean_artifact, _ = predict_residuals(network, targets, device)
    rows = []
    for index, (record, start) in enumerate(keys):
        raw, target = inputs[index].numpy(), targets[index].numpy()
        prediction = raw - artifact[index].numpy()
        clean_output = target - clean_artifact[index].numpy()
        for name, estimate in [("raw", raw), ("eeg_only_student", prediction)]:
            row = {"record": record, "start_sample": start, "method": name,
                   "earlier_smoke_record": record in {5, 14}, **paired_metrics(estimate, target)}
            if name == "eeg_only_student":
                row.update({"clean_" + key: value for key, value in modification_metrics(clean_output, target).items()})
                row.update({"clean_" + key: value for key, value in paired_metrics(clean_output, target).items()})
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "student_metrics.csv", index=False)
    record_frame = frame.drop(columns=["start_sample"]).groupby(["record", "method", "earlier_smoke_record"], as_index=False).mean(numeric_only=True)
    record_frame.to_csv(output / "student_record_metrics.csv", index=False)
    quality_bounds(record_frame, output)
    np.savez_compressed(output / "student_test_predictions.npz", record_ids=np.repeat(records, len(STARTS)), starts=[start for _, start in keys],
                        predictions=(inputs - artifact).numpy(), raw=inputs.numpy(), target=targets.numpy())
    _, labels, valid = paired_gate_loss(gates, inputs, targets, torch.ones(inputs.shape[:2]))
    from sklearn.metrics import average_precision_score
    probability, truth = gates[valid].numpy(), labels[valid].numpy()
    two_classes = len(np.unique(truth)) == 2
    bins = np.minimum((probability * 10).astype(int), 9)
    ece = sum(np.mean(bins == index) * abs(float(probability[bins == index].mean()) - float(truth[bins == index].mean()))
              for index in range(10) if np.any(bins == index))
    save_json(output / "gate_diagnostics.json", {"valid_tokens": int(valid.sum()), "prevalence": float(truth.mean()),
              "auprc": float(average_precision_score(truth, probability)) if two_classes else None,
              "ece_10_equal_width_bins": ece, "definition": "paired ocular burden, not blink versus vertical saccade identity",
              "calibrated": False, "confidence_unit": "tokens correlated within record; no independent-token CI"})


def quality_bounds(records, output):
    """Record bootstrap for this engineering split; no subject/legacy claim."""
    fresh = records[~records.earlier_smoke_record]
    student = fresh[fresh.method == "eeg_only_student"].set_index("record")
    raw = fresh[fresh.method == "raw"].set_index("record").loc[student.index]
    metrics = {"rmse_reduction_fraction": 1 - student.rmse.to_numpy() / raw.rmse.to_numpy(),
               "snr_gain_db": student.snr_db.to_numpy() - raw.snr_db.to_numpy(),
               "clean_relative_change": student.clean_relative_change.to_numpy(),
               "clean_alpha_error_db": student.clean_alpha_error_db.to_numpy(),
               "clean_beta_error_db": student.clean_beta_error_db.to_numpy()}
    generator = np.random.default_rng(42004)
    samples = generator.integers(0, len(student), size=(2000, len(student)))
    rows = []
    for name, values in metrics.items():
        draws = values[samples].mean(axis=1)
        rows.append({"metric": name, "mean": float(values.mean()), "one_sided_95_lower": float(np.quantile(draws, 0.05)),
                     "one_sided_95_upper": float(np.quantile(draws, 0.95)), "records": len(student)})
    pd.DataFrame(rows).to_csv(output / "fresh_record_bootstrap.csv", index=False)
    save_json(output / "student_quality_gate.json", {"fresh_heldout_records": student.index.tolist(),
        "earlier_inspected_smoke_records_excluded_from_bounds": [5, 14], "bootstrap_replicates": 2000,
        "unit": "record, not participant; subject mapping unresolved", "comparison": "raw EEG, not matched old model",
        "full_five_fold_three_seed_gate": False, "adapted_osf_gate": False,
        "complete_removal_claim": False, "victory": False})
