"""Kaggle-only, validation-selected 15/20-dB neural development campaign."""
from pathlib import Path
import json
import shutil
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from .dataset_io import klados_arrays, osf_trials, annotated_score_slice, trial_condition
from .evaluation import paired_metrics, modification_metrics, ocular_proxies
from .experiment import finite_json
from .provenance import save_json, sha256_file
from .training import seed_everything
from .vmd_rolling import rolling_decompose
from .reference_guided_study import window_evidence, artifact_from_evidence, ordered_references
from .snr_student import (VMDSpatialStudent, EOGGainStudent, EOGContextGainStudent, masked_snr_db,
                          snr_target_loss, identity_penalty)

STARTS = [2000, 3024, 4048]
WINDOW = 1024
STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
MODELS = {"eeg_vmd_student": VMDSpatialStudent, "eog_vmd_gain_student": EOGGainStudent,
          "eog_vmd_context_student": EOGContextGainStudent}


def unique_input(name):
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Attach exactly one {name}; found {len(matches)}")
    return matches[0]


def construct_cache(dirty, clean, eyes, records, part, output, training_cache=None):
    """Modes depend on their own input only; failed rows receive zero correction."""
    keys = [(record, start) for record in records for start in STARTS]
    raw_values, clean_values, references = [], [], []
    mode_values, clean_modes, validity, clean_validity, diagnostics = [], [], [], [], []
    for record, start in keys:
        segment = slice(start, start + WINDOW)
        raw_values.append(dirty[record, :, segment])
        clean_values.append(clean[record, :, segment])
        references.append(eyes[record, :, segment])
        pair_modes, pair_valid = [], []
        for kind, values in [("dirty", dirty), ("clean", clean)]:
            modes, valid = [], []
            for channel, row in enumerate(values[record, :, segment]):
                cached = training_cache.get((record, channel)) if training_cache and kind == "dirty" and start == STARTS[0] else None
                if cached is not None:
                    vectors, detail = cached, {"hit_iteration_limit": False, "reused_verified_training_cache": True}
                else:
                    vectors, _, detail = rolling_decompose(row, 3, 2000, max_iterations=2000)
                good = not detail["hit_iteration_limit"]
                modes.append(vectors if good else np.zeros_like(vectors))
                valid.append(good)
                diagnostics.append({"part": part, "record": record, "start": start,
                                    "channel": channel, "input": kind, **detail})
            pair_modes.append(np.stack(modes))
            pair_valid.append(valid)
        mode_values.append(pair_modes[0])
        clean_modes.append(pair_modes[1])
        validity.append(pair_valid[0])
        clean_validity.append(pair_valid[1])
        print("SNR VMD cache", part, record, start, flush=True)
    arrays = [np.stack(values).astype(np.float32) for values in
              [raw_values, clean_values, references, mode_values, clean_modes, validity, clean_validity]]
    np.savez_compressed(output / f"{part}_vmd_cache.npz", keys=np.asarray(keys),
        **dict(zip(["raw", "clean", "eyes", "modes", "clean_modes", "valid", "clean_valid"], arrays)))
    save_json(output / f"{part}_vmd_diagnostics.json", finite_json(diagnostics))
    return [torch.from_numpy(values) for values in arrays], keys


def prediction(network, arrays, device):
    raw, clean, eyes, modes, clean_modes, valid, clean_valid = arrays
    artifacts, identity_artifacts = [], []
    network.eval()
    with torch.no_grad():
        for begin in range(0, len(raw), 8):
            stop = begin + 8
            values, targets, references, vectors, target_vectors, keep, target_keep = [
                value[begin:stop].to(device) for value in arrays]
            mask = torch.ones(values.shape[:2], device=device)
            regions = torch.full(values.shape[:2], 3, dtype=torch.long, device=device)
            artifact = network(values, vectors, references, mask, regions)["artifact"] * keep[..., None]
            identity = network(targets, target_vectors, references, mask, regions)["artifact"] * target_keep[..., None]
            artifacts.append(artifact.cpu())
            identity_artifacts.append(identity.cpu())
    return torch.cat(artifacts), torch.cat(identity_artifacts)


def validation_score(arrays, artifact, identity_artifact, records, strength):
    raw, clean = arrays[:2]
    mask = torch.ones(raw.shape[:2])
    snr = masked_snr_db(raw - strength * artifact, clean, mask).reshape(len(records), len(STARTS)).mean(dim=1)
    relative = ((strength * identity_artifact).square().sum(dim=(1, 2)) /
                clean.square().sum(dim=(1, 2))).sqrt().reshape(len(records), len(STARTS)).mean(dim=1)
    return {"validation_snr_db": float(snr.mean()), "clean_relative_change_mean": float(relative.mean()),
            "clean_relative_change_worst_record": float(relative.max())}


def spectral_preservation(clean, identity_artifact, strength):
    measurements = [paired_metrics((target - strength * change).numpy(), target.numpy())
                    for target, change in zip(clean, identity_artifact)]
    return {"clean_alpha_error_db": float(np.mean([row["alpha_error_db"] for row in measurements])),
            "clean_beta_error_db": float(np.mean([row["beta_error_db"] for row in measurements]))}


def checkpoint(network, selection, protocol, path):
    torch.save({"model": {name: value.detach().cpu().clone() for name, value in network.state_dict().items()},
                "selection": selection, "protocol": protocol}, path)


def train_model(name, train, validation, split, output, device, protocol, teacher, teacher_enabled):
    folder = output / name
    folder.mkdir(exist_ok=True)
    configurations = [{"identity_weight": identity, "learning_rate": rate, "snr_weight": 1.0,
        "target_db": 20.0, "epochs": 160, "seed": 42}
        for identity in [0.02, 0.2] for rate in [0.001, 0.0003]]
    best_safe, best_free = -float("inf"), -float("inf")
    history, candidates, selected = [], [], {}
    for config_id, config in enumerate(configurations):
        seed_everything(config["seed"])
        network = MODELS[name]().to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-5)
        loader = DataLoader(TensorDataset(*train, teacher, teacher_enabled), batch_size=8, shuffle=True,
            generator=torch.Generator().manual_seed(config["seed"]), num_workers=0)
        started = time.perf_counter()
        for epoch in range(config["epochs"]):
            network.train()
            losses = []
            for batch in loader:
                values, targets, eyes, vectors, target_vectors, valid, clean_valid, privileged, enabled = [
                    item.to(device) for item in batch]
                mask = (torch.rand(values.shape[:2], device=device) > 0.15).float()
                mask[:, 0] = 1
                regions = torch.full(values.shape[:2], 3, dtype=torch.long, device=device)
                residual = network(values, vectors, eyes, mask, regions)["artifact"] * valid[..., None]
                clean_residual = network(targets, target_vectors, eyes, mask, regions)["artifact"] * clean_valid[..., None]
                predicted = values - residual
                loss_snr = snr_target_loss(predicted, targets, mask, config["target_db"])
                identity = identity_penalty(clean_residual, targets, mask)
                target_scale = targets.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
                weights = mask[..., None]
                denominator = (weights.sum() * values.shape[-1]).clamp_min(1)
                mse = (((predicted - targets) / target_scale).square() * weights).sum() / denominator
                derivative = torch.diff(predicted / target_scale, dim=-1) - torch.diff(targets / target_scale, dim=-1)
                derivative_loss = (derivative.square() * weights).sum() / denominator
                loss = config["snr_weight"] * loss_snr + config["identity_weight"] * identity + 0.1 * mse + 0.05 * derivative_loss
                if enabled.any():
                    distillation = (((residual[enabled] - privileged[enabled]) / target_scale[enabled]).square() * weights[enabled]).sum()
                    loss = loss + 0.02 * distillation / (weights[enabled].sum() * values.shape[-1]).clamp_min(1)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite SNR optimization")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            history.append({"config_id": config_id, "epoch": epoch + 1, "loss": float(np.mean(losses)),
                "runtime_s": time.perf_counter() - started, **config})
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config["epochs"]:
                artifact, identity_artifact = prediction(network, validation, device)
                for strength in STRENGTHS:
                    metrics = validation_score(validation, artifact, identity_artifact, split["val"], strength)
                    row = {"config_id": config_id, "epoch": epoch + 1, "strength": strength, **config, **metrics}
                    preservation = None
                    if metrics["validation_snr_db"] > best_free:
                        best_free = metrics["validation_snr_db"]
                        preservation = spectral_preservation(validation[1], identity_artifact, strength)
                        selected["unconstrained"] = {**row, **preservation, "preservation_required": False}
                        checkpoint(network, selected["unconstrained"], protocol, folder / "best_unconstrained.pt")
                    feasible = metrics["clean_relative_change_worst_record"] <= 0.01
                    if feasible and metrics["validation_snr_db"] > best_safe:
                        preservation = preservation or spectral_preservation(validation[1], identity_artifact, strength)
                        feasible = preservation["clean_alpha_error_db"] <= 0.5 and preservation["clean_beta_error_db"] <= 0.5
                        if feasible:
                            best_safe = metrics["validation_snr_db"]
                            selected["safe"] = {**row, **preservation, "preservation_required": True}
                            checkpoint(network, selected["safe"], protocol, folder / "best_safe.pt")
                    candidates.append({**row, "clean_change_gate_pass": metrics["clean_relative_change_worst_record"] <= 0.01,
                                       "spectral_gate_checked": preservation is not None, **(preservation or {})})
                pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
                pd.DataFrame(candidates).to_csv(folder / "validation_grid.csv", index=False)
                save_json(folder / "selected.json", finite_json(selected))
                print("SNR TRAIN", name, config_id, epoch + 1, "safe", round(best_safe, 4),
                      "unconstrained", round(best_free, 4), "seconds", round(time.perf_counter() - started, 1), flush=True)
        torch.save({"model": network.state_dict(), "optimizer": optimizer.state_dict(), "config": config},
                   folder / f"config_{config_id}_last.pt")
    return selected


def evaluate(name, arrays, keys, output, device):
    folder = output / name
    rows, predictions = [], {}
    for variant in ["safe", "unconstrained"]:
        state = torch.load(folder / f"best_{variant}.pt", map_location=device, weights_only=True)
        network = MODELS[name]().to(device)
        network.load_state_dict(state["model"])
        if hasattr(network, "inference_gate_mode"):
            network.inference_gate_mode = state["selection"].get("inference_gate_mode", "soft")
            network.inference_gate_offset = state["selection"].get("inference_gate_offset", 0.0)
        artifact, identity_artifact = prediction(network, arrays, device)
        strength = state["selection"]["strength"]
        predicted = arrays[0] - strength * artifact
        predictions[variant] = predicted.numpy()
        for index, (record, start) in enumerate(keys):
            target = arrays[1][index].numpy()
            clean_output = target - strength * identity_artifact[index].numpy()
            rows.append({"method": name + "_" + variant, "record": record, "start": start,
                "previously_inspected_test_record": True, **paired_metrics(predicted[index].numpy(), target),
                **{"clean_" + key: value for key, value in modification_metrics(clean_output, target).items()},
                "clean_alpha_error_db": paired_metrics(clean_output, target)["alpha_error_db"],
                "clean_beta_error_db": paired_metrics(clean_output, target)["beta_error_db"]})
    np.savez_compressed(folder / "test_predictions.npz", keys=np.asarray(keys), raw=arrays[0].numpy(),
                        target=arrays[1].numpy(), **predictions)
    pd.DataFrame(rows).to_csv(folder / "test_metrics.csv", index=False)
    return rows


def teachers(train, keys, selected, output):
    rows, accepted, diagnostics = [], [], []
    raw, clean, eyes, modes, clean_modes, valid, clean_valid = [item.numpy() for item in train]
    settings = selected["selected"]
    for index, (record, start) in enumerate(keys):
        evidence = window_evidence(modes[index], eyes[index], {}, valid[index].astype(bool))
        clean_evidence = window_evidence(clean_modes[index], eyes[index], {}, clean_valid[index].astype(bool))
        artifact = settings["strength"] * artifact_from_evidence(evidence, "correlation", settings["threshold"], True)
        identity = settings["strength"] * artifact_from_evidence(clean_evidence, "correlation", settings["threshold"], True)
        preservation = paired_metrics(clean[index] - identity, clean[index])
        change = modification_metrics(clean[index] - identity, clean[index])["relative_change"]
        improved = paired_metrics(raw[index] - artifact, clean[index])["rmse"] < paired_metrics(raw[index], clean[index])["rmse"]
        good = improved and change <= 0.01 and preservation["alpha_error_db"] <= 0.5 and preservation["beta_error_db"] <= 0.5
        rows.append(artifact.astype(np.float32) if good else np.zeros_like(raw[index]))
        accepted.append(good)
        diagnostics.append({"record": record, "start": start, "accepted": bool(good), "clean_change": change})
    save_json(output / "training_teacher_acceptance.json", finite_json(diagnostics))
    return torch.from_numpy(np.stack(rows)), torch.tensor(accepted)


def transfer_diagnostic(root, output, device, model_names):
    networks = {}
    for name in model_names:
        constructor = MODELS[name]
        state = torch.load(output / name / "best_safe.pt", map_location=device, weights_only=True)
        network = constructor().to(device).eval()
        network.load_state_dict(state["model"])
        if hasattr(network, "inference_gate_mode"):
            network.inference_gate_mode = state["selection"].get("inference_gate_mode", "soft")
            network.inference_gate_offset = state["selection"].get("inference_gate_offset", 0.0)
        networks[name] = (network, state["selection"]["strength"])
    rows, timings, exclusions = [], [], []
    sessions = {}
    for path in sorted((root / "Dataset1_OSF").rglob("*_prep.set")):
        sessions.setdefault(path.parent.name, path)
    for path in sessions.values():
        seen = set()
        for trial in list(osf_trials(path))[5:]:
            kind = trial_condition(trial)
            segment = annotated_score_slice(trial, calibration=0)
            if kind in seen or segment is None:
                continue
            try:
                references = ordered_references(trial)[:, segment]
            except ValueError as error:
                exclusions.append({"session": path.stem, "condition": kind, "reason": str(error)})
                continue
            seen.add(kind)
            raw = trial["eeg"][:, segment].copy()
            modes, valid = [], []
            started = time.perf_counter()
            for channel in raw:
                vectors, _, detail = rolling_decompose(channel, 3, 2000, max_iterations=2000)
                valid.append(not detail["hit_iteration_limit"])
                modes.append(vectors if valid[-1] else np.zeros_like(vectors))
            vmd_seconds = time.perf_counter() - started
            values = torch.from_numpy(raw[None]).to(device)
            vectors = torch.from_numpy(np.stack(modes)[None]).to(device)
            eyes = torch.from_numpy(references[None].copy()).to(device)
            mask = torch.ones(values.shape[:2], device=device)
            # Region identity was unknown during Klados fitting. Do not inject
            # an untrained anatomical flag into the main transfer estimate.
            regions = torch.full(values.shape[:2], 3, dtype=torch.long, device=device)
            for name, (network, strength) in networks.items():
                started = time.perf_counter()
                with torch.no_grad():
                    residual = network(values, vectors, eyes, mask, regions)["artifact"].cpu().numpy()[0]
                residual[~np.asarray(valid)] = 0
                corrected = raw - strength * residual
                rows.append({"method": name, "session": path.stem, "participant": trial["participant"],
                    "condition": kind, "trial": trial["trial"], "channels": len(raw),
                    **ocular_proxies(raw, corrected, {"HEOG": references[0], "VEOG": references[1]}, trial["labels"][segment])})
                timings.append({"method": name, "session": path.stem, "condition": kind,
                    "vmd_seconds": vmd_seconds, "network_and_transfer_seconds": time.perf_counter() - started,
                    "capped_channels_passthrough": int((~np.asarray(valid)).sum())})
            if len(seen) == 4:
                break
        print("SNR frozen OSF", path.stem, sorted(seen), flush=True)
    pd.DataFrame(rows).to_csv(output / "osf_frozen_transfer_proxies.csv", index=False)
    pd.DataFrame(timings).to_csv(output / "osf_transfer_runtimes.csv", index=False)
    save_json(output / "osf_transfer_exclusions.json", exclusions)


def load_verified_cache(source, part, dirty, clean, eyes, records, output):
    path = source / f"{part}_vmd_cache.npz"
    payload = np.load(path, allow_pickle=False)
    keys = [(int(record), int(start)) for record, start in payload["keys"]]
    expected = [(record, start) for record in records for start in STARTS]
    if keys != expected:
        raise ValueError("Prior-run cache keys differ from frozen split")
    for name, values in [("raw", dirty), ("clean", clean), ("eyes", eyes)]:
        actual = np.stack([values[record, :, start:start + WINDOW] for record, start in expected])
        if not np.array_equal(actual, payload[name]):
            raise ValueError("Prior-run cache signal differs from verified source: " + name)
    arrays = [torch.from_numpy(payload[name].copy()) for name in
              ["raw", "clean", "eyes", "modes", "clean_modes", "valid", "clean_valid"]]
    shutil.copyfile(path, output / path.name)
    save_json(output / f"{part}_cache_reuse.json", {"source_sha256": sha256_file(path),
        "signal_and_keys_verified": True, "records": records, "previous_source": str(source)})
    print("SNR verified cache reuse", part, len(keys), flush=True)
    return arrays, keys


def run_snr_target(root, output, device, context=False):
    torch.set_num_threads(2)
    benchmark = unique_input("record_split.json").parent
    source_selection = unique_input("reference_selected.json")
    source = json.loads(source_selection.read_text())
    if source.get("selected") is None or not source.get("all_training_decompositions_converged"):
        raise RuntimeError("Attach accepted converged reference-safe teacher output")
    split = json.loads((benchmark / "record_split.json").read_text())
    groups = [split[part] for part in ["train", "val", "test"]]
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Record split leakage")
    dirty, clean, eyes = klados_arrays(root)
    model_names = ["eog_vmd_context_student"] if context else ["eeg_vmd_student", "eog_vmd_gain_student"]
    reuse_source = unique_input("snr_target_protocol.json").parent if context else None
    if reuse_source is not None:
        prior = json.loads((reuse_source / "snr_target_protocol.json").read_text())
        if prior["split"] != split or prior["K"] != 3 or prior["alpha"] != 2000 or prior["max_iterations"] != 2000:
            raise ValueError("Incompatible completed SNR-target cache")
    if max(STARTS) + WINDOW > dirty.shape[-1]:
        raise ValueError("Fixed windows exceed recordings")
    cached = np.load(source_selection.parent / "training_mode_cache.npz", allow_pickle=False)
    training_cache = {(int(record), int(channel)): modes for (record, channel), modes in zip(cached["keys"], cached["modes"])}
    expected = {(record, channel) for record in split["train"] for channel in range(dirty.shape[1])}
    if set(training_cache) != expected:
        raise ValueError("Frozen VMD cache does not match the training split")
    protocol = {"target_min_db": 15, "loss_target_db": 20, "starts": STARTS, "window": WINDOW,
        "split": split, "model_inputs": {"eeg_vmd_student": "EEG and its own VMD modes; no runtime EOG",
        "eog_vmd_gain_student": "EEG, its VMD modes and runtime HEOG/VEOG"},
        "K": 3, "alpha": 2000, "tolerance": 1e-6, "max_iterations": 2000,
        "failed_vmd_policy": "identity channel passthrough; every original row remains scored",
        "selection": "maximize record-macro validation SNR with worst-record mean clean change <=1% and mean alpha/beta <=0.5dB",
        "unconstrained_checkpoint": "diagnostic only; never substituted for preservation-safe deployment model",
        "test_history": "all eight fixed test records have previous campaign evaluations; no fresh-test claim",
        "test_selection": "test scored only after every configuration completes; no test-based tuning",
        "subject_identity": "unverified; record-held-out only", "source_teacher_sha256": sha256_file(source_selection),
        "source_teacher_git_sha": json.loads((source_selection.parent / "run_config.json").read_text())["git_sha"],
        "augmentation": "15% training-channel dropout only; no split-crossing synthesis",
        "device": device, "full_validation": False}
    if context:
        protocol["model_inputs"] = {"eog_vmd_context_student": "EEG, VMD and runtime HEOG/VEOG; shared support evidence"}
        protocol["support_gate"] = "verified frontal maximum raw absolute EOG correlation; all observed rows fallback; learned bounded threshold, slope40"
        protocol["development_reason"] = "local EOG gain model reached target validation SNR but failed preservation; shared support separates cap-level intervention from per-channel amplitude"
        protocol["cache_source_git_sha"] = json.loads((reuse_source / "run_config.json").read_text())["git_sha"]
    save_json(output / "snr_target_protocol.json", protocol)
    arrays, keys = {}, {}
    for part in ["train", "val"]:
        arrays[part], keys[part] = (load_verified_cache(reuse_source, part, dirty, clean, eyes, split[part], output)
            if context else construct_cache(dirty, clean, eyes, split[part], part, output, training_cache))
    teacher, enabled = teachers(arrays["train"], keys["train"], source, output)
    selections = {}
    started = time.perf_counter()
    for name in model_names:
        selections[name] = train_model(name, arrays["train"], arrays["val"], split, output, device, protocol, teacher, enabled)
    # No held-out clean or inference is accessed for the parameter sweep above.
    arrays["test"], keys["test"] = (load_verified_cache(reuse_source, "test", dirty, clean, eyes, split["test"], output)
        if context else construct_cache(dirty, clean, eyes, split["test"], "test", output))
    rows = []
    for index, (record, start) in enumerate(keys["test"]):
        raw, target = arrays["test"][0][index].numpy(), arrays["test"][1][index].numpy()
        rows.append({"method": "raw", "record": record, "start": start, **paired_metrics(raw, target)})
        evidence = window_evidence(arrays["test"][3][index].numpy(), arrays["test"][2][index].numpy(), {},
                                  arrays["test"][5][index].numpy().astype(bool))
        artifact = artifact_from_evidence(evidence, "correlation", source["selected"]["threshold"], True)
        rows.append({"method": "frozen_reference_vmd", "record": record, "start": start,
                     **paired_metrics(raw - source["selected"]["strength"] * artifact, target)})
    for name in model_names:
        rows.extend(evaluate(name, arrays["test"], keys["test"], output, device))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "test_window_metrics.csv", index=False)
    records = frame.drop(columns=["start"]).groupby(["method", "record"], as_index=False).mean(numeric_only=True)
    records.to_csv(output / "test_record_metrics.csv", index=False)
    summary = records.drop(columns=["record"]).groupby("method", as_index=False).mean(numeric_only=True)
    worst_change = records.groupby("method").clean_relative_change.max()
    summary["worst_record_clean_change"] = summary.method.map(worst_change)
    summary["heldout_preservation_pass"] = ((summary.worst_record_clean_change <= 0.01)
        & (summary.clean_alpha_error_db <= 0.5) & (summary.clean_beta_error_db <= 0.5))
    summary.to_csv(output / "test_summary.csv", index=False)
    result = {"selections": selections, "test": summary.replace({np.nan: None}).to_dict(orient="records"),
        "runtime_training_and_final_evaluation_s": time.perf_counter() - started,
        "snr_definition": "10 log10(sum(clean^2)/sum((estimate-clean)^2)); macro windows within record then records",
        "parameters": {name: sum(p.numel() for p in MODELS[name]().parameters()) for name in model_names},
        "target_15db_met_with_preservation": {name: bool((summary.loc[summary.method == name + "_safe", "snr_db"].iloc[0] >= 15)
            and summary.loc[summary.method == name + "_safe", "heldout_preservation_pass"].iloc[0]) for name in model_names},
        "heldout_is_new": False, "complete_removal": False, "full_validation": False, "victory": False}
    save_json(output / "snr_target_summary.json", finite_json(result))
    print("FINAL SNR", summary[["method", "snr_db", "rmse", "clean_relative_change"]].to_string(index=False), flush=True)
    transfer_diagnostic(root, output, device, model_names)
    from .snr_target_plots import plot_snr_campaign
    plot_snr_campaign(output, model_names)
