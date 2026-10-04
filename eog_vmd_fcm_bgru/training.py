"""Single offline model for both ocular sources; bounded Kaggle feasibility."""
import json
import pickle
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from .dataset_io import klados_arrays, osf_trials, trial_condition, annotated_score_slice
from .channel_regions import REGION_NAMES, region_ids
from .evaluation import paired_metrics, modification_metrics, ocular_proxies
from .experiment import CALIBRATION, WINDOW
from .provenance import save_json
from .spatial_expert import ICAExpert
from .student import SharedChannelStudent, reconstruction_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_campaign(root, output, device="cpu", profile="kaggle_smoke"):
    if profile != "kaggle_smoke":
        raise ValueError("Full grouped multi-seed campaign is not the feasibility profile")
    gates = list(Path("/kaggle/input").rglob("classical_gate.json"))
    if len(gates) != 1:
        raise RuntimeError("Attach one completed benchmark output before training")
    benchmark = gates[0].parent
    gate = json.loads(gates[0].read_text())
    split = json.loads((benchmark / "record_split.json").read_text())
    spatial_config = json.loads((benchmark / "selected_spatial.json").read_text())
    expert, vmd_config = None, None
    if gate.get("vmd_feasible"):
        vmd_config = json.loads((benchmark / "selected_vmd.json").read_text())
        with (benchmark / "vmd_expert.pkl").open("rb") as handle:
            expert = pickle.load(handle)
    elif not gate.get("paired_only_training_allowed"):
        raise RuntimeError("Failed or incomplete benchmark; no authorized training path")
    dirty, clean, eog = klados_arrays(root)
    seed_everything(42)
    torch.set_num_threads(2)
    segment = slice(CALIBRATION, CALIBRATION + WINDOW)
    coverage = {name: records[:6 if name == "train" else 2] for name, records in split.items() if isinstance(records, list)}
    x = torch.from_numpy(dirty[coverage["train"], :, segment].copy())
    y = torch.from_numpy(clean[coverage["train"], :, segment].copy())
    teachers, teacher_diagnostics = [], []
    for record in coverage["train"]:
        raw = dirty[record, :, segment]
        temporal = np.zeros_like(raw)
        if expert is not None:
            temporal, _ = expert.artifact(raw, strength=vmd_config["strength"])
        teacher = temporal
        if expert is not None and "ica" in spatial_config:
            try:
                method, threshold = spatial_config["ica"].split(":")
                spatial = ICAExpert.fit(dirty[record, :, :CALIBRATION], eog[record, :, :CALIBRATION], method)
                refined = spatial.residual(raw, float(threshold), True, vmd_config["K"], vmd_config["alpha"])
                teacher = 0.5 * temporal + 0.5 * refined
            except Exception as error:
                teacher_diagnostics.append({"record": record, "ica_teacher_error": repr(error)})
        teacher_rmse = paired_metrics(raw - teacher, clean[record, :, segment])["rmse"]
        raw_rmse = paired_metrics(raw, clean[record, :, segment])["rmse"]
        usable = expert is not None and teacher_rmse < raw_rmse
        teachers.append(teacher if usable else np.zeros_like(teacher))
        teacher_diagnostics.append({"record": record, "teacher_rmse": teacher_rmse, "raw_rmse": raw_rmse,
                                    "distillation_enabled": bool(usable)})
    save_json(output / "teacher_diagnostics.json", teacher_diagnostics)
    teacher = torch.from_numpy(np.stack(teachers).astype(np.float32))
    usable = torch.tensor([row["distillation_enabled"] for row in teacher_diagnostics if "distillation_enabled" in row])
    loader = DataLoader(TensorDataset(x, y, teacher, usable), batch_size=2, shuffle=True,
                        generator=torch.Generator().manual_seed(42), num_workers=0)
    network = SharedChannelStudent().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.0001)
    parameter_count = sum(item.numel() for item in network.parameters())
    best, history = float("inf"), []
    started = time.perf_counter()
    for epoch in range(20):
        network.train()
        losses = []
        for inputs, targets, privileged, enabled in loader:
            inputs, targets, privileged = (item.to(device) for item in (inputs, targets, privileged))
            enabled = enabled.to(device)
            # Random channel subsets expose cap-size variation. At least one
            # electrode per cap remains valid, and masking precedes encoding.
            mask = (torch.rand(inputs.shape[:2], device=device) > 0.15).float()
            mask[:, 0] = 1
            regions = torch.full(inputs.shape[:2], 3, device=device, dtype=torch.long)
            scale = inputs.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            output_dirty = network(inputs, mask, regions)
            loss = reconstruction_loss(output_dirty["cleaned"] / scale, targets / scale, mask)
            output_clean = network(targets, mask, regions)
            loss = loss + 0.25 * reconstruction_loss(output_clean["cleaned"] / scale, targets / scale, mask)
            if enabled.any():
                loss = loss + 0.10 * reconstruction_loss(output_dirty["artifact"][enabled] / scale[enabled],
                                                          privileged[enabled] / scale[enabled], mask[enabled])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite student loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        network.eval()
        val_scores = []
        with torch.no_grad():
            for record in coverage["val"]:
                values = torch.from_numpy(dirty[record:record + 1, :, segment]).to(device)
                mask = torch.ones(values.shape[:2], device=device)
                regions = torch.full(values.shape[:2], 3, device=device, dtype=torch.long)
                predicted = network(values, mask, regions)["cleaned"].cpu().numpy()[0]
                val_scores.append(paired_metrics(predicted, clean[record, :, segment])["rmse"])
        validation = float(np.mean(val_scores))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_rmse": validation})
        state = {"model": network.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                 "seed": 42, "profile": profile, "teacher_git_sha": json.loads((benchmark / "run_config.json").read_text())["git_sha"]}
        torch.save(state, output / "last.pt")
        if validation < best:
            best = validation
            torch.save(state, output / "best.pt")
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print("Student", epoch + 1, "validation RMSE", validation, flush=True)
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    network.load_state_dict(checkpoint["model"])
    network.eval()
    rows, predictions = [], []
    with torch.no_grad():
        for record in coverage["test"]:
            values = torch.from_numpy(dirty[record:record + 1, :, segment]).to(device)
            target = clean[record, :, segment]
            mask = torch.ones(values.shape[:2], device=device)
            regions = torch.full(values.shape[:2], 3, device=device, dtype=torch.long)
            predicted = network(values, mask, regions)["cleaned"].cpu().numpy()[0]
            clean_input = torch.from_numpy(target[None]).to(device)
            clean_output = network(clean_input, mask, regions)["cleaned"].cpu().numpy()[0]
            rows.append({"record": record, "method": "eeg_only_student", **paired_metrics(predicted, target),
                         **{"clean_" + key: value for key, value in modification_metrics(clean_output, target).items()}})
            rows.append({"record": record, "method": "raw", **paired_metrics(dirty[record, :, segment], target)})
            predictions.append(predicted)
    pd.DataFrame(rows).to_csv(output / "student_metrics.csv", index=False)
    np.savez_compressed(output / "student_test_predictions.npz", record_ids=coverage["test"],
                        predictions=np.stack(predictions), raw=dirty[coverage["test"], :, segment],
                        target=clean[coverage["test"], :, segment])
    save_json(output / "training_summary.json", {"profile": profile, "coverage": coverage, "seed": 42,
              "epochs": 20, "parameters": parameter_count, "device": device,
              "runtime_s": time.perf_counter() - started, "input": "EEG only; no runtime EOG, VMD or ICA",
              "offline": True, "regions": "unknown Klados electrode provenance; fixed regional priors unvalidated",
              "training_mode": "paired plus validated teacher" if expert is not None else "paired only; VMD teacher rejected",
              "full_validation": False, "victory": False})
    evaluate_osf_student(network, root, output, device)
    latency_scaling(network, output, device)


def evaluate_osf_student(network, root, output, device):
    """Frozen Protocol Z subset, without treating OSF proxies as clean truth."""
    rows = []
    for path in sorted((root / "Dataset1_OSF").rglob("*_prep.set"))[:2]:
        seen = set()
        for trial in list(osf_trials(path))[5:]:
            labels = trial["labels"]
            if labels is None or trial["eeg"].shape[-1] < WINDOW:
                continue
            kind = trial_condition(trial)
            segment = annotated_score_slice(trial, calibration=0)
            if kind is None or kind in seen or segment is None:
                continue
            seen.add(kind)
            raw = trial["eeg"][:, segment]
            values = torch.from_numpy(raw[None].copy()).to(device)
            mask = torch.ones(values.shape[:2], device=device)
            ids = region_ids(trial["names"])
            eog = {name: value[segment] for name, value in trial["eog"].items()}
            for variant in ["shared_student", "regional_prior_student"]:
                regions = torch.from_numpy(ids[None]).to(device) if variant == "regional_prior_student" else torch.full(values.shape[:2], 3, device=device, dtype=torch.long)
                with torch.no_grad():
                    predicted = network(values, mask, regions)["cleaned"].cpu().numpy()[0]
                for region, name in enumerate(REGION_NAMES):
                    keep = ids == region
                    if keep.any():
                        rows.append({"study": trial["study"], "participant": trial["participant"],
                                     "session": trial["session"], "trial": trial["trial"], "coverage_type": kind,
                                     "variant": variant, "region": name, "channels": int(keep.sum()),
                                     "scoring_start": segment.start, "scoring_stop": segment.stop,
                                     **ocular_proxies(raw[keep], predicted[keep], eog, labels[segment])})
            pd.DataFrame(rows).to_csv(output / "student_osf_proxies.csv", index=False)
            if len(seen) == 4:
                break


def latency_scaling(network, output, device):
    """Measured offline inference timing; synthetic cap tensors, not accuracy."""
    rows = []
    for channels in [1, 8, 19, 32, 64, 128]:
        values = torch.randn(1, channels, WINDOW, device=device)
        mask = torch.ones(values.shape[:2], device=device)
        regions = torch.full(values.shape[:2], 3, device=device, dtype=torch.long)
        with torch.no_grad():
            for _ in range(3):
                network(values, mask, regions)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            measurements = []
            for _ in range(10):
                started = time.perf_counter()
                network(values, mask, regions)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                measurements.append((time.perf_counter() - started) * 1000)
            peak = torch.cuda.max_memory_allocated() - baseline if device.startswith("cuda") else None
        rows.append({"channels": channels, "samples": WINDOW, "batch": 1, "repetitions": 10,
                     "median_ms": float(np.median(measurements)), "p95_ms": float(np.percentile(measurements, 95)),
                     "cuda_extra_peak_allocated_bytes": peak, "device": device,
                     "scope": "synthetic offline tensor throughput; no real-time certification"})
    pd.DataFrame(rows).to_csv(output / "student_latency_scaling.csv", index=False)
