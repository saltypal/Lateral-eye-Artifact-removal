"""Manifest-driven EEG-only training, preservation selection and frozen evaluation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .campaign_contracts import atomic_json, canonical_hash, sha256_file
from .controlled_corpus import iter_materialized_examples
from .losses_v2 import LossV2Config, regional_student_loss
from .model_factory import build_deployment_model, REGIONAL_ARCHITECTURE
from .splits_v2 import assert_oof_fit_ids


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


class CorpusDataset(Dataset):
    """One frozen split role; test targets are never opened by training jobs."""
    def __init__(self, root, role, teacher_root=None):
        self.root = Path(root)
        if role not in {"train", "val", "test"}:
            raise ValueError("Unknown corpus split role")
        # Read metadata for all roles; load array content only for the requested role.
        self.rows = [row for row in _jsonl(self.root / "corpus_manifest.jsonl") if row["split_role"] == role]
        if not self.rows or any(not row.get("array_path") for row in self.rows):
            raise ValueError(f"No materialized {role} examples")
        self.teacher_root = None if teacher_root is None else Path(teacher_root)
        self.teachers = {} if teacher_root is None else {row["example_id"]: row for row in _jsonl(self.teacher_root / "teacher_manifest.jsonl")}
        if teacher_root is not None:
            for row in self.rows:
                target = self.teachers.get(row["example_id"])
                if target is None or not target.get("teacher_oof"):
                    raise ValueError("Missing OOF teacher target: " + row["example_id"])
                ids = [row["record_id"], row.get("recipient_id"), row.get("donor_id")]
                assert_oof_fit_ids(target["fit_record_ids"], [value for value in ids if value])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["array_path"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Corpus path escapes artifact root")
        with np.load(path, allow_pickle=False) as values:
            arrays = {name: values[name].copy() for name in values.files}
        for name in ("eeg", "mask", "regions", "hemispheres", "coordinates", "coordinate_mask"):
            if name not in arrays:
                raise ValueError("Missing corpus tensor: " + name)
        if arrays["eeg"].ndim != 2 or arrays["mask"].shape != arrays["eeg"].shape[:1]:
            raise ValueError("Invalid corpus shapes")
        if not np.isfinite(arrays["eeg"][arrays["mask"].astype(bool)]).all():
            raise ValueError("Nonfinite valid EEG")
        if self.teacher_root is not None:
            teacher = self.teachers[row["example_id"]]
            path = (self.teacher_root / teacher["array_path"]).resolve()
            if not path.is_relative_to(self.teacher_root.resolve()):
                raise ValueError("Teacher path escapes artifact root")
            with np.load(path, allow_pickle=False) as target:
                arrays["teacher_artifact"] = target["teacher_artifact"].copy()
                weight = target["teacher_weight"].copy()
            if weight.ndim == 1:
                weight = np.broadcast_to(weight[:, None], arrays["eeg"].shape).copy()
            arrays["teacher_weight"] = weight
            arrays["teacher_oof"] = np.asarray(True)
        return row, arrays


def collate_caps(examples):
    """Pad channel counts only; temporal windows must have an identical contract."""
    rows = [row for row, _ in examples]
    lengths = {arrays["eeg"].shape[-1] for _, arrays in examples}
    if len(lengths) != 1:
        raise ValueError("Training windows must have a fixed sample length")
    count = max(arrays["eeg"].shape[0] for _, arrays in examples)
    common = set.intersection(*(set(arrays) for _, arrays in examples))
    for name in ("paired_reference", "teacher_artifact", "teacher_weight", "teacher_oof", "event_targets", "event_mask"):
        if any(name in arrays for _, arrays in examples) and name not in common:
            raise ValueError(f"Mixed supervision fields in a batch: {name}")
    tensors = {}
    channel_fields = {"eeg", "paired_reference", "mask", "regions", "hemispheres", "coordinates", "coordinate_mask", "teacher_artifact", "teacher_weight", "event_targets", "event_mask"}
    for name in common.intersection(channel_fields | {"teacher_oof"}):
        padded = []
        for _, arrays in examples:
            value = arrays[name]
            if name in channel_fields:
                target = np.full((count, *value.shape[1:]), 3 if name in {"regions", "hemispheres"} else 0, dtype=value.dtype)
                target[:value.shape[0]] = value
                value = target
            padded.append(value)
        tensors[name] = torch.as_tensor(np.stack(padded))
    return rows, tensors


def _forward(model, batch, eeg=None):
    return model(batch["eeg"].float() if eeg is None else eeg, batch["mask"], batch["regions"],
                 batch["hemispheres"], batch["coordinates"].float(), batch["coordinate_mask"])


def _bandpower(values, low, high, fs=200):
    spectrum = np.abs(np.fft.rfft(values - values.mean(axis=-1, keepdims=True), axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(values.shape[-1], 1 / fs)
    return spectrum[..., (frequencies >= low) & (frequencies < high)].sum(axis=-1)


def _metrics(clean, dirty, predicted, preserved):
    energy = float(np.square(clean).sum())
    error = float(np.square(predicted - clean).sum())
    input_error = float(np.square(dirty - clean).sum())
    if energy <= 0:
        raise ValueError("Zero-energy paired reference cannot define SNR")
    alpha = np.abs(10 * np.log10((_bandpower(preserved, 8, 13) + 1e-20) / (_bandpower(clean, 8, 13) + 1e-20)))
    beta = np.abs(10 * np.log10((_bandpower(preserved, 13, 30) + 1e-20) / (_bandpower(clean, 13, 30) + 1e-20)))
    left = clean - clean.mean(axis=-1, keepdims=True)
    right = preserved - preserved.mean(axis=-1, keepdims=True)
    covariance = left @ left.T / max(1, clean.shape[-1] - 1)
    estimate = right @ right.T / max(1, clean.shape[-1] - 1)
    pearson = []
    for source, target in zip(left, right):
        denominator = np.linalg.norm(source) * np.linalg.norm(target)
        if denominator > 0:
            pearson.append(float(source @ target / denominator))
    return {"reference_energy": energy, "error_energy": error, "input_error_energy": input_error,
            "snr_db": 10 * np.log10(energy / error) if error > 0 else None,
            "snr_status": "perfect_reconstruction" if error == 0 else "finite",
            "clean_modification_ratio": float(np.square(preserved - clean).sum()) / energy,
            "alpha_distortion_db": float(alpha.mean()), "beta_distortion_db": float(beta.mean()),
            "covariance_distortion": float(np.linalg.norm(estimate - covariance) / max(np.linalg.norm(covariance), 1e-20)),
            "pearson_deterioration": 1 - float(np.mean(pearson)) if pearson else None}


def summarize_rows(rows):
    """Macro-average recording results; overlapping windows are not independent subjects."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["record_id"]].append(row)
    records = []
    keys = ("clean_modification_ratio", "alpha_distortion_db", "beta_distortion_db", "covariance_distortion", "pearson_deterioration")
    for record_id, windows in groups.items():
        energy = sum(row["reference_energy"] for row in windows)
        error = sum(row["error_energy"] for row in windows)
        value = {"record_id": record_id, "participant_id": windows[0].get("participant_id"), "target_kind": windows[0].get("target_kind"),
                 "windows": len(windows), "snr_db": 10 * np.log10(energy / error) if error > 0 else None,
                 "snr_status": "perfect_reconstruction" if error == 0 else "finite",
                 "normalized_error": error / energy, "input_normalized_error": sum(row["input_error_energy"] for row in windows) / energy}
        value.update({key: float(np.mean([row[key] for row in windows if row[key] is not None])) if any(row[key] is not None for row in windows) else None for key in keys})
        records.append(value)
    thresholds = dict(clean_modification_ratio=.01, alpha_distortion_db=.5, beta_distortion_db=.5,
                      covariance_distortion=.02, pearson_deterioration=.005)
    gates = {key: all(row[key] is not None and row[key] <= limit for row in records) for key, limit in thresholds.items()}
    finite_snr = [row["snr_db"] for row in records if row["snr_db"] is not None]
    participants = defaultdict(list)
    for row in records:
        participants[row["participant_id"] or "unverified:" + row["record_id"]].append(row)
    grouped = [{"group": group, "verified_participant": not group.startswith("unverified:"),
                "records": len(items), "normalized_error": float(np.mean([row["normalized_error"] for row in items])),
                **{key: float(np.mean([row[key] for row in items])) if all(row[key] is not None for row in items) else None for key in keys}}
               for group, items in sorted(participants.items())]
    errors = np.asarray([group["normalized_error"] for group in grouped])
    generator = np.random.default_rng(42)
    interval = np.quantile([np.mean(generator.choice(errors, len(errors), replace=True)) for _ in range(1000)], [.025, .975]).tolist() if len(errors) > 1 else None
    return {"records": records, "record_count": len(records), "window_count": len(rows),
            "mean_record_snr_db": float(np.mean(finite_snr)) if finite_snr else None,
            "perfect_reconstruction_records": sum(row["snr_status"] == "perfect_reconstruction" for row in records),
            "participant_or_engineering_groups": grouped, "group_error_bootstrap_95ci": interval,
            "mean_record_normalized_error": float(np.mean([row["normalized_error"] for row in records])),
            "preservation_gates": gates, "feasible": all(gates.values()),
            "snr_mean_scope": "finite recording SNR only; perfect reconstructions have a separate status",
            "target_scope": "native-clean unknown for OSF controlled recipient references; paired Klados engineering cohort"}


def evaluate(model, dataset, device, output=None):
    model.eval()
    rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            entry, arrays = dataset[index]
            if "paired_reference" not in arrays:
                raise ValueError("Paired validation must have a declared reference target")
            _, batch = collate_caps([(entry, arrays)])
            batch = {key: value.to(device) for key, value in batch.items()}
            predicted = _forward(model, batch)["cleaned"][0].cpu().numpy()
            clean = batch["paired_reference"].float()
            preserved = _forward(model, batch, clean)["cleaned"][0].cpu().numpy()
            keep = arrays["mask"].astype(bool)
            scores = _metrics(arrays["paired_reference"][keep], arrays["eeg"][keep], predicted[keep], preserved[keep])
            rows.append({"example_id": entry["example_id"], "record_id": entry["record_id"],
                         "participant_id": entry.get("participant_id"), "target_kind": entry["target_kind"], **scores})
    summary = summarize_rows(rows)
    if output:
        atomic_json(Path(output) / "evaluation.json", summary)
        (Path(output) / "window_metrics.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    return summary


def _save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(path, device):
    # Trusted exact-parent checkpoint has already been SHA256 verified by campaign_v2.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_deployment_model(checkpoint["architecture_id"], checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def train_one(output, config, train, validation, device, initial=None, resume=None):
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = build_deployment_model(REGIONAL_ARCHITECTURE, config.get("architecture", {})).to(device)
    if initial:
        pretrained, _ = _load_checkpoint(initial, device)
        model.load_state_dict(pretrained.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", .001)), weight_decay=float(config.get("weight_decay", .0001)))
    loss_config = LossV2Config(**config.get("loss", {}))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=int(config.get("microbatch", 1)), shuffle=True, num_workers=0,
                        collate_fn=collate_caps, generator=generator)
    best_feasible = float("inf")
    best_unconstrained = float("inf")
    start_epoch, stagnant = 0, 0
    configuration_hash = canonical_hash(config)
    if resume:
        restored, state = _load_checkpoint(resume, device)
        if state["training_config_hash"] != configuration_hash:
            raise ValueError("Resume configuration differs from checkpoint")
        model.load_state_dict(restored.state_dict())
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_feasible, best_unconstrained, stagnant = state["best_feasible"], state["best_unconstrained"], state["stagnant"]
        torch.set_rng_state(state["torch_rng"].cpu())
        generator.set_state(state["loader_rng"].cpu())
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    started = time.monotonic()
    history = []
    accumulation = max(1, int(config.get("effective_batch", 8)) // int(config.get("microbatch", 1)))
    status = "maximum_epochs"
    for epoch in range(start_epoch, int(config.get("max_epochs", 80))):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, batches = 0., 0
        for index, (_, batch) in enumerate(loader):
            batch = {name: value.to(device) for name, value in batch.items()}
            prediction = _forward(model, batch)
            valid = batch["mask"].bool()[..., None]
            scale = (torch.where(valid, batch["eeg"].float(), 0).square().sum(dim=(1, 2), keepdim=True) /
                     (valid.sum(dim=(1, 2), keepdim=True) * batch["eeg"].shape[-1]).clamp_min(1)).sqrt().clamp_min(1e-6)
            scaled = {**prediction, "cleaned": prediction["cleaned"] / scale, "artifact": prediction["artifact"] / scale}
            loss, _ = regional_student_loss(scaled, batch["eeg"].float() / scale, batch["mask"],
                paired_clean=None if "paired_reference" not in batch else batch["paired_reference"].float() / scale,
                teacher_artifact=None if "teacher_artifact" not in batch else batch["teacher_artifact"].float() / scale,
                teacher_weight=batch.get("teacher_weight"), teacher_oof=batch.get("teacher_oof"),
                event_targets=batch.get("event_targets"), event_mask=batch.get("event_mask"), config=loss_config)
            if "paired_reference" in batch:
                clean = batch["paired_reference"].float()
                preserved = _forward(model, batch, clean)
                identity, _ = regional_student_loss({**preserved, "cleaned": preserved["cleaned"] / scale,
                    "artifact": preserved["artifact"] / scale}, clean / scale, batch["mask"], clean_input=clean / scale,
                    config=replace(loss_config, paired_weight=0, teacher_weight=0, event_weight=0, snr_shortfall_weight=0))
                loss = loss + identity
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training objective")
            group_start = (index // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            (loss / group_size).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
            batches += 1
        metrics = evaluate(model, validation, device)
        error = metrics["mean_record_normalized_error"]
        improved = metrics["feasible"] and error < best_feasible
        stagnant = 0 if improved else stagnant + 1
        payload = {"architecture_id": REGIONAL_ARCHITECTURE, "architecture": model.architecture_config(),
                   "state_dict": model.state_dict(), "training_config_hash": configuration_hash,
                   "epoch": epoch, "validation": metrics, "optimizer": optimizer.state_dict(),
                   "best_feasible": min(best_feasible, error) if improved else best_feasible,
                   "best_unconstrained": min(best_unconstrained, error), "stagnant": stagnant,
                   "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                   "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        if improved:
            best_feasible = error
            _save_checkpoint(output / "best_feasible.pt", payload)
        if error < best_unconstrained:
            best_unconstrained = error
            _save_checkpoint(output / "best_unconstrained.pt", payload)
        _save_checkpoint(output / "resume.pt", payload)
        history.append({"epoch": epoch, "loss": total / max(1, batches), "validation": metrics})
        atomic_json(output / "history.json", history)
        print(f"Epoch {epoch}: loss={total / max(1, batches):.6g}, val SNR={metrics['mean_record_snr_db']}, feasible={metrics['feasible']}", flush=True)
        if stagnant >= int(config.get("patience", 15)):
            status = "early_stopping"
            break
        if time.monotonic() - started >= float(config.get("max_seconds", 14400)):
            status = "wall_time_budget_at_epoch_boundary"
            break
    feasible_path = output / "best_feasible.pt"
    summary = {"feasible_checkpoint": feasible_path.name if feasible_path.exists() else None,
               "epochs_completed": len(history), "stop_reason": status, "runtime_seconds": time.monotonic() - started,
               "training_config_hash": configuration_hash, "config": config, "seed": seed,
               "development_scope": "training and validation only; no test targets opened"}
    if feasible_path.exists():
        _, checkpoint = _load_checkpoint(feasible_path, "cpu")
        summary["validation"] = checkpoint["validation"]
        summary["checkpoint_sha256"] = sha256_file(feasible_path)
    atomic_json(output / "training_summary.json", summary)
    return summary


def run_student_stage(stage, data_root, output, config, parents):
    device = "cuda:0" if torch.cuda.is_available() and config.get("accelerator", "gpu") != "cpu" else "cpu"
    corpus = parents.get("corpus-build")
    if stage in {"student-paired", "student-distill"}:
        if corpus is None:
            raise ValueError("Training requires corpus-build parent")
        teacher = parents.get("teacher-oof") if stage == "student-distill" else None
        if stage == "student-distill" and teacher is None:
            raise ValueError("Distillation requires verified OOF teacher parent")
        train = CorpusDataset(corpus, "train", teacher)
        validation = CorpusDataset(corpus, "val")
        initial = None
        if parents.get("student-paired"):
            initial = Path(parents["student-paired"]) / "best_feasible.pt"
            if not initial.is_file():
                raise ValueError("Paired parent has no feasible checkpoint")
        configurations = config.get("search", [{}])
        summaries = []
        for index, trial in enumerate(configurations):
            trial_config = {key: value for key, value in config.items() if key != "search"}
            trial_config.update(trial)
            trial_config["loss"] = {**config.get("loss", {}), **trial.get("loss", {})}
            location = output / f"trial-{index:03d}"
            summaries.append({"trial_path": str(location.relative_to(output)), **train_one(location, trial_config, train, validation, device, initial=initial)})
        feasible = [row for row in summaries if row["feasible_checkpoint"]]
        selected = min(feasible, key=lambda row: row["validation"]["mean_record_normalized_error"]) if feasible else None
        summary = {"trials": summaries, "selected_trial": selected["trial_path"] if selected else None,
                   "selection_partition": "val", "test_accessed": False, "feasible": selected is not None}
        if selected:
            import shutil
            shutil.copyfile(output / selected["trial_path"] / "best_feasible.pt", output / "best_feasible.pt")
        atomic_json(output / "training_summary.json", summary)
        return summary
    if stage == "select-freeze":
        candidates = []
        for alias, root in parents.items():
            path = Path(root) / "best_feasible.pt"
            if path.is_file():
                _, checkpoint = _load_checkpoint(path, "cpu")
                candidates.append((checkpoint["validation"]["mean_record_normalized_error"], alias, path, checkpoint))
        if not candidates:
            raise ValueError("No preservation-feasible student to freeze")
        _, alias, path, checkpoint = min(candidates, key=lambda item: item[0])
        import shutil
        shutil.copyfile(path, output / "frozen_model.pt")
        frozen = {"schema_version": 2, "selected_parent": alias, "model_sha256": sha256_file(output / "frozen_model.pt"),
                  "validation": checkpoint["validation"], "architecture_id": checkpoint["architecture_id"],
                  "architecture": checkpoint["architecture"], "training_config_hash": checkpoint["training_config_hash"],
                  "git_sha": config["git_sha"], "selection_frozen": True, "test_accessed": False}
        atomic_json(output / "frozen_manifest.json", frozen)
        return frozen
    if stage in {"final-eval", "export"}:
        if corpus is None or parents.get("select-freeze") is None:
            raise ValueError("Frozen evaluation/export requires corpus and frozen model parents")
        frozen_root = Path(parents["select-freeze"])
        frozen = json.loads((frozen_root / "frozen_manifest.json").read_text())
        if not frozen.get("selection_frozen") or sha256_file(frozen_root / "frozen_model.pt") != frozen["model_sha256"]:
            raise ValueError("Frozen model identity mismatch")
        model, checkpoint = _load_checkpoint(frozen_root / "frozen_model.pt", device)
        if stage == "final-eval":
            test = CorpusDataset(corpus, "test")
            return evaluate(model, test, device, output)
        summary = {"frozen_model_sha256": frozen["model_sha256"], "test_accessed": False}
        if stage == "export":
            from .export_model import export_and_verify
            # Deployment parity uses an already development-exposed fixture; no final targets reopen.
            fixture = CorpusDataset(corpus, "val")
            entry, arrays = fixture[0]
            _, batch = collate_caps([(entry, arrays)])
            batch = {name: value.to(device) for name, value in batch.items()}
            model.eval()
            with torch.no_grad():
                prediction = _forward(model, batch)["cleaned"].cpu().numpy()
            atomic_json(output / "run_config.json", {"git_sha": config["git_sha"], "seed": config["seed"]})
            atomic_json(output / "training_summary.json", {"frozen": frozen, "evaluation": summary})
            export_and_verify(model, output, arrays["eeg"][None], prediction,
                arrays["mask"][None], arrays["regions"][None], arrays["hemispheres"][None],
                arrays["coordinates"][None], arrays["coordinate_mask"][None])
            summary["fresh_process_export_verified"] = True
        return summary
    raise ValueError("Unsupported student stage: " + stage)
