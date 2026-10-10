"""Kaggle-only clean-control audit and saved-prediction distortion diagnosis."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .experiments import corpus_parent
from .io import atomic_json
from .metrics import paired, preservation_pass


def run_diagnosis(input_root, output, config, experiment):
    corpus, rows = corpus_parent(input_root)
    rows = [row for row in rows if row["target_kind"] == "controlled_recipient_reference"]
    lookup = {row["example_id"]: row for row in rows}
    contracts = []
    # Check the actual arrays, not just the condition labels in the manifest.
    for row in rows:
        if row["partition"]["role"] != "development":
            raise ValueError("Diagnosis may not open reserved confirmation sources")
        with np.load(corpus / row["array_path"], allow_pickle=False) as arrays:
            eeg = arrays["eeg"]
            target = arrays["paired_reference"]
            artifact = arrays["true_artifact"]
            mask = arrays["mask"].astype(bool)
            if eeg.shape != target.shape or eeg.shape != artifact.shape:
                raise ValueError("Paired array shapes differ")
            if mask.shape != (eeg.shape[0],) or not mask.any():
                raise ValueError("Invalid channel mask")
            if not all(np.isfinite(values[mask]).all() for values in (eeg, target, artifact)):
                raise ValueError("Nonfinite valid-channel input or target")
            is_clean = row["condition"] == "clean"
            if is_clean and (not np.array_equal(eeg[mask], target[mask])
                             or np.count_nonzero(artifact[mask])):
                raise ValueError("Clean-control input/target or zero-artifact contract failed")
            closure = float(np.max(np.abs(eeg[mask] - (target[mask] + artifact[mask]))))
            if closure > 1e-6 * max(float(np.max(np.abs(eeg[mask]))), 1e-12):
                raise ValueError("Controlled mixture does not reconstruct its input")
            contracts.append({"example_id": row["example_id"], "condition": row["condition"],
                              "clean_input_equals_target": is_clean, "closure_max": closure})
    pd.DataFrame(contracts).to_csv(output / "mixture_contracts.csv", index=False)

    requested = experiment.get("parents", [])
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Declare unique completed training parents")
    summaries, records = [], []
    for run_id in requested:
        candidates = []
        for path in Path(input_root).rglob("training_summary.json"):
            spec = json.loads((path.parent / "run_spec.json").read_text())
            if spec["run_id"] == run_id:
                candidates.append(path.parent)
        if len(candidates) != 1:
            raise ValueError("Missing or ambiguous diagnosis parent: " + run_id)
        parent = candidates[0]
        verify_parent(parent, ("training_summary.json", "neural_partitions.json"))
        summary = json.loads((parent / "training_summary.json").read_text())
        if (summary["campaign_id"] != config["campaign_id"]
                or summary["reserved_confirmation_opened"] is not False
                or summary.get("training_complete", True) is not True):
            raise ValueError("Diagnosis parent is incomplete or from another campaign")
        partitions = json.loads((parent / "neural_partitions.json").read_text())
        expected = {item["example_id"] for item in partitions["validation"]}
        paths = list((parent / "validation_predictions").glob("*.npz"))
        if {path.stem for path in paths} != expected:
            raise ValueError("Saved validation predictions are incomplete")
        clean_passes = []
        for path in paths:
            row = lookup[path.stem]
            with np.load(corpus / row["array_path"], allow_pickle=False) as arrays:
                source = arrays["eeg"].astype(np.float32)
                target = arrays["paired_reference"].astype(np.float32)
                source_mask = arrays["mask"].astype(bool)
            with np.load(path, allow_pickle=False) as saved:
                predicted = saved["cleaned"]
                mask = saved["mask"].astype(bool)
                if (not np.array_equal(mask, source_mask)
                        or not np.array_equal(saved["input"], source)
                        or not np.array_equal(saved["paired_reference"], target)):
                    raise ValueError("Saved prediction has different inputs, targets or masks")
                if predicted.shape != target.shape or not np.isfinite(predicted[mask]).all():
                    raise ValueError("Invalid saved neural estimate")
            if row["condition"] == "clean":
                identity = paired(source[mask], target[mask], config["fs"])
                if not preservation_pass(identity, config["preservation"]):
                    raise ValueError("Identity baseline failed on an exact clean control")
                metrics = paired(predicted[mask], target[mask], config["fs"])
                clean_passes.append(preservation_pass(metrics, config["preservation"]))
            for channel in np.flatnonzero(mask):
                truth = target[channel].astype(float)
                estimate = predicted[channel].astype(float)
                energy = float(truth @ truth)
                if energy <= 1e-24:
                    raise ValueError("Zero target energy needs a separately declared metric policy")
                records.append({"run_id": run_id, "example_id": row["example_id"],
                    "recipient": row["recipient"], "condition": row["condition"],
                    "input_level": row.get("input_snr_db"), "channel": int(channel),
                    "target_projection_gain": float(estimate @ truth / energy),
                    "target_relative_error": float(np.linalg.norm(estimate - truth) / np.sqrt(energy)),
                    "output_input_rms_ratio": float(np.linalg.norm(estimate) / max(np.linalg.norm(source[channel]), 1e-12))})
        if not clean_passes:
            raise ValueError("Diagnosis parent has no clean validation controls")
        summaries.append({"run_id": run_id, "arm": summary["experiment"]["arm"],
            "loss_profile": summary["experiment"]["loss_profile"],
            "reported_snr_db": summary["snr_db"], "clean_windows": len(clean_passes),
            "clean_windows_passing": int(sum(clean_passes)),
            "scientific_qualified": False})
    pd.DataFrame(records).to_csv(output / "prediction_distortion.csv.gz", index=False)
    atomic_json(output / "neural_diagnosis_summary.json", {
        "campaign_id": config["campaign_id"], "clean_contract_passed": True,
        "saved_prediction_identity_verified": True, "controlled_examples": len(rows),
        "parents": summaries, "reserved_confirmation_opened": False,
        "scientific_qualified": False,
        "scope": "Array and saved-prediction audit; loss causality is not established"})
