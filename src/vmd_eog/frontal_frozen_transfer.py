"""Frontal-only transfer of the archived EOG-assisted VMD gain model.

No weight fitting, K selection or posterior VMD occurs. The historical anatomy
feature is disabled because it was never trained on verified frontal metadata.
External verified masks still strictly determine which electrodes are processed.
"""
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch

from .artifacts import verify_parent
from .experiments import corpus_parent
from .frontal_evaluation import selected_parent, validate_evaluation_sources
from .frontal_methods import FixedRecipe, FrontalCorrector, frontal_indices
from .frontal_replication import ArchivedModes, finite_value
from .frontal_review import strict_mean, save_controlled_review
from .io import atomic_json, read_jsonl, sha256_file
from .metrics import paired
from .paper_metrics import paired_channels
from .posterior_context import select_examples


LEGACY_SHA = "0b183e6c09f6ecfbb3ed4818028eb5abbdf842fc"
CHECKPOINT_SHA = "4bb7c1b0cfa0e8341c4053ab3099efb16832f045f99e37afde1904a93c2cd0a9"


def apply_frontal(model, eeg, references, names, regions, mask, bundles):
    """One subtraction from original EEG; paired clean targets are not inputs."""
    ids = frontal_indices(eeg, names, regions, mask)
    artifact = np.zeros_like(eeg, dtype=np.float64)
    if not len(ids):
        return np.asarray(eeg).copy(), artifact, []
    if set(bundles) != set(int(index) for index in ids):
        raise ValueError("Frozen transfer modes must cover exactly the verified frontal rows")
    modes = np.stack([bundles[int(index)][0] for index in ids])
    if modes.shape != (len(ids), 3, eeg.shape[-1]):
        raise ValueError("Frozen transfer requires fixed K3 modes")
    usable = np.asarray([not (bundles[int(index)][2].get("failure") or
        bundles[int(index)][2].get("hit_iteration_limit")) for index in ids])
    values = torch.from_numpy(np.asarray(eeg[ids], dtype=np.float32)[None].copy())
    eyes = torch.from_numpy(np.asarray(references, dtype=np.float32)[None].copy())
    vectors = torch.from_numpy(np.asarray(modes, dtype=np.float32)[None].copy())
    keep = torch.ones(values.shape[:2], dtype=torch.float32)
    # The frozen model was trained with this feature equal to zero. Do not
    # activate an untrained feature merely because the caller now has anatomy.
    feature_regions = torch.full(values.shape[:2], 3, dtype=torch.long)
    with torch.no_grad():
        estimated = model(values, vectors, eyes, keep, feature_regions)["artifact"].numpy()[0]
    diagnostics = []
    for local, index in enumerate(ids):
        failed = not usable[local] or not np.isfinite(estimated[local]).all()
        if not failed:
            artifact[index] = estimated[local]
        diagnostics.append({"channel": str(names[index]), "failure": "scored_passthrough" if failed else None})
    return np.asarray(eeg, dtype=np.float64) - artifact, artifact, diagnostics


def run_transfer(input_root, output, config):
    parents = list(Path(input_root).rglob("snr_target_protocol.json"))
    if len(parents) != 1:
        raise ValueError("Attach exactly one archived model parent")
    checkpoint = parents[0].parent / "eog_vmd_context_student/best_safe.pt"
    if sha256_file(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Frozen source checkpoint digest differs")
    repository = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(["git", "-C", str(repository), "show",
        LEGACY_SHA+":eog_vmd_fcm_bgru/snr_student.py"])
    path = output / "archived_snr_student.py"
    path.write_bytes(source)
    specification = importlib.util.spec_from_file_location("archived_gain_model", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    model = module.EOGContextGainStudent().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.inference_gate_mode, model.inference_gate_offset = "hard", 0.
    torch.set_num_threads(2)
    parent = selected_parent(input_root)
    selections = {row["fold"]: row for row in read_jsonl(parent / "frontal_fold_selections.jsonl")
                  if row["family"] == "direct"}
    exclusions = {row["fold"]: row for row in read_jsonl(parent / "frontal_detector_folds.jsonl")}
    corpus, manifest = corpus_parent(input_root)
    cache = ArchivedModes(input_root, corpus, config)
    rows = select_examples(manifest, "full")
    scores, failures = [], []
    shards = output / "transfer_predictions"
    shards.mkdir()
    for number, row in enumerate(rows):
        validate_evaluation_sources(row, exclusions[row["partition"]["fold"]])
        data = dict(np.load(corpus / row["array_path"], allow_pickle=False))
        calibration = dict(np.load(corpus / row["calibration"], allow_pickle=False))
        ids = frontal_indices(data["eeg"], data["channel_names"], data["regions"], data["mask"])
        bundles = cache.load(row, data, FixedRecipe(3))
        prediction, artifact, details = apply_frontal(model, data["eeg"], data["references"],
            data["channel_names"], data["regions"], data["mask"], bundles)
        direct = FrontalCorrector(selections[row["partition"]["fold"]]["selected"], config)
        comparator, _, _ = direct.denoise(data["eeg"], data["references"], calibration["references"],
            data["channel_names"], data["regions"], data["mask"], calibration_eeg=calibration["eeg"])
        outside = np.ones(len(data["eeg"]), dtype=bool)
        outside[ids] = False
        if not np.array_equal(prediction[outside], data["eeg"][outside]):
            raise RuntimeError("Frozen transfer changed nonfrontal input")
        target, original = data["paired_reference"][ids], data["eeg"][ids]
        for arm, corrected in (("identity", data["eeg"]), ("direct", comparator), ("frozen_K3_gain_transfer", prediction)):
            paper = paired_channels(corrected[ids], target, original)
            preservation = paired(corrected[ids], target, config["fs"])
            for local, index in enumerate(ids):
                failure = details[local]["failure"] if arm == "frozen_K3_gain_transfer" else None
                scores.append({"example_id": row["example_id"], "recipient": row["recipient"], "donor": row["donor"],
                    "condition": row["condition"], "input_snr_db": row["input_snr_db"], "channel": str(data["channel_names"][index]),
                    "evaluated_family": arm, "failure": failure, "clean_change": float(paper["rrmse_time"][local]),
                    "alpha_db": preservation["alpha_db"], "beta_db": preservation["beta_db"],
                    "covariance_error": preservation["covariance"],
                    **{metric: finite_value(values[local]) for metric, values in paper.items()}})
                if failure:
                    failures.append({"example_id": row["example_id"], "channel": str(data["channel_names"][index]), "failure": failure})
        np.savez_compressed(shards / (row["example_id"]+".npz"), original=original, target=target,
            cleaned=prediction[ids], comparator=comparator[ids], artifact=artifact[ids], names=data["channel_names"][ids])
        atomic_json(output / "frontal_transfer_progress.json", {"completed": number+1, "total": len(rows)})
        if (number+1) % 20 == 0:
            print("FRONTAL_FROZEN_TRANSFER", number+1, len(rows), flush=True)
    frame = pd.DataFrame(scores)
    comparator = frame[frame.evaluated_family == "direct"][["example_id", "channel", "pearson_cc"]].rename(columns={"pearson_cc":"direct_cc"})
    frame = frame.merge(comparator, on=["example_id", "channel"], validate="many_to_one")
    frame["cc_deterioration"] = frame.direct_cc - frame.pearson_cc
    frame.to_csv(output / "frontal_transfer_scores.csv", index=False)
    dirty = frame[frame.condition != "clean"]
    fields = ["snr_energy_db", "snr_improvement_db", "mse", "rrmse_time", "pearson_cc", "cc_deterioration"]
    grouped = dirty.groupby(["evaluated_family", "recipient", "condition", "input_snr_db"])[fields].agg(strict_mean)
    grouped.groupby(["evaluated_family", "recipient"]).agg(strict_mean).groupby("evaluated_family").agg(strict_mean).to_csv(output / "frontal_transfer_means.csv")
    save_controlled_review(frame, output, config)
    atomic_json(output / "frontal_transfer_summary.json", {"scope":"verified_frontal_only",
        "examples": len(rows), "checkpoint_sha256": CHECKPOINT_SHA, "model_source_sha256": sha256_file(path),
        "model_source_git_sha": LEGACY_SHA, "trained_anatomy_feature_disabled": True,
        "failure_rows": len(failures), "K":3, "alpha":2000, "weights_trained":False,
        "autovmd_executed":False, "reserved_confirmation_opened":False,
        "native_validation_required":True, "qualification":False,
        "method":"frozen historical gain-model transfer; explicit frontal-only adaptation"})
    atomic_json(output / "frontal_transfer_failures.json", failures)
