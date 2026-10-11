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
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired
from .paper_metrics import paired_channels
from .posterior_context import select_examples


LEGACY_SHA = "0b183e6c09f6ecfbb3ed4818028eb5abbdf842fc"
CHECKPOINT_SHA = "4bb7c1b0cfa0e8341c4053ab3099efb16832f045f99e37afde1904a93c2cd0a9"


def apply_frontal(model, eeg, references, names, regions, mask, bundles, unit_scale=1.):
    """One subtraction from original EEG; paired clean targets are not inputs."""
    if not np.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("Model unit scale must be positive and finite")
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
    values = torch.from_numpy(np.asarray(eeg[ids] * unit_scale, dtype=np.float32)[None].copy())
    eyes = torch.from_numpy(np.asarray(references * unit_scale, dtype=np.float32)[None].copy())
    vectors = torch.from_numpy(np.asarray(modes * unit_scale, dtype=np.float32)[None].copy())
    keep = torch.ones(values.shape[:2], dtype=torch.float32)
    # The frozen model was trained with this feature equal to zero. Do not
    # activate an untrained feature merely because the caller now has anatomy.
    feature_regions = torch.full(values.shape[:2], 3, dtype=torch.long)
    with torch.no_grad():
        estimated = model(values, vectors, eyes, keep, feature_regions)["artifact"].numpy()[0] / unit_scale
    diagnostics = []
    for local, index in enumerate(ids):
        failed = not usable[local] or not np.isfinite(estimated[local]).all()
        if not failed:
            artifact[index] = estimated[local]
        diagnostics.append({"channel": str(names[index]), "failure": "scored_passthrough" if failed else None})
    return np.asarray(eeg, dtype=np.float64) - artifact, artifact, diagnostics


def gain_adapter(model_class, axis_gate=False, anchored=False):
    """Frozen-weight adaptations; calibration references never include targets."""
    class Adapter(model_class):
        calibration_mean = None

        def correction_gains(self, features, correlations, mask, regions):
            gains = super().correction_gains(features, correlations, mask, regions)
            if axis_gate:
                # Rows here are already verified frontal. Signed reference
                # coefficients remain intact; only intervention support differs.
                evidence = correlations[:, :, 0].abs().masked_fill(~mask.bool()[..., None], 0).amax(dim=1)
                threshold = .1 + .8 * torch.sigmoid(self.threshold_logit)
                gains = gains * (evidence >= threshold)[:, None].to(gains.dtype)
            self.latest_gains = gains
            return gains

        def forward(self, eeg, modes, references, mask, regions):
            result = super().forward(eeg, modes, references, mask, regions)
            if anchored:
                if self.calibration_mean is None:
                    raise ValueError("Mean anchoring requires disjoint calibration references")
                mean = references.mean(dim=-1, keepdim=True)
                eyes = references - mean
                scale = eyes.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
                eyes = eyes / scale
                values = eeg - eeg.mean(dim=-1, keepdim=True)
                length = eeg.shape[-1]
                gram = eyes @ eyes.transpose(1,2) / length
                regularized = gram + 1e-4 * torch.eye(2, device=eeg.device, dtype=eeg.dtype)[None]
                coefficients = torch.linalg.solve(regularized, eyes @ values.transpose(1,2) / length).transpose(1,2)
                offset = ((mean - self.calibration_mean) / scale).squeeze(-1)
                bias = (coefficients * self.latest_gains * offset[:,None]).sum(-1,keepdim=True)
                bias = torch.where(mask.bool()[...,None],bias,torch.zeros_like(bias))
                artifact = result["artifact"] + bias
                result = {"artifact":artifact, "cleaned":eeg-artifact}
            return result
    return Adapter()


def run_transfer(input_root, output, config, hybrid=False, unit_audit=False, calibration_hybrid=False, axis_hybrid=False, event_hybrid=False,event_gain_scaled=False,monotonic_events=False):
    parents = list(Path(input_root).rglob("snr_target_protocol.json"))
    if len(parents) != 1:
        raise ValueError("Attach exactly one archived model parent")
    checkpoint = parents[0].parent / "eog_vmd_context_student/best_safe.pt"
    if sha256_file(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Frozen source checkpoint digest differs")
    repository = Path(__file__).resolve().parents[2]
    # Campaign notebooks intentionally shallow-clone the current branch. The
    # unrelated archived model commit must be fetched explicitly; HEAD stays
    # pinned to the experiment source throughout.
    subprocess.check_call(["git", "-C", str(repository), "fetch", "--quiet", "--depth", "1", "origin", LEGACY_SHA])
    resolved = subprocess.check_output(["git", "-C", str(repository), "rev-parse", LEGACY_SHA+"^{commit}"], text=True).strip()
    if resolved != LEGACY_SHA:
        raise ValueError("Archived model source identity differs")
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
    models = {"frozen_K3_gain_transfer": model}
    unit_scales = {"frozen_K3_gain_transfer": 1.}
    if hybrid:
        for name, axis, anchored in (("K3_axis_gate",True,False),
                ("K3_calibration_anchor",False,True), ("K3_axis_gate_anchor",True,True)):
            adapted = gain_adapter(module.EOGContextGainStudent, axis, anchored).eval()
            adapted.load_state_dict(state["model"])
            adapted.inference_gate_mode, adapted.inference_gate_offset = "hard", 0.
            models[name] = adapted
            unit_scales[name] = 1.
    if unit_audit:
        # Corpus amplitudes are verified MNE volts. The archived model uses
        # absolute floors originally exercised on much larger numeric inputs.
        # Convert every input together, then restore the artifact to volts.
        for name, axis, anchored in (("K3_microvolt_transfer",False,False),
                ("K3_microvolt_anchor",False,True)):
            adapted = gain_adapter(module.EOGContextGainStudent, axis, anchored).eval()
            adapted.load_state_dict(state["model"])
            adapted.inference_gate_mode, adapted.inference_gate_offset = "hard", 0.
            models[name] = adapted
            unit_scales[name] = 1e6
    torch.set_num_threads(2)
    parent = selected_parent(input_root)
    selections = {row["fold"]: row for row in read_jsonl(parent / "frontal_fold_selections.jsonl")
                  if row["family"] == "direct"}
    exclusions = {row["fold"]: row for row in read_jsonl(parent / "frontal_detector_folds.jsonl")}
    corpus, manifest = corpus_parent(input_root)
    cache = ArchivedModes(input_root, corpus, config)
    rows = select_examples(manifest, "full")
    scores, failures = [], []
    event_gates = {}
    shards = output / "transfer_predictions"
    shards.mkdir()
    decisions = output / "transfer_decisions"
    decisions.mkdir()
    for number, row in enumerate(rows):
        validate_evaluation_sources(row, exclusions[row["partition"]["fold"]])
        data = dict(np.load(corpus / row["array_path"], allow_pickle=False))
        calibration = dict(np.load(corpus / row["calibration"], allow_pickle=False))
        ids = frontal_indices(data["eeg"], data["channel_names"], data["regions"], data["mask"])
        bundles = cache.load(row, data, FixedRecipe(3))
        estimates, arm_details, artifacts = {}, {}, {}
        for name, network in models.items():
            if hasattr(network,"calibration_mean"):
                network.calibration_mean = torch.from_numpy(np.asarray(
                    calibration["references"].mean(-1)*unit_scales[name],dtype=np.float32)[None,:,None].copy())
            estimates[name], artifacts[name], arm_details[name] = apply_frontal(network, data["eeg"], data["references"],
                data["channel_names"], data["regions"], data["mask"], bundles, unit_scale=unit_scales[name])
        if calibration_hybrid:
            from .frontal_calibration_transfer import calibration_transfer
            for name,mode_context in (("calibration_raw_gain",None),("K3_calibration_mode_gain",bundles)):
                estimates[name],artifacts[name],arm_details[name] = calibration_transfer(
                    data["eeg"],data["references"],calibration,data["channel_names"],
                    data["regions"],data["mask"],bundles=mode_context)
            if event_hybrid:
                from .frontal_event_support import fit_event_support,fit_axis_amplitude_support,gate_artifact
                key = row["calibration"]
                if key not in event_gates:
                    try:
                        event_gates[key] = (fit_axis_amplitude_support(calibration) if monotonic_events
                                            else fit_event_support(calibration))
                    except ValueError as error:
                        event_gates[key] = str(error)
                gate = event_gates[key]
                for base,name in (("calibration_raw_gain","calibration_raw_event_gain"),
                                  ("K3_calibration_mode_gain","K3_calibration_event_gain")):
                    if isinstance(gate,str):
                        estimates[name] = np.asarray(data["eeg"],dtype=float).copy()
                        artifacts[name] = np.zeros_like(estimates[name])
                        arm_details[name] = [{"channel":str(data["channel_names"][index]),
                            "failure":"event_calibration: "+gate} for index in ids]
                    else:
                        estimates[name],artifacts[name],arm_details[name] = gate_artifact(
                            data["eeg"],artifacts[base],data["references"],gate,arm_details[base])
                if event_gain_scaled:
                    for base,name in (("calibration_raw_gain","calibration_raw_scaled_event_gain"),
                                      ("K3_calibration_mode_gain","K3_calibration_scaled_event_gain")):
                        if isinstance(gate,str):
                            estimates[name] = np.asarray(data["eeg"],dtype=float).copy()
                            artifacts[name] = np.zeros_like(estimates[name])
                            arm_details[name] = [{"channel":str(data["channel_names"][index]),
                                "failure":"event_calibration: "+gate} for index in ids]
                        else:
                            estimates[name],artifacts[name],arm_details[name] = gate_artifact(
                                data["eeg"],artifacts[base],data["references"],gate,arm_details[base],gain_scaled=True)
            if axis_hybrid:
                for name,mode_context in (("calibration_raw_axis_gain",None),("K3_calibration_axis_gain",bundles)):
                    estimates[name],artifacts[name],arm_details[name] = calibration_transfer(
                        data["eeg"],data["references"],calibration,data["channel_names"],
                        data["regions"],data["mask"],bundles=mode_context,gain_mode="axis")
        direct = FrontalCorrector(selections[row["partition"]["fold"]]["selected"], config)
        comparator, _, _ = direct.denoise(data["eeg"], data["references"], calibration["references"],
            data["channel_names"], data["regions"], data["mask"], calibration_eeg=calibration["eeg"])
        outside = np.ones(len(data["eeg"]), dtype=bool)
        outside[ids] = False
        if any(not np.array_equal(prediction[outside], data["eeg"][outside]) for prediction in estimates.values()):
            raise RuntimeError("Frozen transfer changed nonfrontal input")
        target, original = data["paired_reference"][ids], data["eeg"][ids]
        for arm, corrected in {"identity": data["eeg"], "direct": comparator, **estimates}.items():
            paper = paired_channels(corrected[ids], target, original)
            preservation = paired(corrected[ids], target, config["fs"])
            for local, index in enumerate(ids):
                failure = arm_details[arm][local]["failure"] if arm in arm_details else None
                scores.append({"example_id": row["example_id"], "recipient": row["recipient"], "donor": row["donor"],
                    "condition": row["condition"], "input_snr_db": row["input_snr_db"], "channel": str(data["channel_names"][index]),
                    "evaluated_family": arm, "failure": failure, "clean_change": float(paper["rrmse_time"][local]),
                    "alpha_db": preservation["alpha_db"], "beta_db": preservation["beta_db"],
                    "covariance_error": preservation["covariance"],
                    **{metric: finite_value(values[local]) for metric, values in paper.items()}})
                if failure:
                    failures.append({"example_id": row["example_id"], "arm":arm, "channel": str(data["channel_names"][index]), "failure": failure})
        write_jsonl(decisions / (row["example_id"]+".jsonl"),[
            {"example_id":row["example_id"],"arm":arm,"decisions":details}
            for arm,details in arm_details.items()])
        np.savez_compressed(shards / (row["example_id"]+".npz"), original=original, target=target,
            comparator=comparator[ids], names=data["channel_names"][ids],
            **{name:values[ids] for name,values in estimates.items()},
            **{"artifact_"+name:values[ids] for name,values in artifacts.items()})
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
        "predeclared_arms":list(estimates), "hybrid_ablation":hybrid or calibration_hybrid,
        "calibration_hybrid":calibration_hybrid, "axis_hybrid":axis_hybrid,"event_hybrid":event_hybrid,
        "event_gain_scaled":event_gain_scaled,"monotonic_events":monotonic_events,
        "mean_anchor_source":"unscored calibration EOG only",
        "unit_audit":unit_audit, "model_input_scales":unit_scales, "output_units":"volts",
        "autovmd_executed":False, "reserved_confirmation_opened":False,
        "native_validation_required":True, "qualification":False,
        "method":"fixed signed calibration-field hybrid with frozen transfer control" if calibration_hybrid
            else "frozen historical gain-model transfer; explicit frontal-only adaptation"})
    atomic_json(output / "frontal_transfer_failures.json", failures)
