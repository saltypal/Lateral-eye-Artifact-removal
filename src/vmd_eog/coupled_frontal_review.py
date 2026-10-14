"""Fixed frontal contribution check on a verified coupled engineering pilot."""
import json
import time

import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .corpus import unique_parent
from .frontal_calibration_transfer import calibration_transfer
from .frontal_methods import FixedRecipe, decompose_frontal
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired
from .nonfrontal_challengers import balanced_gain
from .nonfrontal_ridge import fit_frozen_ridge
from .paper_metrics import paired_channels
from .reference import lag_matrix


def readiness(input_root, config):
    if (config.get("scope") != "fixed_frontal_coupled_pilot_review"
            or config.get("autovmd_enabled") is not False
            or config.get("fixed_recipes") != [{"K": 3, "alpha": 2000}]):
        raise ValueError("Fixed frontal review policy differs")
    parent = unique_parent(input_root, "coupled_pilot_summary.json").parent
    expected = config["review_parent"]
    if sha256_file(parent / "artifact_manifest.json") != expected["manifest_sha256"]:
        raise ValueError("Coupled pilot manifest differs")
    verify_parent(parent, ("run_spec.json", "coupled_pilot_summary.json", "corpus_manifest.jsonl"))
    spec = json.loads((parent / "run_spec.json").read_text())
    if any(spec.get(key) != expected[key] for key in ("run_id", "campaign_id", "git_sha")):
        raise ValueError("Coupled pilot identity differs")
    summary = json.loads((parent / "coupled_pilot_summary.json").read_text())
    if (summary.get("construction_contracts_passed") is not True or summary.get("examples") != 20
            or summary.get("reserved_confirmation_opened") is not False
            or summary.get("source_split_sha256") != config["source_split_sha256"]):
        raise ValueError("Coupled pilot is not ready")
    return parent


def regional_direct(eeg, references, calibration, regions, mask):
    """Same full-field, region-gain mechanism as the retained direct control."""
    fitted = fit_frozen_ridge(calibration["eeg"], calibration["references"], calibration["boundaries"])
    field = fitted.artifact(references)
    _, valid = lag_matrix(references, fitted.lags)
    artifact = np.zeros_like(eeg, dtype=np.float64)
    details = []
    for region in (1, 2):
        ids = np.flatnonzero(mask & (regions == region))
        if not len(ids):
            continue
        gain, association = balanced_gain(eeg[ids][:, valid], field[ids][:, valid])
        triggered = bool(association >= .4)
        if triggered:
            artifact[ids] = gain * field[ids]
        details.append({"region": region, "gain": gain, "association": association,
                        "triggered": triggered, "paired_target_access": False})
    return artifact, details, fitted


def run(input_root, output, config, profile):
    if profile != "pilot":
        raise ValueError("Two-pair engineering review is not a full selection run")
    parent = readiness(input_root, config)
    rows = read_jsonl(parent / "corpus_manifest.jsonl")
    started = time.perf_counter()
    records, decisions, gate_rows = [], [], []
    for folder in ("predictions", "modes", "fits"):
        (output / folder).mkdir()
    for row in rows:
        verify_parent(parent, (row["array_path"], row["calibration"]))
        with np.load(parent / row["array_path"], allow_pickle=False) as archive:
            stored = dict(archive)
        with np.load(parent / row["calibration"], allow_pickle=False) as archive:
            calibration = dict(archive)
        # Explicit inference boundary: no paired target, true artifact or generator gain.
        data = {key: stored[key] for key in ("eeg", "references", "channel_names", "regions", "mask")}
        eeg, refs = data["eeg"], data["references"]
        names, regions, mask = data["channel_names"], data["regions"], data["mask"]
        bundles = decompose_frontal(eeg, names, regions, FixedRecipe(3), config, mask)
        mode_arrays, mode_info = {}, {}
        for index, (modes, residual, info) in bundles.items():
            np.testing.assert_allclose(modes.sum(0) + residual, eeg[index], rtol=1e-5, atol=1e-12)
            mode_arrays[f"channel_{index}_modes"] = modes
            mode_arrays[f"channel_{index}_residual"] = residual
            mode_info[str(index)] = info
        np.savez_compressed(output / "modes" / (row["example_id"] + ".npz"), **mode_arrays)
        atomic_json(output / "modes" / (row["example_id"] + ".json"),
                    {"input_sha256": sha256_file(parent / row["array_path"]), "K": 3,
                     "alpha": 2000, "channels": mode_info})
        frontal = {}
        detail = {}
        for arm, context in (("raw", None), ("vmd", bundles)):
            _, frontal[arm], detail[arm] = calibration_transfer(eeg, refs, calibration,
                                                              names, regions, mask, bundles=context)
        nonfrontal, detail["nonfrontal"], fitted = regional_direct(eeg, refs, calibration, regions, mask)
        np.savez_compressed(output / "fits" / (row["example_id"] + ".npz"),
                            coefficients=fitted.coefficients, mean=fitted.reference_mean,
                            scale=fitted.reference_scale, active=fitted.active_columns, lags=fitted.lags)
        estimates = {"identity": eeg.copy()}
        for arm, artifact in frontal.items():
            for suffix, support in (("pass", np.zeros_like(nonfrontal)), ("ridge", nonfrontal)):
                if np.any(artifact[regions != 0]) or np.any(support[(regions == 0) | ~mask]):
                    raise RuntimeError("Regional artifact masks overlap or correct unavailable rows")
                estimates[arm + "+" + suffix] = eeg - (artifact + support)
        # Evaluation starts after all predictions exist.
        target = stored["paired_reference"]
        scopes = {"whole": np.flatnonzero(mask)}
        scopes.update({name: np.flatnonzero(mask & (regions == region))
                       for name, region in (("frontal", 0), ("posterior", 1), ("central_shared", 2))})
        for arm, cleaned in estimates.items():
            for scope, ids in scopes.items():
                if not len(ids):
                    continue
                metrics = paired_channels(cleaned[ids], target[ids], eeg[ids])
                engineering = paired(cleaned[ids], target[ids], config["fs"])
                comparator = paired_channels(estimates["raw+ridge"][ids], target[ids])
                cc_loss = float(np.mean(comparator["pearson_cc"] - metrics["pearson_cc"]))
                gate_rows.append({"example_id": row["example_id"], "arm": arm, "scope": scope,
                                  "condition": row["condition"], "cc_deterioration": cc_loss,
                                  "cc_gate": cc_loss <= config["preservation"]["pearson_deterioration"],
                                  "clean_preservation_gate": bool(
                                      engineering["relative_error"] <= config["preservation"]["clean_change"]
                                      and engineering["alpha_db"] <= config["preservation"]["alpha_db"]
                                      and engineering["beta_db"] <= config["preservation"]["beta_db"]
                                      and engineering["covariance"] <= config["preservation"]["covariance"]
                                  ) if row["condition"] == "clean" else None})
                for local, index in enumerate(ids):
                    records.append({**{key: row[key] for key in ("example_id", "recipient", "donor", "condition", "input_snr_db")},
                                    "arm": arm, "scope": scope, "channel": str(names[index]),
                                    **{key: float(value[local]) for key, value in metrics.items()},
                                    "alpha_db": engineering["alpha_db"], "beta_db": engineering["beta_db"],
                                    "covariance": engineering["covariance"], "pooled_snr_db": engineering["snr_db"]})
        np.savez_compressed(output / "predictions" / (row["example_id"] + ".npz"),
                            **estimates, channel_names=names, regions=regions, mask=mask)
        decisions.append({"example_id": row["example_id"], "decisions": detail})
        write_jsonl(output / "decisions.jsonl", decisions)
        write_jsonl(output / "engineering_checks.jsonl", gate_rows)
        pd.DataFrame(records).to_csv(output / "channel_metrics.csv", index=False)
        print("FIXED_FRONTAL_SAVED", row["example_id"], len(decisions), len(rows), flush=True)
    table = pd.DataFrame(records)
    # Keep clean perfect-reconstruction infinity out of contaminated SNR means.
    contaminated = table[table.condition != "clean"]
    grouped = contaminated.groupby(["recipient", "donor", "arm", "scope", "condition", "input_snr_db"]).mean(numeric_only=True).reset_index()
    grouped.to_csv(output / "record_metrics.csv", index=False)
    means = grouped.groupby(["arm", "scope"])[["mse", "rrmse_time", "pearson_cc", "snr_energy_db", "pooled_snr_db"]].mean()
    means.to_csv(output / "method_means.csv")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, scope in zip(axes, ("frontal", "whole")):
        shown = means.xs(scope, level="scope").drop(index="identity")
        positions = np.arange(len(shown))
        axis.bar(positions - .18, shown["snr_energy_db"], .36, label="Channel-first mean")
        axis.bar(positions + .18, shown["pooled_snr_db"], .36, label="Mean pooled-window SNR")
        axis.set_xticks(positions, shown.index, rotation=25, ha="right")
        axis.set_ylabel("Output SNR (dB)")
        axis.set_title(scope.replace("_", " ").title())
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.22), ncol=2, frameon=False)
    fig.suptitle("Coupled engineering pilot: two development pairs, not qualification", y=1.08)
    fig.savefig(output / "fixed_frontal_snr.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    atomic_json(output / "coupled_frontal_summary.json", {"examples": len(rows), "arms": 5,
                "recipient_count": 2, "engineering_pilot_only": True, "teacher_qualified": False,
                "reserved_confirmation_opened": False, "new_fixed_frontal_decompositions": len(rows),
                "runtime_s": time.perf_counter() - started,
                "limitation": "two development pairs; regression-generated fields; no native qualification"})
