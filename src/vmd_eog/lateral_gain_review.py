"""Matched-subspace and signed-axis repairs using immutable frontal VMD caches."""
import json
import time

import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .corpus import unique_parent
from .coupled_frontal_review import validate_mode_reconstruction
from .frontal_calibration_transfer import calibration_transfer
from .frontal_methods import frontal_indices
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired
from .paper_metrics import paired_channels
from .reference import correlations


ARMS = ("identity", "raw_shared", "vmd_shared", "vmd_projection", "raw_axis", "vmd_projected_axis")


def readiness(input_root, config):
    if (config.get("scope") != "bounded_lateral_gain_repair"
            or config.get("autovmd_enabled") is not False
            or config.get("fixed_recipes") != [{"K": 3, "alpha": 2000}]
            or config.get("reserved_access") != "closed until verified freeze"):
        raise ValueError("Lateral gain repair policy differs")
    parents = {}
    for name, summary_name in (("corpus", "coupled_pilot_summary.json"),
                                ("review", "coupled_frontal_summary.json")):
        parent = unique_parent(input_root, summary_name).parent
        expected = config["repair_parents"][name]
        if sha256_file(parent / "artifact_manifest.json") != expected["manifest_sha256"]:
            raise ValueError("Lateral gain parent manifest differs: " + name)
        verify_parent(parent, ("run_spec.json", summary_name, "stage_tests_summary.json"))
        spec = json.loads((parent / "run_spec.json").read_text())
        if any(spec.get(key) != expected[key] for key in ("run_id", "campaign_id", "git_sha")):
            raise ValueError("Lateral gain parent identity differs: " + name)
        summary = json.loads((parent / summary_name).read_text())
        if (summary.get("examples") != 20 or summary.get("reserved_confirmation_opened") is not False
                or json.loads((parent / "stage_tests_summary.json").read_text()).get("passed") is not True):
            raise ValueError("Lateral gain parent did not pass its engineering checkpoint")
        parents[name] = parent
    if json.loads((parents["corpus"] / "coupled_pilot_summary.json").read_text())["source_split_sha256"] != config["source_split_sha256"]:
        raise ValueError("Lateral gain source split differs")
    return parents["corpus"], parents["review"]


def run(input_root, output, config, profile):
    if profile != "pilot":
        raise ValueError("Lateral gain repair is a bounded development pilot")
    corpus, review = readiness(input_root, config)
    verify_parent(corpus, ("corpus_manifest.jsonl",))
    rows = read_jsonl(corpus / "corpus_manifest.jsonl")
    if len(rows) != 20 or any(row["partition"]["role"] != "development" for row in rows):
        raise ValueError("Unexpected or reserved lateral repair cohort")
    started = time.perf_counter()
    records, decisions, gates, cache_checks = [], [], [], []
    (output / "predictions").mkdir()
    for row in rows:
        verify_parent(corpus, (row["array_path"], row["calibration"]))
        paths = tuple(folder + "/" + row["example_id"] + suffix
                      for folder, suffix in (("modes", ".npz"), ("modes", ".json"), ("predictions", ".npz")))
        verify_parent(review, paths)
        with np.load(corpus / row["array_path"], allow_pickle=False) as archive:
            stored = dict(archive)
        with np.load(corpus / row["calibration"], allow_pickle=False) as archive:
            calibration = dict(archive)
        with np.load(review / paths[0], allow_pickle=False) as archive:
            cache = dict(archive)
        mode_info = json.loads((review / paths[1]).read_text())
        if (mode_info["input_sha256"] != sha256_file(corpus / row["array_path"])
                or mode_info["K"] != 3 or mode_info["alpha"] != 2000):
            raise ValueError("Frontal cache input or fixed recipe differs")
        data = {key: stored[key] for key in ("eeg", "references", "channel_names", "regions", "mask")}
        original, refs = data["eeg"], data["references"]
        regions, mask, names = data["regions"], data["mask"], data["channel_names"]
        ids = frontal_indices(original, names, regions, mask)
        if set(mode_info["channels"]) != {str(index) for index in ids}:
            raise ValueError("Frontal cache channels differ")
        bundles, diagnostics = {}, []
        for index in ids:
            modes, residual = cache[f"channel_{index}_modes"], cache[f"channel_{index}_residual"]
            if modes.shape != (3, original.shape[-1]):
                raise ValueError("Fixed frontal cache shape differs")
            reconstruction = validate_mode_reconstruction(modes, residual, original[index])
            bundles[int(index)] = modes, residual, mode_info["channels"][str(index)]
            diagnostics.append({"channel": str(names[index]), "reconstruction": reconstruction,
                                "mode_reference_cc": correlations(modes, refs, 20).tolist(),
                                "residual_reference_cc": correlations(residual[None], refs, 20).tolist(),
                                "residual_energy": float(np.sum(residual.astype(np.float64)**2))})
        with np.load(review / paths[2], allow_pickle=False) as archive:
            saved = dict(archive)
        for key in ("channel_names", "regions", "mask"):
            np.testing.assert_array_equal(saved[key], data[key])
        np.testing.assert_array_equal(saved["identity"], original)
        nonfrontal = mask & np.isin(regions, (1, 2))
        np.testing.assert_array_equal(saved["raw+ridge"][nonfrontal], saved["vmd+ridge"][nonfrontal])
        estimates = {"identity": saved["identity"], "raw_shared": saved["raw+ridge"],
                     "vmd_shared": saved["vmd+ridge"]}
        detail = {}
        for arm, gain_mode, context in (("vmd_projection", "mode_projection", bundles),
                                        ("raw_axis", "axis", None),
                                        ("vmd_projected_axis", "projected_axis", bundles)):
            cleaned, artifact, detail[arm] = calibration_transfer(original, refs, calibration,
                names, regions, mask, bundles=context, gain_mode=gain_mode)
            if np.any(artifact[(regions != 0) | ~mask]):
                raise RuntimeError("Frontal repair modified a nonfrontal or masked row")
            cleaned[nonfrontal] = saved["raw+ridge"][nonfrontal]
            np.testing.assert_array_equal(cleaned[nonfrontal], saved["raw+ridge"][nonfrontal])
            estimates[arm] = cleaned
        # Labels and paired truth are evaluator-only, after all predictions exist.
        target = stored["paired_reference"]
        scopes = {"whole": np.flatnonzero(mask), "frontal": ids,
                  "posterior": np.flatnonzero(mask & (regions == 1)),
                  "central_shared": np.flatnonzero(mask & (regions == 2))}
        for arm, cleaned in estimates.items():
            for scope, scope_ids in scopes.items():
                if not len(scope_ids):
                    continue
                metrics = paired_channels(cleaned[scope_ids], target[scope_ids], original[scope_ids])
                engineering = paired(cleaned[scope_ids], target[scope_ids], config["fs"])
                direct = paired_channels(estimates["raw_shared"][scope_ids], target[scope_ids])
                axis_direct = paired_channels(estimates["raw_axis"][scope_ids], target[scope_ids])
                cc_loss = float(np.mean(direct["pearson_cc"] - metrics["pearson_cc"]))
                axis_cc_loss = float(np.mean(axis_direct["pearson_cc"] - metrics["pearson_cc"]))
                gates.append({"example_id": row["example_id"], "arm": arm, "scope": scope,
                              "condition": row["condition"], "cc_deterioration_vs_shared": cc_loss,
                              "cc_deterioration_vs_axis": axis_cc_loss,
                              "cc_gate_shared": cc_loss <= .005, "cc_gate_axis": axis_cc_loss <= .005,
                              "clean_checks": {"modification": engineering["relative_error"],
                                               "alpha_db": engineering["alpha_db"], "beta_db": engineering["beta_db"],
                                               "covariance": engineering["covariance"]} if row["condition"] == "clean" else None})
                for local, index in enumerate(scope_ids):
                    records.append({**{key: row[key] for key in ("example_id", "recipient", "donor", "condition", "input_snr_db")},
                                    "arm": arm, "scope": scope, "channel": str(names[index]),
                                    **{key: float(value[local]) for key, value in metrics.items()},
                                    "pooled_snr_db": engineering["snr_db"]})
        np.savez_compressed(output / "predictions" / (row["example_id"] + ".npz"),
                            **estimates, channel_names=names, regions=regions, mask=mask)
        decisions.append({"example_id": row["example_id"], "new_decisions": detail,
                          "mode_diagnostics": diagnostics})
        cache_checks.append({"example_id": row["example_id"], "input_hash_verified": True,
                             "nonfrontal_parity": True, "new_vmd_decompositions": 0})
        write_jsonl(output / "decisions.jsonl", decisions)
        write_jsonl(output / "engineering_checks.jsonl", gates)
        write_jsonl(output / "cache_checks.jsonl", cache_checks)
        pd.DataFrame(records).to_csv(output / "channel_metrics.csv", index=False)
        print("LATERAL_GAIN_SAVED", len(decisions), len(rows), row["example_id"], flush=True)
    table = pd.DataFrame(records)
    contaminated = table[table.condition != "clean"]
    grouped = contaminated.groupby(["recipient", "donor", "arm", "scope", "condition", "input_snr_db"]).mean(numeric_only=True).reset_index()
    grouped.to_csv(output / "record_metrics.csv", index=False)
    columns = ["mse", "rrmse_time", "pearson_cc", "snr_energy_db", "pooled_snr_db"]
    grouped.groupby(["arm", "scope"])[columns].mean().to_csv(output / "method_means.csv")
    grouped.groupby(["arm", "scope", "condition", "input_snr_db"])[columns].mean().to_csv(output / "condition_means.csv")
    plot_conditions(grouped, output)
    atomic_json(output / "lateral_gain_summary.json", {"examples": len(rows), "arms": list(ARMS),
                "new_vmd_decompositions": 0, "nonfrontal_outputs_frozen": True,
                "teacher_qualified": False, "reserved_confirmation_opened": False,
                "runtime_s": time.perf_counter() - started, "engineering_pilot_only": True})


def plot_conditions(records, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.subplots_adjust(top=.76, bottom=.16, wspace=.28)
    shown = records[(records.scope == "frontal") & (records.arm != "identity")]
    for axis, condition in zip(axes, ("blink", "lateral", "mixed")):
        subset = shown[shown.condition == condition]
        for arm in ARMS[1:]:
            points = subset[subset.arm == arm].groupby("input_snr_db")["pooled_snr_db"].mean()
            axis.plot(points.index, points.values, marker="o", label=arm.replace("_", " "))
        axis.set_title(condition.title())
        axis.set_xlabel("Nominal whole-montage input SNR (dB)")
        axis.set_xticks([-5, 0, 5])
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Frontal mean pooled-window output SNR (dB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    fig.suptitle("Lateral gain repair: same frontal modes and nonfrontal outputs; two development pairs", y=.995)
    fig.savefig(output / "lateral_gain_conditions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
