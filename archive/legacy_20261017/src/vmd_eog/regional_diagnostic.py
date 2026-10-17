"""Attribute frozen development errors; no fitting, selection or correction."""
import json
import time

import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .frontal_methods import frontal_indices
from .frontal_review import strict_mean
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .paper_metrics import paired_channels
from .posterior_context import PAPER_KEYS
from .regional_integration import unique_artifact


GROUPS = {0: "frontal", 1: "posterior", 2: "central_shared", 3: "unknown"}
ENERGIES = ("target_energy", "input_error_energy", "output_error_energy", "modification_energy")


def readiness(input_root, config):
    if config.get("schema_version") != 4 or config.get("autovmd_enabled") is not False:
        raise ValueError("Diagnostic requires schema4 and stopped AutoVMD")
    if config.get("scope") != "frozen_regional_error_diagnostic":
        raise ValueError("Diagnostic scope differs")
    if config.get("reserved_access") != "closed until verified freeze":
        raise ValueError("Reserved access differs")
    parent = unique_artifact(input_root, "regional_summary.json").parent
    expected = config["diagnostic_parent"]
    if sha256_file(parent / "artifact_manifest.json") != expected["manifest_sha256"]:
        raise ValueError("Diagnostic parent manifest differs")
    verify_parent(parent, ("regional_summary.json", "run_spec.json", "configuration.json",
                           "regional_selection_manifest.jsonl", "regional_channel_scores.csv.gz",
                           "regional_example_scores.csv", "regional_method_means.csv"))
    spec = json.loads((parent / "run_spec.json").read_text())
    for key in ("run_id", "git_sha", "campaign_id"):
        if spec.get(key) != expected[key]:
            raise ValueError("Diagnostic parent identity differs: " + key)
    summary = json.loads((parent / "regional_summary.json").read_text())
    if summary.get("reserved_confirmation_opened") is not False or summary.get("autovmd_executed") is not False:
        raise ValueError("Parent scope/confirmation contract differs")
    if summary.get("source_split_sha256") != expected["split_sha256"]:
        raise ValueError("Frozen split identity differs")
    if summary.get("VMD_scope") != "verified frontal K3/2000 caches only":
        raise ValueError("Frontal recipe scope differs")
    return parent, summary


def validate_rows(rows, expected_examples, expected_recipients):
    if len(rows) != expected_examples or len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("Missing or duplicate example identities")
    if len({row["recipient"] for row in rows}) != expected_recipients:
        raise ValueError("Recipient coverage differs")
    for row in rows:
        if row["partition"].get("role") != "development" or row["partition"].get("fold") is None:
            raise ValueError("Reserved or ungrouped source in diagnostic")


def groups_for(saved):
    original, target = np.asarray(saved["original"]), np.asarray(saved["target"])
    names, mask, regions = (np.asarray(saved[key]) for key in ("names", "mask", "regions"))
    if original.ndim != 2 or target.shape != original.shape or not np.isfinite(original).all() or not np.isfinite(target).all():
        raise ValueError("Invalid finite prediction input/target")
    count = len(original)
    if names.shape != (count,) or mask.shape != (count,) or regions.shape != (count,) or mask.dtype.kind != "b":
        raise ValueError("Channel metadata axes or mask type differ")
    if len(set(names.tolist())) != count or not np.isin(regions, list(GROUPS)).all():
        raise ValueError("Duplicate names or invalid region codes")
    available = np.flatnonzero(mask)
    if not len(available):
        raise ValueError("No available channels")
    groups = {name: np.flatnonzero(mask & (regions == code)) for code, name in GROUPS.items()}
    if not np.array_equal(frontal_indices(original, names, regions, mask), groups["frontal"]):
        raise ValueError("Frontal whitelist/region disagreement")
    joined = np.concatenate(list(groups.values()))
    if len(np.unique(joined)) != len(available) or not np.array_equal(np.sort(joined), available):
        raise ValueError("Region partition coverage or overlap differs")
    return available, groups


def energy_snr(target_energy, error_energy):
    """Engineering policy: undefined perfect/zero energy cases remain explicit."""
    if target_energy <= 1e-24:
        return np.nan, "zero_target_energy"
    if error_energy == 0:
        return np.nan, "perfect_reconstruction"
    return float(10 * np.log10(target_energy / error_energy)), "finite"


def energy_rows(original, target, corrected, available, groups):
    corrected = np.asarray(corrected)
    if corrected.shape != original.shape or not np.isfinite(corrected).all():
        raise ValueError("Invalid corrected waveform")
    for name in ("central_shared", "unknown"):
        if not np.array_equal(corrected[groups[name]], original[groups[name]]):
            raise ValueError("Frozen pass-through group was modified: " + name)
    channel_values = {
        "target_energy": np.sum(target.astype(float) ** 2, axis=-1),
        "input_error_energy": np.sum((original.astype(float) - target) ** 2, axis=-1),
        "output_error_energy": np.sum((corrected.astype(float) - target) ** 2, axis=-1),
        "modification_energy": np.sum((corrected.astype(float) - original) ** 2, axis=-1),
    }
    whole_error = float(channel_values["output_error_energy"][available].sum())
    rows = []
    for name, ids in {**groups, "whole": available}.items():
        values = {key: float(channel_values[key][ids].sum()) for key in ENERGIES}
        snr, status = energy_snr(values["target_energy"], values["output_error_energy"])
        input_snr, input_status = energy_snr(values["target_energy"], values["input_error_energy"])
        rows.append({"group": name, "channels": len(ids), "samples": len(ids) * target.shape[-1],
                     **values, "pooled_energy_snr_db": snr, "snr_status": status,
                     "pooled_input_snr_db": input_snr, "input_snr_status": input_status,
                     "pooled_snr_improvement_db": snr - input_snr,
                     "output_error_share": values["output_error_energy"] / whole_error if whole_error else np.nan})
    for key in ENERGIES:
        if not np.isclose(sum(row[key] for row in rows[:-1]), rows[-1][key], rtol=1e-12, atol=1e-24):
            raise ValueError("Regional energy conservation failed: " + key)
    return rows


def assert_parity(actual, expected, metric):
    atol = 1e-22 if metric == "mse" else 1e-7
    if not np.allclose(actual, expected, rtol=1e-7, atol=atol, equal_nan=True):
        raise ValueError("Archived metric parity failed: " + metric)


def balanced_means(frame, keys, fields):
    dirty = frame[frame.condition != "clean"]
    if dirty[keys + ["recipient", "condition", "input_snr_db"]].isna().any().any():
        raise ValueError("Undefined dirty grouping key cannot reproduce archived means")
    strata = dirty.groupby(keys + ["recipient", "condition", "input_snr_db"], dropna=False)[fields].agg(strict_mean)
    recipients = strata.groupby(keys + ["recipient"], dropna=False)[fields].agg(strict_mean)
    return recipients.groupby(keys, dropna=False)[fields].agg(strict_mean).reset_index(), recipients.reset_index()


def plot_results(output):
    import matplotlib.pyplot as plt
    summary = pd.read_csv(output / "diagnostic_group_means.csv")
    shares = pd.read_csv(output / "diagnostic_energy_means.csv")
    table = shares[(shares.arm == "K3_calibration_mode_gain+direct:posterior") & (shares.group != "whole")]
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(table.group, table.output_error_share, color="#2364aa")
    axis.set_ylabel("Mean fraction of whole-cap reconstruction error")
    axis.set_title("Frozen VMD + posterior regression: development error attribution")
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(output / "diagnostic_error_shares.png", dpi=150)
    plt.close(figure)
    whole = summary[summary.group == "whole"]
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.barh(whole.arm, whole.snr_energy_db, color="#2364aa")
    axis.axvline(15, color="black", linestyle="--", label="15 dB project target")
    axis.set_xlabel("Channel-first recipient-balanced mean output SNR (dB)")
    axis.set_title("Frozen development controls; no new correction or tuning")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "diagnostic_whole_snr.png", dpi=150)
    plt.close(figure)


def run(input_root, output, config, profile):
    started = time.perf_counter()
    parent, summary = readiness(input_root, config)
    rows = read_jsonl(parent / "regional_selection_manifest.jsonl")
    validate_rows(rows, config["diagnostic_parent"]["examples"], config["diagnostic_parent"]["recipients"])
    if profile == "pilot":
        recipients = {}
        for row in rows:
            recipients.setdefault(row["partition"]["fold"], row["recipient"])
        rows = [row for row in rows if row["recipient"] == recipients[row["partition"]["fold"]]]
    arms, row_ids = config["diagnostic_arms"], {row["example_id"] for row in rows}
    chunks = []
    for chunk in pd.read_csv(parent / "regional_channel_scores.csv.gz", chunksize=100000):
        chunks.append(chunk[(chunk.scope == "whole") & chunk.evaluated_family.isin(arms) & chunk.example_id.isin(row_ids)])
    old = pd.concat(chunks).set_index(["example_id", "evaluated_family", "channel"], verify_integrity=True)
    previous_examples = pd.read_csv(parent / "regional_example_scores.csv")
    previous_examples = previous_examples[(previous_examples.scope == "whole") & previous_examples.arm.isin(arms) & previous_examples.example_id.isin(row_ids)]
    previous_examples = previous_examples.set_index(["example_id", "arm"], verify_integrity=True)
    channel_rows, energies, coverage, failures = [], [], [], []
    channel_checks = pooled_checks = 0
    write_jsonl(output / "diagnostic_selection_manifest.jsonl", rows)
    for number, row in enumerate(rows):
        path = parent / "regional_predictions" / (row["example_id"] + ".npz")
        verify_parent(parent, (str(path.relative_to(parent)),))
        with np.load(path, allow_pickle=False) as archive:
            available, groups = groups_for(archive)
            original, target, names, regions = (archive[key] for key in ("original", "target", "names", "regions"))
            if not np.array_equal(archive["identity"], original):
                raise ValueError("Identity arm modified original")
            meta = {key: row[key] for key in ("example_id", "recipient", "donor", "condition", "input_snr_db")}
            meta["fold"] = row["partition"]["fold"]
            for index in range(len(names)):
                coverage.append({**meta, "channel": str(names[index]), "region_code": int(regions[index]),
                                 "group": GROUPS[int(regions[index])], "available": bool(archive["mask"][index])})
            comparator = paired_channels(archive[summary["comparator"]][available], target[available], original[available])
            for arm in arms:
                if arm not in archive.files:
                    raise ValueError("Missing frozen arm: " + arm)
                corrected = archive[arm]
                paper = paired_channels(corrected[available], target[available], original[available])
                paper["cc_deterioration"] = comparator["pearson_cc"] - paper["pearson_cc"]
                labels = [(row["example_id"], arm, str(names[index])) for index in available]
                previous = old.loc[labels]
                for metric in list(PAPER_KEYS) + ["cc_deterioration"]:
                    assert_parity(paper[metric], previous[metric].to_numpy(), metric)
                    channel_checks += len(available)
                if previous.failure.notna().any():
                    failures.append({**meta, "arm": arm, "inherited_failure": str(previous.failure.iloc[0]),
                                     "note": "Inherited example label, not a unique decomposition failure count"})
                for local, index in enumerate(available):
                    channel_rows.append({**meta, "arm": arm, "channel": str(names[index]),
                                         "group": GROUPS[int(regions[index])],
                                         **{metric: float(paper[metric][local]) for metric in list(PAPER_KEYS) + ["cc_deterioration"]}})
                regional = energy_rows(original, target, corrected, available, groups)
                assert_parity(regional[-1]["pooled_energy_snr_db"], previous_examples.loc[(row["example_id"], arm), "pooled_energy_snr_db"], "pooled_energy_snr_db")
                pooled_checks += 1
                energies.extend({**meta, "arm": arm, **item} for item in regional)
        atomic_json(output / "diagnostic_progress.json", {"completed_examples": number + 1, "examples": len(rows)})
        print("DIAGNOSTIC_SAVED", number + 1, len(rows), flush=True)
    frame = pd.DataFrame(channel_rows)
    if len(frame) != len(old):
        raise ValueError("Archived channel score coverage differs")
    frame.to_csv(output / "diagnostic_channel_scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame(coverage).to_csv(output / "diagnostic_coverage.csv", index=False)
    pd.DataFrame(energies).to_csv(output / "diagnostic_example_energies.csv", index=False)
    write_jsonl(output / "diagnostic_inherited_failures.jsonl", failures)
    fields = list(PAPER_KEYS) + ["cc_deterioration"]
    scopes = pd.concat([frame, frame.assign(group="whole")], ignore_index=True)
    means, recipients = balanced_means(scopes, ["group", "arm"], fields)
    means.to_csv(output / "diagnostic_group_means.csv", index=False)
    recipients.to_csv(output / "diagnostic_recipient_means.csv", index=False)
    channel_means, _ = balanced_means(frame, ["group", "arm", "channel"], fields)
    channel_means.to_csv(output / "diagnostic_channel_means.csv", index=False)
    condition_means = scopes[scopes.condition != "clean"].groupby(
        ["group", "arm", "recipient", "condition", "input_snr_db"])[fields].agg(strict_mean)
    condition_means.groupby(["group", "arm", "condition", "input_snr_db"]).agg(strict_mean).to_csv(
        output / "diagnostic_condition_means.csv")
    energy_fields = list(ENERGIES) + ["output_error_share", "pooled_energy_snr_db", "pooled_snr_improvement_db"]
    energy_means, _ = balanced_means(pd.DataFrame(energies), ["group", "arm"], energy_fields)
    energy_means.to_csv(output / "diagnostic_energy_means.csv", index=False)
    global_energy = pd.DataFrame(energies).query("condition != 'clean'").groupby(["group", "arm"])[list(ENERGIES)].sum()
    for key, group in global_energy.iterrows():
        snr, status = energy_snr(group.target_energy, group.output_error_energy)
        global_energy.loc[key, "pooled_energy_snr_db"] = snr
        global_energy.loc[key, "snr_status"] = status
    global_energy.to_csv(output / "diagnostic_global_pooled_energies.csv")
    mean_checks = 0
    if profile == "full":
        old_means = pd.read_csv(parent / "regional_method_means.csv").set_index(["scope", "evaluated_family"])
        for record in means[means.group.isin(["whole", "frontal", "posterior"])].to_dict("records"):
            previous = old_means.loc[(record["group"], record["arm"])]
            for metric in fields:
                assert_parity(record[metric], previous[metric], metric)
                mean_checks += 1
    atomic_json(output / "diagnostic_summary.json", {"examples": len(rows), "recipients": int(frame.recipient.nunique()),
        "arms": len(arms), "profile": profile, "channel_metric_parity_checks": channel_checks,
        "whole_pooled_parity_checks": pooled_checks, "archived_balanced_mean_checks": mean_checks,
        "parent_run": config["diagnostic_parent"], "groups": GROUPS, "runtime_s": time.perf_counter() - started,
        "source_split_sha256": summary["source_split_sha256"], "weights_trained": False,
        "correction_changed": False, "autovmd_executed": False, "reserved_confirmation_opened": False,
        "qualification": False, "native_preservation_unresolved": True,
        "aggregation": "Channel-first recipient-balanced primary; example-mean and selected-window-weighted globally pooled energy supplementary; no deduplication of overlapping time samples"})
    plot_results(output)
