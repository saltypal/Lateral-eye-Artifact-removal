"""Bounded synthetic corpus producer; never a denoising or teacher algorithm."""
import json
import time

import numpy as np

from .artifacts import verify_parent
from .corpus import unique_parent
from .coupled_field import fit_donor_field, generated_pair, valid_driver_design
from .data import channel_metadata
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl


def whole_trial_prefix(boundaries, donor_duration, recipient_duration):
    """Choose the longest complete donor-trial prefix that fits the recipient."""
    cuts = np.asarray(boundaries)
    if (cuts.ndim != 1 or not np.isfinite(cuts).all() or np.any(cuts != np.floor(cuts))
            or np.any(cuts <= 0) or np.any(cuts >= donor_duration)
            or np.any(np.diff(cuts) <= 0)):
        raise ValueError("Invalid donor trial boundaries")
    cuts = cuts.astype(int)
    ends = np.r_[cuts, donor_duration]
    eligible = ends[ends <= recipient_duration]
    if not len(eligible):
        raise ValueError("No complete calibration trial fits recipient prefix")
    duration = int(eligible[-1])
    return duration, cuts[cuts < duration]


def channel_intersection(recipient_names, donor_names):
    left = [str(name).upper() for name in recipient_names]
    right = [str(name).upper() for name in donor_names]
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError("Duplicate verified channel names")
    lookup = {name: index for index, name in enumerate(right)}
    common = [(index, lookup[name]) for index, name in enumerate(left) if name in lookup]
    if len(common) < 8:
        raise ValueError("Insufficient exact-name montage intersection")
    recipient_ids, donor_ids = zip(*common)
    return list(recipient_ids), list(donor_ids)


def readiness(input_root, config):
    if (config.get("scope") != "bounded_homogeneous_coupled_pilot"
            or config.get("autovmd_enabled") is not False
            or config.get("reserved_access") != "closed until verified freeze"
            or config.get("window") != 1024 or config.get("fs") != 200):
        raise ValueError("Coupled pilot policy differs")
    probe = unique_parent(input_root, "source_probe_summary.json").parent
    corpus = unique_parent(input_root, "corpus_summary.json").parent
    for name, parent in (("probe", probe), ("corpus", corpus)):
        expected = config["producer_parents"][name]
        if sha256_file(parent / "artifact_manifest.json") != expected["manifest_sha256"]:
            raise ValueError("Coupled pilot parent manifest differs: " + name)
        verify_parent(parent, ("run_spec.json", "split_manifest.json"))
        spec = json.loads((parent / "run_spec.json").read_text())
        if any(spec.get(key) != expected[key] for key in ("run_id", "campaign_id", "git_sha")):
            raise ValueError("Coupled pilot parent identity differs: " + name)
        if sha256_file(parent / "split_manifest.json") != config["source_split_sha256"]:
            raise ValueError("Coupled pilot source split differs")
    verify_parent(probe, ("source_probe_summary.json", "source_probe_selection.jsonl"))
    summary = json.loads((probe / "source_probe_summary.json").read_text())
    if (summary.get("pairs") != 2 or summary.get("source_recovery_passed") is not True
            or summary.get("reserved_confirmation_opened") is not False):
        raise ValueError("Original source recovery is not eligible")
    pairs = read_jsonl(probe / "source_probe_selection.jsonl")
    split = json.loads((probe / "split_manifest.json").read_text())
    if (len(pairs) != 2 or len({p["recipient"] for p in pairs}) != 2
            or len({p["donor"] for p in pairs}) != 2
            or len({p["partition"]["fold"] for p in pairs}) != 2):
        raise ValueError("Pilot needs two distinct recipient/donor development folds")
    for pair in pairs:
        partition = pair["partition"]
        if (partition.get("role") != "development"
                or split["partitions"]["lemon"][pair["recipient"]] != partition
                or split["partitions"]["osf"][pair["donor"]] != partition):
            raise ValueError("Pilot source partition differs or is reserved")
    return probe, corpus, pairs


def checked_arrays(parent, relative):
    verify_parent(parent, (relative,))
    with np.load(parent / relative, allow_pickle=False) as archive:
        return dict(archive)


def run(input_root, output, config, profile):
    if profile != "pilot":
        raise ValueError("Coupled producer is a bounded pilot only")
    started = time.perf_counter()
    probe, corpus, pairs = readiness(input_root, config)
    write_jsonl(output / "source_selection.jsonl", pairs)
    atomic_json(output / "split_manifest.json", json.loads((probe / "split_manifest.json").read_text()))
    rows, diagnostics = [], []
    (output / "arrays").mkdir()
    for pair in pairs:
        participant = pair["recipient"].removeprefix("lemon:")
        base = "original_sources/" + participant + "/"
        names = ("recipient_record.json", "donor_record.json")
        verify_parent(probe, tuple(base + name for name in names))
        recipient = json.loads((probe / base / names[0]).read_text())
        donor = json.loads((probe / base / names[1]).read_text())
        if (donor["record_id"] != pair["donor_record"] or donor["participant"] != pair["donor"]
                or donor["partition"] != pair["partition"]
                or not set(donor["calibration_trial_ids"]).isdisjoint(donor["scoring_trial_ids"])):
            raise ValueError("Original donor identity or trial separation differs")
        cal = checked_arrays(probe, base + "donor_calibration.npz")
        background = checked_arrays(probe, base + "recipient_calibration.npz")
        selected = [w for w in recipient["scoring_windows"] if w["start"] == pair["recipient_start"]]
        if len(selected) != 1:
            raise ValueError("Frozen chronological recipient window is ambiguous")
        target_path = base + selected[0]["array"]
        target_full = checked_arrays(probe, target_path)["eeg"]
        rids, dids = channel_intersection(background["channel_names"], cal["channel_names"])
        metadata = channel_metadata(background["channel_names"][rids].tolist())
        duration, cuts = whole_trial_prefix(cal["boundaries"], cal["eeg"].shape[-1], background["eeg"].shape[-1])
        model = fit_donor_field(cal["eeg"][dids, :duration], cal["references"][:, :duration], cuts)
        cal_driver = model.excursions(cal["references"][:, :duration])
        mixed_cal = generated_pair(background["eeg"][rids, :duration], cal_driver, model, 0., cuts)
        np.testing.assert_allclose(model.forward(mixed_cal["references"], cuts),
                                   mixed_cal["true_artifact"], rtol=1e-10, atol=1e-18)
        pair_id = participant + "-" + donor["record_id"]
        caldir = output / "calibration" / pair_id
        caldir.mkdir(parents=True)
        # Only allowed observed calibration reaches the downstream denoiser.
        np.savez_compressed(caldir / "calibration.npz", eeg=mixed_cal["eeg"],
                            references=mixed_cal["references"], boundaries=cuts,
                            trial_types=cal["trial_types"][:duration], **metadata)
        truthdir = output / "producer_truth" / pair_id
        truthdir.mkdir(parents=True)
        np.savez_compressed(truthdir / "field.npz", baseline=model.baseline, scales=model.scales,
                            coefficients=model.coefficients, lags=model.lags,
                            calibration_background=background["eeg"][rids, :duration],
                            calibration_artifact=mixed_cal["true_artifact"])
        drivers, source_windows = {}, {}
        for condition in ("blink", "lateral"):
            eligible = [w for w in donor["windows"] if w["condition"] == condition]
            if not eligible:
                raise ValueError("Missing annotated donor condition: " + condition)
            window = eligible[0]  # chronological source order; never metric-selected
            if window["trial"] not in donor["scoring_trial_ids"]:
                raise ValueError("Scoring donor window overlaps calibration")
            number = donor["windows"].index(window)
            relative = "arrays/" + donor["record_id"] + f"-native-{number:03d}.npz"
            ocular = checked_arrays(corpus, relative)
            if ocular["references"].shape != (2, 1024):
                raise ValueError("Donor scoring reference shape differs")
            drivers[condition] = model.excursions(ocular["references"])
            source_windows[condition] = {"window": window, "array": relative,
                                         "sha256": sha256_file(corpus / relative)}
        drivers["mixed"] = (drivers["blink"] + drivers["lateral"]) / np.sqrt(2.)
        drivers["clean"] = np.zeros((2, 1024))
        target = target_full[rids]
        if target.shape != (len(rids), 1024):
            raise ValueError("Recipient scoring shape differs")
        for condition in ("clean", "blink", "lateral", "mixed"):
            for level in ([None] if condition == "clean" else [-5., 0., 5.]):
                generated = generated_pair(target, drivers[condition], model, level)
                np.testing.assert_allclose(generated["eeg"], target + generated["true_artifact"], rtol=1e-12, atol=1e-18)
                np.testing.assert_allclose(model.forward(generated["references"]), generated["true_artifact"], rtol=1e-10, atol=1e-18)
                if not all(np.isfinite(value).all() for key, value in generated.items()
                           if key != "generation_amplitude"):
                    raise ValueError("Nonfinite generated pilot array")
                if level is not None:
                    measured = 10 * np.log10(np.sum(target ** 2) / np.sum(generated["true_artifact"] ** 2))
                    if abs(float(measured) - level) > 1e-9:
                        raise ValueError("Measured controlled input SNR differs")
                suffix = "clean" if level is None else f"snr{int(level):+d}"
                example_id = pair_id + "-" + condition + "-" + suffix
                # Producer parameters and truth are not inputs to correction APIs.
                np.savez_compressed(output / "arrays" / (example_id + ".npz"),
                                    **{k: v for k, v in generated.items() if k != "generation_amplitude"}, **metadata)
                regional_snr = {}
                for label, region in (("frontal", 0), ("posterior", 1), ("central_shared", 2)):
                    ids = metadata["regions"] == region
                    energy = float(np.sum(generated["true_artifact"][ids] ** 2))
                    regional_snr[label] = float(10 * np.log10(np.sum(target[ids] ** 2) / energy)) if energy > 0 else None
                rows.append({"example_id": example_id, "array_path": "arrays/" + example_id + ".npz",
                             "recipient": pair["recipient"], "donor": pair["donor"], "partition": pair["partition"],
                             "condition": condition, "input_snr_db": level, "regional_input_snr_db": regional_snr,
                             "target_kind": "controlled_recipient_reference", "dataset": "controlled_LEMON_OSF_coupled",
                             "calibration": "calibration/" + pair_id + "/calibration.npz",
                             "recipient_start": pair["recipient_start"], "target_sha256": sha256_file(probe / target_path),
                             "scoring_field_mean": generated["true_artifact"].mean(-1).tolist(),
                             "valid_lag_samples": int(valid_driver_design(generated["references"], model.lags, None)[1].sum()),
                             "generation_amplitude_evaluator_only": generated["generation_amplitude"]})
        _, valid = valid_driver_design(cal_driver, model.lags, cuts)
        diagnostics.append({"pair_id": pair_id, "source_windows": source_windows,
                            "calibration_samples": duration, "calibration_boundaries": cuts.tolist(),
                            "valid_calibration_samples": int(valid.sum()), "channels": len(rids),
                            "calibration_generation_amplitude_evaluator_only": mixed_cal["generation_amplitude"],
                            "calibration_field_mean": mixed_cal["true_artifact"].mean(-1).tolist(),
                            "recipient_units": recipient["units"], "donor_units": donor.get("units", "unknown original numerical coordinates")})
        write_jsonl(output / "corpus_manifest.jsonl", rows)
        atomic_json(output / "producer_diagnostics.json", diagnostics)
        print("COUPLED_PAIR_SAVED", pair_id, len(rows), flush=True)
    if len(rows) != 20:
        raise ValueError("Bounded pilot must contain exactly twenty records")
    atomic_json(output / "coupled_pilot_summary.json", {"examples": len(rows), "pairs": 2,
                "source_split_sha256": config["source_split_sha256"], "construction_contracts_passed": True,
                "new_vmd_decompositions": 0, "weights_trained": False, "qualification": False,
                "reserved_confirmation_opened": False, "regression_generated_bias_remains": True,
                "idealized_clean_controls": True, "runtime_s": time.perf_counter() - started})
