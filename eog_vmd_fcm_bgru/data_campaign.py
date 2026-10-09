"""Campaign-v2 data/governance stages. Numerical source parsing runs in Kaggle."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import hashlib
import numpy as np

from .campaign_contracts import atomic_json, canonical_hash, sha256_file
from .channel_regions import DICTIONARY_VERSION, REGION_NAMES, region_ids
from .controlled_corpus import build_corpus, fit_corpus
from .dataset_io import osf_trials
from .reference_guided import reference_projection
from .provenance import read_osf
from .splits_v2 import append_exposure_rows, freeze_splits


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _jsonl_read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _klados_rows(data_root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    package_root = Path(__file__).resolve().parents[1]
    mapping_path = package_root / "data_provenance" / "klados_verified_export_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    source = data_root / "klados"
    required = [source / name for name in ("klados_pure_eeg.npy", "klados_contaminated_eeg.npy", "klados_heog.npy", "klados_veog.npy")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Klados source files: " + ", ".join(missing))
    group_by_record = config.get("klados_clean_target_groups", {})
    rows = []
    for entry in mapping["records"]:
        index = int(entry["npy_record"])
        publisher_id = entry["publisher_ids"]["klados_pure_eeg.npy"][0]
        record_id = f"klados:{publisher_id}"
        rows.append({"record_id": record_id, "dataset": "klados", "source_recording_id": publisher_id,
                     "source_files": [{"path": str(path.relative_to(data_root)), "sha256": sha256_file(path)} for path in required],
                     "source_path": None,
                     "participant_id": None, "participant_verification": "unverified",
                     "clean_target_group": group_by_record.get(str(index), record_id), "fs_native": 200,
                     "units": "unverified_original_units", "reference": "publisher described; row-specific mapping unverified",
                     "channels": [], "channel_mapping": "unverified", "region_mapping": "unknown",
                     "eog": {"heog": "klados_heog.npy", "veog": "klados_veog.npy"},
                     "paired_clean": True, "eligibility": {"teacher": True, "student": True, "low_ocular_candidate": False},
                     "limitations": ["Klados channel order, units and subject identities are unverified.",
                                     "Do not make regional or participant-independent claims from this row."],
                     "npy_record": index})
    return rows


def _osf_rows(data_root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    requested_low = set(config.get("low_ocular_record_ids", []))
    rows: list[dict[str, Any]] = []
    for path in sorted((data_root / "Dataset1_OSF").rglob("*_prep.set")):
        item = read_osf(path)
        participant = str(item["participant"])
        record_id = f"osf:{item['study']}:{participant}:{path.stem}"
        names = [item["names"][index] for index in item["eeg_indices"]]
        groups = region_ids(names)
        channels = []
        for local_index, (source_index, name, group) in enumerate(zip(item["eeg_indices"], names, groups)):
            location = item["locations"][source_index]
            coordinates = [location.get(axis) for axis in ("X", "Y", "Z")]
            channels.append({"source_index": int(source_index), "index": local_index, "name": name,
                             "region": REGION_NAMES[int(group)], "region_dictionary": DICTIONARY_VERSION,
                             "coordinates_eeglab": coordinates})
        eog_names = {item["names"][index].upper(): item["names"][index] for index in item["eog_indices"]}
        rows.append({"record_id": record_id, "dataset": "osf", "source_recording_id": path.stem,
                     "source_files": [{"path": str(path.relative_to(data_root)), "sha256": sha256_file(path)}],
                     "source_path": str(path.relative_to(data_root)),
                     "participant_id": participant, "participant_verification": "verified",
                     "study": item["study"], "session": path.stem, "clean_target_group": None,
                     "fs_native": item["fs"], "units": "source_units_preserved", "reference": item["reference"],
                     "channels": channels, "channel_mapping": "verified_names_from_eeglab", "region_mapping": DICTIONARY_VERSION,
                     "eog": {"heog": eog_names.get("HEOG"), "veog": eog_names.get("VEOG")},
                     "paired_clean": False, "annotations": {"sample_labels": item["sample_labels"] is not None,
                                                               "trial_labels": item["trial_labels"] is not None},
                     "eligibility": {"teacher": True, "student": False,
                                     "low_ocular_candidate": record_id in requested_low},
                     "limitations": ["No paired clean EEG target; real-data outcomes are suppression/preservation proxies."]})
    return rows


def _burden_proxy(path: Path) -> float:
    """EOG-only session burden proxy for eligibility, never a native-clean label."""
    values = []
    for trial in osf_trials(path):
        heog, veog = trial["eog"].get("HEOG"), trial["eog"].get("VEOG")
        if heog is not None and veog is not None:
            values.append(float(np.sqrt(np.mean(heog ** 2) + np.mean(veog ** 2))))
    return float(np.median(values)) if values else float("inf")


def _nominate_low_ocular_candidates(recordings, split, data_root, config):
    """Freeze a threshold on train sessions, then apply it to every split."""
    proxies = {}
    for row in recordings:
        if row["dataset"] == "osf" and row.get("source_path"):
            proxies[row["record_id"]] = _burden_proxy(Path(data_root) / row["source_path"])
    train = [proxies[row["record_id"]] for row in recordings
             if row["dataset"] == "osf" and split["assignment"][row["record_id"]] == "train"
             and np.isfinite(proxies.get(row["record_id"], np.inf))]
    if not train:
        raise ValueError("No finite train-partition OSF EOG burdens for frozen recipient threshold")
    quantile = float(config.get("low_ocular_train_quantile", .25))
    if not 0 < quantile <= 1:
        raise ValueError("low_ocular_train_quantile must be in (0,1]")
    threshold = float(np.quantile(train, quantile))
    for row in recordings:
        if row["dataset"] == "osf":
            row["eligibility"]["low_ocular_candidate"] = bool(proxies.get(row["record_id"], np.inf) <= threshold)
            row["eligibility"]["low_ocular_proxy"] = proxies.get(row["record_id"], float("inf"))
    return {"method": "train_partition_eog_rms_quantile", "quantile": quantile, "threshold": threshold,
            "warning": "Low ocular burden is eligibility only; native EEG cleanliness remains unknown."}


def _hemispheres(names):
    values = []
    for name in names:
        digits = "".join(character for character in name if character.isdigit())
        values.append(-1 if digits and int(digits) % 2 else 1 if digits else 0)
    return np.asarray(values, dtype=np.int8)


def _coordinates(channels):
    coordinates, valid = [], []
    for channel in channels:
        value = channel.get("coordinates_eeglab", [None, None, None])
        try:
            numeric = np.asarray(value, dtype=float)
            good = numeric.shape == (3,) and np.isfinite(numeric).all()
        except (TypeError, ValueError):
            numeric, good = np.zeros(3), False
        coordinates.append(numeric if good else np.zeros(3))
        valid.append(good)
    return np.asarray(coordinates, dtype=np.float32), np.asarray(valid, dtype=bool)


def _write_npz(path, *, eeg, paired_reference, names, channels, references=None):
    coordinates, coordinate_mask = _coordinates(channels)
    arrays = {"eeg": np.asarray(eeg, dtype=np.float32), "mask": np.ones(len(eeg), dtype=bool),
              "regions": region_ids(names), "hemispheres": _hemispheres(names),
              "coordinates": coordinates, "coordinate_mask": coordinate_mask}
    if paired_reference is not None:
        arrays["paired_reference"] = np.asarray(paired_reference, dtype=np.float32)
    if references is not None:
        arrays["references"] = np.asarray(references, dtype=np.float32)
    np.savez_compressed(path, **arrays)


def _one_trial(path: Path, window: int):
    for trial in osf_trials(path):
        if trial["eeg"].shape[-1] >= window:
            return trial
    raise ValueError("No OSF trial long enough for requested corpus window")


def _materialize_osf(entry, by_id, data_root, array_root, window):
    recipient, donor = by_id[entry["recipient_id"]], by_id[entry["donor_id"]]
    recipient_trial = _one_trial(data_root / recipient["source_path"], window)
    donor_trial = _one_trial(data_root / donor["source_path"], window)
    references = np.stack([donor_trial["eog"]["HEOG"], donor_trial["eog"]["VEOG"]])
    names = recipient_trial["names"]
    lookup = {name.upper(): index for index, name in enumerate(donor_trial["names"])}
    # Fit the empirical donor field only from donor EEG/EOG.  Unmatched recipient
    # channels receive no invented field; the recipe remains montage-safe.
    artifact = np.zeros_like(recipient_trial["eeg"])
    for recipient_index, name in enumerate(names):
        donor_index = lookup.get(name.upper())
        if donor_index is not None:
            artifact[recipient_index] = reference_projection(donor_trial["eeg"][donor_index:donor_index + 1], references)[0]
    seed = int(entry["recipe"]["gain_seed"])
    gain = .5 + (int(hashlib.sha256(entry["example_id"].encode()).hexdigest()[:8], 16) ^ seed) % 101 / 100
    start = seed % (recipient_trial["eeg"].shape[-1] - window + 1)
    donor_start = seed % (references.shape[-1] - window + 1)
    eeg = recipient_trial["eeg"][:, start:start + window] + gain * artifact[:, donor_start:donor_start + window]
    target = recipient_trial["eeg"][:, start:start + window]
    out = array_root / (entry["example_id"].replace(":", "_") + ".npz")
    channel_meta = recipient["channels"]
    _write_npz(out, eeg=eeg, paired_reference=target, names=names, channels=channel_meta,
               references=references[:, donor_start:donor_start + window])
    entry["array_path"] = str(out.relative_to(array_root.parent))
    entry["channels"] = channel_meta
    entry["materialization"] = {"gain": gain, "recipient_start": start, "donor_start": donor_start,
                                "shared_field_channels": int(np.count_nonzero(np.any(artifact != 0, axis=1)))}


def _materialize_klados(recordings, split, data_root, array_root, window, starts):
    folder = data_root / "klados"
    dirty = np.load(folder / "klados_contaminated_eeg.npy", allow_pickle=False)
    clean = np.load(folder / "klados_pure_eeg.npy", allow_pickle=False)
    heog = np.load(folder / "klados_heog.npy", allow_pickle=False).reshape(len(dirty), -1, dirty.shape[-1])[:, 0]
    veog = np.load(folder / "klados_veog.npy", allow_pickle=False).reshape(len(dirty), -1, dirty.shape[-1])[:, 0]
    rows = []
    for record in recordings:
        if record["dataset"] != "klados":
            continue
        index, length = record["npy_record"], dirty.shape[-1]
        for start in starts:
            if start < 0 or start + window > length:
                continue
            example_id = f"klados:{record['source_recording_id']}:window:{start}"
            output = array_root / (example_id.replace(":", "_") + ".npz")
            names = [f"UNKNOWN_{channel}" for channel in range(dirty.shape[1])]
            _write_npz(output, eeg=dirty[index, :, start:start + window], paired_reference=clean[index, :, start:start + window],
                       names=names, channels=[], references=np.stack([heog[index, start:start + window], veog[index, start:start + window]]))
            rows.append({"example_id": example_id, "record_id": record["record_id"], "participant_id": None,
                         "recipient_id": record["record_id"], "donor_id": None, "split_role": split["assignment"][record["record_id"]],
                         "array_path": str(output.relative_to(array_root.parent)), "target_kind": "paired_klados_engineering",
                         "native_clean_status": "paired_publisher_clean; units/subject/rows unverified", "channels": [],
                         "recipe": {"kind": "publisher_pair", "start": start, "window": window},
                         "recipe_hash": canonical_hash({"kind": "publisher_pair", "record": record["record_id"], "start": start, "window": window}),
                         "limitations": record["limitations"]})
    return rows


def build_provenance(data_root: Path, output: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rows = _klados_rows(data_root, config) + _osf_rows(data_root, config)
    if not rows:
        raise ValueError("No campaign recordings discovered")
    _jsonl_write(output / "recording_manifest.jsonl", rows)
    channel_rows = [{"record_id": row["record_id"], **channel} for row in rows for channel in row["channels"]]
    _jsonl_write(output / "channel_manifest.jsonl", channel_rows)
    summary = {"schema_version": 2, "kind": "provenance_summary_v2", "recordings": len(rows),
               "recording_manifest_hash": canonical_hash(rows), "channel_manifest_hash": canonical_hash(channel_rows),
               "klados_region_claim_allowed": False,
               "notes": "Unknown Klados units/rows/participants remain explicit; OSF metadata is parsed from original EEGLAB sessions."}
    summary["artifact_hash"] = canonical_hash(summary)
    atomic_json(output / "provenance_summary_v2.json", summary)
    return summary


def run_data_stage(stage: str, data_root: Path, output: Path, config: Mapping[str, Any],
                   parents: Mapping[str, Path]) -> dict[str, Any]:
    """Execute only campaign data/governance stages; all callers run this on Kaggle."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if stage == "provenance":
        return build_provenance(Path(data_root), output, config)
    source = parents.get("provenance")
    if source is None:
        raise ValueError(f"{stage} requires a provenance parent artifact directory")
    recordings = _jsonl_read(Path(source) / "recording_manifest.jsonl")
    if stage == "split-freeze":
        exposure = parents.get("exposure_ledger")
        return freeze_splits(recordings, config, output,
                             Path(exposure) / "exposure_ledger.jsonl" if exposure else None)
    split_source = parents.get("split-freeze")
    if split_source is None:
        raise ValueError(f"{stage} requires a split-freeze parent artifact directory")
    split = json.loads((Path(split_source) / "split_manifest.json").read_text(encoding="utf-8"))
    if stage == "corpus-fit":
        return fit_corpus(recordings, split, output, config, parents)
    if stage == "corpus-build":
        fit_source = parents.get("corpus-fit")
        if fit_source is None:
            raise ValueError("corpus-build requires corpus-fit parent artifact directory")
        fit = json.loads((Path(fit_source) / "corpus_fit_manifest.json").read_text(encoding="utf-8"))
        return build_corpus(recordings, split, fit, output, config, parents)
    raise ValueError("Unsupported data stage: " + stage)
