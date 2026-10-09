"""Fresh, Kaggle-executed data preparation for the VMD-EOG campaign.

The module makes data lineage explicit.  It does not claim that a low-EOG OSF
segment is biologically clean, and it never guesses Klados channel metadata.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from fractions import Fraction
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import loadmat

try:  # Root's shared I/O module is deliberately small and dependency-free.
    from .io import atomic_json, sha256_file, write_jsonl
except ImportError:  # Keeps this module independently importable during package assembly.
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def atomic_json(path: Path, value: Any) -> None:
        Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        Path(path).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


FS = 200
REGIONS = ("frontal", "posterior", "central_temporal", "unknown")
FRONTAL = {"FP1", "FP2", "FPZ", "AFZ", "FZ"} | {f"{p}{n}" for p in ("AF", "F") for n in range(1, 11)}
POSTERIOR = {"PZ", "POZ", "OZ", "IZ", "O1", "O2"} | {f"{p}{n}" for p in ("P", "PO") for n in range(1, 11)}


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _region(name: str) -> int:
    name = name.strip().upper()
    if name in FRONTAL:
        return 0
    if name in POSTERIOR:
        return 1
    if name.startswith(("FC", "FT", "C", "CP", "TP", "T")):
        return 2
    return 3


def _hemisphere(name: str) -> int:
    name = name.strip().upper()
    if name.startswith("UNKNOWN_"):
        return 3
    digits = "".join(char for char in name if char.isdigit())
    if not digits or name.endswith("Z"):
        return 2
    return 0 if int(digits) % 2 else 1


def _preprocess(values: np.ndarray, native_fs: float) -> np.ndarray:
    """Resample complete trial first, then zero-phase filter under the offline contract."""
    if native_fs <= 80:
        raise ValueError("Native Nyquist cannot support 40 Hz")
    values = np.asarray(values, dtype=np.float64)
    if native_fs != FS:
        ratio = Fraction(FS / native_fs).limit_denominator(10_000)
        values = signal.resample_poly(values, ratio.numerator, ratio.denominator, axis=-1)
    sos = signal.butter(4, [0.5, 40], fs=FS, btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, values, axis=-1).astype(np.float32)


def _mat_eeg(path: Path) -> dict[str, Any]:
    """Read EEGLAB .set payloads, including v7.3 files and external .fdt arrays.

    This is copied as transparent source rather than importing any legacy module.
    """
    try:
        payload = loadmat(path, simplify_cells=True)
    except NotImplementedError:
        from pymatreader import read_mat
        payload = read_mat(path)
    eeg = payload.get("EEG", payload)
    channels, samples, trials = (int(eeg[key]) for key in ("nbchan", "pnts", "trials"))
    raw = eeg["data"]
    if isinstance(raw, str):
        declared = raw.replace("\\", "/").rsplit("/", 1)[-1]
        fdt = path.parent / declared
        if not fdt.is_file():
            fdt = path.with_suffix(".fdt")
        if not fdt.is_file() or fdt.stat().st_size != channels * samples * trials * 4:
            raise ValueError(f"Invalid EEGLAB FDT companion for {path.name}")
        raw = np.memmap(fdt, dtype="<f4", mode="r", shape=(channels, samples, trials), order="F")
    else:
        raw = np.asarray(raw).reshape(channels, samples, trials, order="F")
    locations = eeg["chanlocs"]
    if isinstance(locations, dict):
        labels = locations["labels"]
        if isinstance(labels, str):
            locations = [locations]
        else:
            locations = [{key: (value[index] if isinstance(value, (list, tuple, np.ndarray)) and len(value) == channels else value)
                          for key, value in locations.items()} for index in range(channels)]
    locations = list(locations)
    names = [str(item["labels"]).strip() for item in locations]
    if len(names) != channels or len({name.upper() for name in names}) != channels:
        raise ValueError(f"Missing/duplicate EEGLAB labels in {path.name}")
    normalized = [name.upper().replace("-", "").replace("_", "") for name in names]
    aliases = {"EOG", "HEOG", "VEOG", "EOGH", "EOGV", "LEOG", "REOG", "UEOG", "DEOG", "EOG1", "EOG2"}
    eog_indices = [index for index, name in enumerate(normalized) if name in aliases]
    eeg_indices = [index for index in range(channels) if index not in eog_indices and normalized[index] not in {"LABEL", "BLOCK", "ARTIFACTCLASSES"}]
    return {"data": np.moveaxis(raw, -1, 0), "fs": float(eeg["srate"]), "names": names,
            "locations": locations, "eeg_indices": eeg_indices, "eog_indices": eog_indices}


def _find_axis(names: list[str], values: np.ndarray, axis: str) -> np.ndarray | None:
    aliases = {"HEOG": {"HEOG", "EOGH", "LEOG", "REOG", "EOG1"}, "VEOG": {"VEOG", "EOGV", "UEOG", "DEOG", "EOG2"}}
    matches = [index for index, name in enumerate(names) if name.upper().replace("-", "").replace("_", "") in aliases[axis]]
    if not matches:
        return None
    return values[matches[0]]


def _coords(locations, indices) -> tuple[np.ndarray, np.ndarray]:
    coordinates, mask = [], []
    for index in indices:
        current = []
        for axis in ("X", "Y", "Z"):
            raw = np.asarray(locations[index].get(axis, [])).reshape(-1)
            current.append(float(raw[0]) if raw.size == 1 and np.isfinite(raw[0]) else 0.0)
        ok = all(np.isfinite(current)) and any(current)
        coordinates.append(current if ok else [0., 0., 0.])
        mask.append(ok)
    return np.asarray(coordinates, dtype=np.float32), np.asarray(mask, dtype=bool)


def _groups(records: list[dict[str, Any]], seed: int) -> dict[str, str]:
    units = {}
    for record in records:
        participant = record.get("participant_id")
        key = "participant:" + participant if participant and record.get("participant_verified") else "clean:" + record.get("clean_hash", record["record_id"])
        units.setdefault(key, []).append(record["record_id"])
    order = sorted(units, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    labels = ("train", "val", "test")
    assignment = {}
    for position, key in enumerate(order):
        assignment.update({record_id: labels[(position * 3) // max(1, len(order))] for record_id in units[key]})
    return assignment


def _save_array(path: Path, eeg, names, channels, *, paired_reference=None, references=None) -> None:
    count = eeg.shape[0]
    if channels:
        coordinates, coordinate_mask = _coords(channels["locations"], channels["indices"])
    else:
        coordinates, coordinate_mask = np.zeros((count, 3), np.float32), np.zeros(count, bool)
    output = {"eeg": np.asarray(eeg, dtype=np.float32), "mask": np.ones(count, bool),
              "regions": np.asarray([_region(name) for name in names], np.int64),
              "hemispheres": np.asarray([_hemisphere(name) for name in names], np.int64),
              "coordinates": coordinates, "coordinate_mask": coordinate_mask,
              "channel_names": np.asarray(names, dtype="U64")}
    if paired_reference is not None:
        output["paired_reference"] = np.asarray(paired_reference, dtype=np.float32)
    if references is not None:
        output["references"] = np.asarray(references, dtype=np.float32)
    np.savez_compressed(path, **output)


def prepare_corpus(source_root: Path, output: Path, config: dict) -> dict:
    """Create frozen manifests and actual NPZ examples. Invoke only from a Kaggle job."""
    source_root, output = Path(source_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    window, hop, max_windows = int(config.get("window", 1024)), int(config.get("hop", 512)), int(config.get("max_windows_per_record", 2))
    if window <= 0 or hop <= 0 or max_windows <= 0:
        raise ValueError("window, hop and max_windows_per_record must be positive")
    records: list[dict[str, Any]] = []
    klados = source_root / "klados"
    clean_path, dirty_path = klados / "klados_pure_eeg.npy", klados / "klados_contaminated_eeg.npy"
    if clean_path.is_file() and dirty_path.is_file():
        clean = np.load(clean_path, allow_pickle=False)
        dirty = np.load(dirty_path, allow_pickle=False)
        if clean.shape != dirty.shape or clean.ndim != 3:
            raise ValueError("Klados paired arrays are not aligned [record,channel,sample]")
        for index in range(len(clean)):
            records.append({"record_id": f"klados:{index}", "dataset": "klados", "index": index,
                            "participant_id": None, "participant_verified": False,
                            "clean_hash": hashlib.sha256(np.ascontiguousarray(clean[index]).tobytes()).hexdigest(),
                            "units": "unknown", "region_metadata": "unknown", "paired_clean": True,
                            "limitations": ["Klados channel order, units and participant IDs are unresolved."]})
    osf_entries = []
    for path in sorted((source_root / "Dataset1_OSF").rglob("*_prep.set")):
        item = _mat_eeg(path)
        participant = path.stem.split("_")[1] if "_" in path.stem else path.stem
        # Existing source readme states IDs are globally unique across studies.
        record = {"record_id": f"osf:{participant}:{path.stem}", "dataset": "osf", "source_path": str(path.relative_to(source_root)),
                  "participant_id": participant, "participant_verified": True, "fs": item["fs"], "paired_clean": False,
                  "units": "source_units_preserved", "region_metadata": "verified_eeglab_names",
                  "limitations": ["Native clean target is unknown; controlled targets are recipient references only."]}
        records.append(record); osf_entries.append((record, item))
    if not records:
        raise ValueError("No supported Klados or OSF sources found")
    split = _groups(records, int(config.get("seed", 42)))
    atomic_json(output / "split_manifest.json", {"schema_version": 1, "assignment": split,
                "policy": "verified global OSF participant IDs; otherwise duplicate paired-clean hashes", "limits": "Klados is not participant independent."})
    arrays = output / "arrays"; arrays.mkdir(exist_ok=True)
    entries: list[dict[str, Any]] = []
    # Paired Klados engineering corpus.  Full records are preprocessed before windows.
    if clean_path.is_file() and dirty_path.is_file():
        heog = np.load(klados / "klados_heog.npy", allow_pickle=False).reshape(len(clean), -1, clean.shape[-1])[:, 0]
        veog = np.load(klados / "klados_veog.npy", allow_pickle=False).reshape(len(clean), -1, clean.shape[-1])[:, 0]
        clean_f, dirty_f, refs_f = _preprocess(clean, FS), _preprocess(dirty, FS), _preprocess(np.stack([heog, veog], axis=1), FS)
        for record in (row for row in records if row["dataset"] == "klados"):
            for start in range(0, clean_f.shape[-1] - window + 1, hop)[:max_windows]:
                filename = f"klados_{record['index']}_{start}.npz"; path = arrays / filename
                names = [f"UNKNOWN_{index}" for index in range(clean_f.shape[1])]
                _save_array(path, dirty_f[record["index"], :, start:start + window], names, {}, paired_reference=clean_f[record["index"], :, start:start + window], references=refs_f[record["index"], :, start:start + window])
                entries.append({"example_id": filename[:-4], "record_id": record["record_id"], "recipient_id": record["record_id"], "donor_id": None,
                                "split_role": split[record["record_id"]], "array_path": f"arrays/{filename}", "target_kind": "paired_klados_engineering",
                                "native_clean_status": "paired publisher array; units/metadata unknown", "channel_names": names})
    # Controlled OSF mixtures.  Donor and recipient must be distinct frozen participant groups.
    burdens, prepared = {}, []
    for record, item in osf_entries:
        eeg = _preprocess(item["data"][:, item["eeg_indices"]], item["fs"])
        heog, veog = _find_axis(item["names"], item["data"], "HEOG"), _find_axis(item["names"], item["data"], "VEOG")
        if heog is None or veog is None:
            continue
        eye = _preprocess(np.stack([heog, veog]), item["fs"])
        burdens[record["record_id"]] = float(np.sqrt(np.mean(eye ** 2)))
        prepared.append((record, item, eeg, eye))
    train_burdens = [burdens[row["record_id"]] for row, *_ in prepared if split[row["record_id"]] == "train"]
    threshold = float(np.quantile(train_burdens, float(config.get("low_eog_quantile", .25)))) if train_burdens else None
    for record, item, eeg, eye in prepared:
        if threshold is None or burdens[record["record_id"]] > threshold:
            continue
        eligible = [(drecord, ditem, deeg, deye) for drecord, ditem, deeg, deye in prepared
                    if split[drecord["record_id"]] == split[record["record_id"]] and drecord["participant_id"] != record["participant_id"]]
        if not eligible:
            continue
        donor, donor_item, donor_eeg, donor_eye = eligible[0]
        recipient_names = [item["names"][index] for index in item["eeg_indices"]]
        donor_names = [donor_item["names"][index] for index in donor_item["eeg_indices"]]
        donor_lookup = {name.upper(): index for index, name in enumerate(donor_names)}
        field = np.zeros((len(recipient_names), donor_eeg.shape[-1]), dtype=np.float32)
        design = np.c_[np.ones(donor_eye.shape[-1]), donor_eye.T]
        for target, name in enumerate(recipient_names):
            source = donor_lookup.get(name.upper())
            if source is not None:
                beta, *_ = np.linalg.lstsq(design, donor_eeg[:, source].T, rcond=None)
                field[target] = (design[:, 1:] @ beta[1:]).astype(np.float32)
        channels = {"locations": item["locations"], "indices": item["eeg_indices"]}
        available = min(eeg.shape[-1], field.shape[-1], donor_eye.shape[-1])
        for start in range(0, available - window + 1, hop)[:max_windows]:
            filename = f"osf_{record['participant_id']}_{donor['participant_id']}_{start}.npz"; path = arrays / filename
            mixed = eeg[0, :, start:start + window] + field[:, start:start + window]
            target = eeg[0, :, start:start + window]
            _save_array(path, mixed, recipient_names, channels, paired_reference=target, references=donor_eye[:, start:start + window])
            entries.append({"example_id": filename[:-4], "record_id": record["record_id"], "participant_id": record["participant_id"],
                            "recipient_id": record["record_id"], "donor_id": donor["record_id"], "split_role": split[record["record_id"]],
                            "array_path": f"arrays/{filename}", "target_kind": "controlled_osf_recipient_reference", "native_clean_status": "unknown",
                            "channel_names": recipient_names, "recipe": {"field": "empirical donor EEG projected on HEOG/VEOG", "low_eog_threshold_train_only": threshold}})
    write_jsonl(output / "recording_manifest.jsonl", records)
    write_jsonl(output / "corpus_manifest.jsonl", entries)
    summary = {"schema_version": 1, "recordings": len(records), "examples": len(entries), "split_hash": _hash_json(split),
               "low_eog_threshold": threshold, "native_clean_warning": "OSF native EEG is not asserted clean.",
               "klados_warning": "Klados examples are engineering paired data with unknown channel/participant metadata."}
    atomic_json(output / "corpus_summary.json", summary)
    return summary
