"""File integrity and raw dataset audit. Run all array inspection on Kaggle."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def environment(repository: Path) -> dict:
    packages = {}
    for name in ["numpy", "scipy", "pandas", "mne", "torch", "vmdpy",
                 "scikit-fuzzy", "scikit-learn", "python-picard", "mne-denoise", "ARMBR"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "git_sha": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()}


def read_osf(path: Path) -> dict:
    """Read original EEGLAB arrays; never apply EEG unit scaling to labels.

    EEGLAB stores (channel, sample, trial), with external FDT in Fortran order.
    Keeping original units avoids MNE's EEG-to-volts conversion rounding integer
    annotation channels to zero. Each trial remains a separate continuous unit.
    """
    payload = loadmat(path, simplify_cells=True)
    eeg = payload.get("EEG", payload)
    channels, samples, trials = (int(eeg[name]) for name in ("nbchan", "pnts", "trials"))
    data = eeg["data"]
    if isinstance(data, str):
        external = (path.parent / data).resolve()
        if external.parent != path.parent.resolve():
            raise ValueError("External FDT must be in the session directory")
        if external.stat().st_size != channels * samples * trials * 4:
            raise ValueError("External FDT length does not match declared dimensions")
        data = np.memmap(external, dtype="<f4", mode="r", shape=(channels, samples, trials), order="F")
    else:
        data = np.asarray(data).reshape(channels, samples, trials, order="F")
    data = np.moveaxis(data, -1, 0)
    locations = eeg["chanlocs"]
    if isinstance(locations, dict):
        locations = [locations]
    locations = list(locations)
    names = [str(item["labels"]).strip() for item in locations]
    if len(names) != channels or len(set(name.upper() for name in names)) != channels:
        raise ValueError("Missing or duplicate channel names")
    normalized = [name.upper().replace("-", "").replace("_", "") for name in names]
    annotations = {name: normalized.index(name) for name in ["ARTIFACTCLASSES", "LABEL", "BLOCK"] if name in normalized}
    eog_aliases = {"HEOG", "VEOG", "EOGH", "EOGV", "LEOG", "REOG", "UEOG", "DEOG"}
    eog_indices = [i for i, name in enumerate(normalized)
                   if name in eog_aliases or str(locations[i].get("type", "")).upper() == "EOG"]
    excluded = set(annotations.values()) | set(eog_indices)
    eeg_indices = [i for i, item in enumerate(locations) if i not in excluded
                   and str(item.get("type", "EEG")).upper() in ("", "EEG")]
    match = re.fullmatch(r"(study\d+)_(p\d+)_prep", path.stem)
    if match is None:
        raise ValueError("Session filename does not establish participant identity")
    sample_labels = None
    if "ARTIFACTCLASSES" in annotations:
        original = data[:, annotations["ARTIFACTCLASSES"]]
        if not np.allclose(original, np.round(original), atol=1e-4):
            raise ValueError("Sample labels are not unscaled integers")
        sample_labels = np.round(original).astype(np.int16)
        if not np.isin(sample_labels, np.arange(7)).all():
            raise ValueError("Unexpected artifactclasses codes")
    trial_labels = None
    if "LABEL" in annotations:
        trial_labels = np.round(data[:, annotations["LABEL"], 0]).astype(int).tolist()
    return {"path": path, "study": match[1], "participant": match[2], "fs": float(eeg["srate"]),
            "data": data, "names": names, "locations": locations, "eeg_indices": eeg_indices,
            "eog_indices": eog_indices, "annotations": annotations, "sample_labels": sample_labels,
            "trial_labels": trial_labels, "reference": str(eeg.get("ref", "not declared"))}


def audit(root: Path, manifest_path: Path, output: Path, repository: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    integrity = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        match = path.is_file() and path.stat().st_size == entry["bytes"] and sha256_file(path) == entry["sha256"]
        integrity.append({"path": entry["path"], "match": bool(match)})
    save_json(output / "integrity.json", integrity)
    if not all(item["match"] for item in integrity):
        raise RuntimeError("Uploaded dataset failed source hash verification")
    klados = root / "klados"
    arrays = {name: np.load(klados / filename, mmap_mode="r", allow_pickle=False)
              for name, filename in {"dirty": "klados_contaminated_eeg.npy", "clean": "klados_pure_eeg.npy",
                                     "heog": "klados_heog.npy", "veog": "klados_veog.npy"}.items()}
    shapes = {name: list(value.shape) for name, value in arrays.items()}
    if arrays["dirty"].ndim != 3 or arrays["dirty"].shape != arrays["clean"].shape:
        raise ValueError("Klados dirty and target shapes are not exactly aligned")
    n_records = arrays["dirty"].shape[0]
    for name, value in arrays.items():
        if value.shape[0] != n_records or value.shape[-1] != arrays["dirty"].shape[-1]:
            raise ValueError(f"Unexplained Klados {name} alignment mismatch")
        if not all(np.isfinite(record).all() for record in value):
            raise ValueError(f"Nonfinite Klados {name} values")
    fingerprints = [hashlib.sha256(np.ascontiguousarray(record).tobytes()).hexdigest()
                    for record in arrays["clean"]]
    save_json(output / "klados_record_groups.json", fingerprints)
    session_rows = []
    for path in sorted((root / "Dataset1_OSF").rglob("*_prep.set")):
        try:
            item = read_osf(path)
            data = item["data"][:, item["eeg_indices"]]
            nonfinite = np.any(~np.isfinite(data), axis=(0, 2))
            padded = np.all(data == 0, axis=(0, 2))
            flat = np.all(np.ptp(data, axis=-1) == 0, axis=0)
            labels = item["sample_labels"]
            names = [item["names"][i] for i in item["eeg_indices"]]
            valid = ~(nonfinite | padded | flat)
            coordinates = []
            for i in item["eeg_indices"]:
                position = []
                for axis in ("X", "Y", "Z"):
                    value = np.asarray(item["locations"][i].get(axis, [])).reshape(-1)
                    position.append(float(value[0]) if value.size == 1 and np.isfinite(value[0]) else None)
                coordinates.append(position)
            session_rows.append({"file": str(path.relative_to(root)), "session": path.stem,
                                 "study": item["study"], "participant": item["participant"],
                                 "status": "usable" if valid.sum() >= 1 else "excluded",
                                 "sampling_hz": item["fs"], "trials": data.shape[0], "samples_per_trial": data.shape[-1],
                                 "reference": item["reference"], "eeg_names": names,
                                 "valid_channels": valid.tolist(), "coordinates_eeglab_unconverted": coordinates,
                                 "eog_names": [item["names"][i] for i in item["eog_indices"]],
                                 "trial_labels": item["trial_labels"],
                                 "sample_label_counts": {str(code): int((labels == code).sum()) for code in range(7)} if labels is not None else None})
        except Exception as error:
            session_rows.append({"file": str(path.relative_to(root)), "status": "excluded", "reason": repr(error)})
    save_json(output / "osf_sessions.json", session_rows)
    block_sessions = {path.name.replace("_block_dt.mat", "") for path in (root / "Dataset1_OSF").rglob("*_block_dt.mat")}
    set_sessions = {path.name.replace("_prep.set", "") for path in (root / "Dataset1_OSF").rglob("*_prep.set")}
    summary = {"uploaded_files": len(integrity), "uploaded_bytes": sum(item["bytes"] for item in manifest["files"]),
               "hash_verification": "passed", "klados_shapes": shapes, "klados_records": n_records,
               "klados_expected_published_records": 54, "klados_channel_mapping": "unverified",
               "klados_subject_mapping": "unverified", "klados_units": "unverified_original_units",
               "klados_sampling_hz": 200, "klados_duplicate_target_groups": n_records - len(set(fingerprints)),
               "osf_raw_sessions": len(session_rows), "osf_usable_sessions": sum(item["status"] == "usable" for item in session_rows),
               "osf_unique_participants": len({item["participant"] for item in session_rows if item["status"] == "usable"}),
               "osf_participant_provenance": "Dataset1_OSF/readme.txt: IDs unique across studies",
               "sessions_with_timing_but_missing_set": sorted(block_sessions - set_sessions),
               "EyeTrack": "uploaded; excluded from primary protocol pending separate provenance audit",
               "derivatives": "uploaded; not counted as independent raw sessions or clean targets",
               "proceed_to_classical_gate": bool(n_records >= 10 and any(item["status"] == "usable" for item in session_rows)),
               "anatomical_klados_gate": False, "victory": False}
    save_json(output / "audit_summary.json", summary)
    save_json(output / "environment.json", environment(repository))
    print(json.dumps(summary, indent=2))
    return summary
