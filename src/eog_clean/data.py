"""Klados paired arrays and original named-channel OSF trials, without mixtures."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from .osf_reader import read_osf


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def region(name):
    """Engineering groups from verified labels, never from numerical row order."""
    name = name.strip().upper()
    if re.fullmatch(r"(?:FP|AF|F)(?:\d+|Z)", name):
        return "frontal"
    if re.fullmatch(r"(?:P|PO|O)(?:\d+|Z)", name) or name == "IZ":
        return "posterior"
    if re.fullmatch(r"(?:FC|FT|C|CP|TP|T)(?:\d+|Z)", name):
        return "central"
    return "unknown"


def preprocess(values, fs=200):
    if fs <= 80:
        raise ValueError("Source Nyquist cannot support 40 Hz")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite source samples")
    if fs != 200:
        ratio = Fraction(200/float(fs)).limit_denominator(10000)
        values = resample_poly(values, ratio.numerator, ratio.denominator, axis=-1)
    sos = butter(4, [0.5, 40], fs=200, btype="bandpass", output="sos")
    return sosfiltfilt(sos, values, axis=-1).astype(np.float32)


def _extract(stream, members, destination):
    destination = Path(destination).resolve()
    for name in members:
        if not (destination/name).resolve().is_relative_to(destination):
            raise ValueError("Archive member escapes destination")
        stream.extract(name, destination)


def prepare_sources(input_root, destination, output, supplement_sha):
    """Verify container; extract only Klados and OSF source members.

    The old source bundle predates the later generated corpora. No generated
    recipient/reference mixtures are read or attached to the replacement job.
    """
    manifests = list(Path(input_root).rglob("source_manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"Need one original source manifest; found {len(manifests)}")
    payload = json.loads(manifests[0].read_text())
    declaration = payload["archive"]
    archive = manifests[0].parent/declaration["name"]
    if archive.stat().st_size != declaration["bytes"] or sha256(archive) != declaration["sha256"]:
        raise ValueError("Original source archive checksum mismatch")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as stream:
        selected = [name for name in stream.namelist()
                    if "klados" in Path(name).parts or "Dataset1_OSF" in Path(name).parts]
        _extract(stream, selected, destination)
    candidates = list(destination.rglob("klados_contaminated_eeg.npy"))
    if len(candidates) != 1:
        raise ValueError("Expected one original Klados array tree")
    root = candidates[0].parent.parent
    verified = []
    for entry in payload["files"]:
        relative = Path(entry["path"])
        if relative.parts[0] not in {"klados", "Dataset1_OSF"}:
            continue
        path = (root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("Required source file absent: "+str(relative))
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError("Source file differs: "+str(relative))
        verified.append(entry)
    for archive in (root/"Dataset1_OSF").glob("study*.zip"):
        with zipfile.ZipFile(archive) as stream:
            _extract(stream, stream.namelist(), archive.parent)
    supplements = list(Path(input_root).rglob("supplement_manifest.json"))
    if len(supplements) != 1:
        raise ValueError("Exactly one versioned OSF restoration supplement is required")
    extra = json.loads(supplements[0].read_text())
    archive = supplements[0].parent/extra["archive"]["name"]
    if extra["archive"]["sha256"] != supplement_sha or sha256(archive) != supplement_sha:
        raise ValueError("OSF restoration supplement differs from approved source")
    with zipfile.ZipFile(archive) as stream:
        _extract(stream, stream.namelist(), root)
    for entry in extra["files"]:
        if sha256(root/entry["path"]) != entry["sha256"]:
            raise ValueError("Restored OSF source differs")
    save_json(Path(output)/"source_verification.json", {
        "original_manifest_sha256": sha256(manifests[0]), "source_archive_sha256": declaration["sha256"],
        "selected_files": verified, "supplement_sha256": supplement_sha,
        "active_datasets": ["Klados", "OSF"], "generated_mixtures_loaded": False,
    })
    return root


def klados(root):
    paths = {name: root/"klados"/f"klados_{name}.npy" for name in
             ("contaminated_eeg", "pure_eeg", "heog", "veog")}
    arrays = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
    dirty, clean = arrays["contaminated_eeg"], arrays["pure_eeg"]
    if dirty.shape != clean.shape or dirty.ndim != 3 or dirty.shape[1] != 19:
        raise ValueError("Unexpected Klados N×19×T dimensions")
    references = []
    for axis in ("heog", "veog"):
        reference = arrays[axis].reshape(len(dirty), -1, dirty.shape[-1])
        if reference.shape[1] != 1:
            raise ValueError("Ambiguous Klados ocular reference axis")
        references.append(reference[:, 0])
    return dirty, clean, np.stack(references, axis=1), {
        "files": {name: sha256(path) for name, path in paths.items()},
        "shape": list(dirty.shape), "channel_names": None, "participant_ids": None,
        "regional_claim_permitted": False, "source_previously_filtered": True,
        "additional_filter": "matched fourth-order zero-phase 0.5–40 Hz, separately within calibration/scoring segments",
    }


def osf_sessions(root, partition, limit):
    selected = []
    for path in sorted((root/"Dataset1_OSF").rglob("*_prep.set")):
        participant = path.stem.split("_")[1]
        identity = "osf:"+participant
        if identity not in partition or partition[identity]["role"] != "development":
            continue
        selected.append(path)
    return selected[:limit] if limit else selected


def osf_trials(path):
    item = read_osf(path)
    normalized = [name.upper().replace("_", "").replace("-", "") for name in item["names"]]
    references = [normalized.index(axis) for axis in ("HEOG", "VEOG")]
    if any(index not in item["eog_indices"] for index in references):
        raise ValueError("Named H/V channels are not verified EOG channels")
    eval_indices = []
    for axis, fallback in zip(("HEOGLPF", "VEOGLPF"), references):
        eval_indices.append(normalized.index(axis) if axis in normalized else fallback)
    if item["trial_labels"] is None:
        raise ValueError("OSF trial labels absent")
    blocks = None
    if "BLOCK" in item["annotations"]:
        block_values = item["data"][:, item["annotations"]["BLOCK"]]
        if not np.allclose(block_values, block_values[:, :1]):
            raise ValueError("OSF block changes inside a trial")
        blocks = np.round(block_values[:, 0]).astype(int)
    names = [item["names"][i] for i in item["eeg_indices"]]
    groups = [region(name) for name in names]
    trials = []
    for index, label in enumerate(item["trial_labels"]):
        label_values = item["data"][index, item["annotations"]["LABEL"]]
        if not np.allclose(label_values, label):
            raise ValueError("OSF condition changes inside a trial")
        raw = item["data"][index]
        evaluation = np.asarray(raw[eval_indices], float)
        if item["fs"] != 200:
            ratio = Fraction(200/item["fs"]).limit_denominator(10000)
            evaluation = resample_poly(evaluation, ratio.numerator, ratio.denominator, axis=-1)
        trials.append({"index": index, "condition": int(label),
            "block": int(blocks[index]) if blocks is not None else None,
            "eeg": preprocess(raw[item["eeg_indices"]], item["fs"]),
            "eyes": preprocess(raw[references], item["fs"]), "evaluation_eyes": evaluation.astype(np.float32)})
    metadata = {"record": path.stem, "participant": item["participant"], "study": item["study"],
        "channel_names": names, "regions": groups, "native_fs": item["fs"],
        "reference": item["reference"], "source_set_sha256": sha256(path),
        "evaluation_references": [item["names"][i] for i in eval_indices],
        "evaluation_reference_processing": "resampling only; original values retained unchanged",
        "units": "source amplitude units; physical scale not independently established"}
    return trials, metadata


def calibration_split(trials):
    """Use declared block 1 vs block 2; otherwise whole earliest condition trials."""
    blocks = {item["block"] for item in trials}
    if {1, 2}.issubset(blocks):
        calibration = [item for item in trials if item["block"] == 1]
        scoring = [item for item in trials if item["block"] == 2]
        policy = "block 1 calibration; block 2 scoring"
    else:
        first = {}
        for item in trials:
            first.setdefault(item["condition"], item["index"])
        stop = max(first.values())
        calibration = [item for item in trials if item["index"] <= stop]
        scoring = [item for item in trials if item["index"] > stop]
        policy = "chronological whole-trial prefix through first occurrence of each condition"
    if {item["index"] for item in calibration} & {item["index"] for item in scoring}:
        raise ValueError("Calibration/scoring overlap")
    if not {1, 2, 3, 4}.issubset({item["condition"] for item in calibration}) or not scoring:
        raise ValueError("Insufficient separate calibration/scoring conditions")
    return calibration, scoring, policy
