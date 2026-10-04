"""Inspect publisher MATLAB originals and test exact NPY provenance on Kaggle."""
import hashlib
import json
import re
from pathlib import Path
import urllib.request
import numpy as np
from scipy.io import loadmat
from .provenance import save_json, sha256_file


def describe(value, depth=0):
    if isinstance(value, dict):
        return {str(key): describe(item, depth + 1) for key, item in value.items() if not str(key).startswith("__")}
    if isinstance(value, np.ndarray):
        result = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if value.dtype.kind in "US" and value.size <= 100:
            result["strings"] = value.astype(str).tolist()
        elif value.dtype == object and depth < 4:
            result["first_items"] = [describe(item, depth + 1) for item in value.flat][:3]
        return result
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value.item() if isinstance(value, np.generic) else value
    return {"type": type(value).__name__}


def matrices(value, prefix="root", depth=0):
    """Preserve original cell/axis paths; do not invent subject identities."""
    if depth > 12:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not str(key).startswith("__"):
                yield from matrices(item, prefix + "/" + str(key), depth + 1)
    elif isinstance(value, (list, tuple)) or isinstance(value, np.ndarray) and value.dtype == object:
        items = value.flat if isinstance(value, np.ndarray) else value
        for index, item in enumerate(items):
            yield from matrices(item, prefix + f"[{index}]", depth + 1)
    elif isinstance(value, np.ndarray) and value.dtype.kind in "fiu":
        if value.ndim == 2 and 19 in value.shape:
            yield prefix, value if value.shape[0] == 19 else value.T
        elif value.ndim == 3:
            for channel_axis in [axis for axis, size in enumerate(value.shape) if size == 19]:
                for time_axis in [axis for axis, size in enumerate(value.shape) if size >= 5401 and axis != channel_axis]:
                    record_axis = next(axis for axis in range(3) if axis not in {channel_axis, time_axis})
                    ordered = np.transpose(value, (record_axis, channel_axis, time_axis))
                    for index, record in enumerate(ordered):
                        yield prefix + f"[record_axis={record_axis},record={index},channel_axis={channel_axis},time_axis={time_axis}]", record


def fingerprint(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def exact_prefix_matches(exported, original_records):
    """Check a declared start crop; never infer offsets or reorder electrodes."""
    lookup = {}
    for location, record in original_records:
        if record.ndim != 2 or record.shape[0] != exported.shape[1] or record.shape[1] < exported.shape[2]:
            continue
        length = exported.shape[2]
        for scale in [1.0, 1e6, 1e-6]:
            candidate = record[:, :length] * scale
            lookup.setdefault(fingerprint(candidate), []).append({
                "source_path": location, "amplitude_scale": scale,
                "source_samples": record.shape[1], "start_sample": 0, "stop_sample": length,
                "operation": "full record" if record.shape[1] == length else "exact start crop"})
    return [{"npy_record": index, "exact_float32_matches": lookup.get(fingerprint(record), [])}
            for index, record in enumerate(exported)]


def inspect_originals(root, output, catalog_path):
    catalog = json.loads(catalog_path.read_text())
    original = output / "publisher_matlab"
    original.mkdir(exist_ok=True)
    verified = []
    for entry in catalog["files"]:
        path = original / entry["name"]
        request = urllib.request.Request(entry["download"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=45) as response, path.open("wb") as destination:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                destination.write(block)
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("Publisher Klados original hash/size mismatch: " + entry["name"])
        verified.append(entry)
        save_json(output / "klados_publisher_downloads.json", {"source": catalog["source"], "verified_files": verified})
        print("Verified Klados publisher source", entry["name"], flush=True)
    payloads, structures = {}, {}
    for entry in verified:
        path = original / entry["name"]
        try:
            payload = loadmat(path, simplify_cells=True)
        except NotImplementedError:
            from pymatreader import read_mat
            payload = read_mat(path)
        payloads[path.name] = payload
        structures[path.name] = describe(payload)
    save_json(output / "klados_matlab_structure.json", structures)
    matches = {}
    for npy_name, mat_name in [("klados_pure_eeg.npy", "Pure_Data.mat"),
                               ("klados_contaminated_eeg.npy", "Contaminated_Data.mat"),
                               ("klados_heog.npy", "HEOG.mat"), ("klados_veog.npy", "VEOG.mat")]:
        exported = np.load(root / "klados" / npy_name, allow_pickle=False)
        if "eog.npy" in npy_name:
            exported = exported.reshape(len(exported), 1, exported.shape[-1])
            original_records = [("root/" + key, np.asarray(value).reshape(1, -1))
                                for key, value in payloads[mat_name].items() if not key.startswith("__")]
        else:
            original_records = list(matrices(payloads[mat_name]))
        record_matches = exact_prefix_matches(exported, original_records)
        used = {match["source_path"] for record in record_matches for match in record["exact_float32_matches"]}
        matches[npy_name] = {"export_shape": list(exported.shape), "source_matrix_candidates": len(original_records),
                             "source_shapes": sorted({str(record.shape) for _, record in original_records}),
                             "matched_records": sum(bool(item["exact_float32_matches"]) for item in record_matches),
                             "record_matches": record_matches,
                             "unused_originals": [name for name, _ in original_records if name not in used],
                             "test": "exact float32 start crop at sample 0; declared axis/scales; no filtering, offsets or electrode reordering"}
        print("Klados exact source matches", npy_name, matches[npy_name]["matched_records"], "of", len(exported), flush=True)
    save_json(output / "klados_original_matches.json", matches)
    alignment = []
    for index in range(len(exported)):
        ids = {}
        for name, result in matches.items():
            options = result["record_matches"][index]["exact_float32_matches"]
            ids[name] = sorted({int(re.search(r"(?:sim|heog_|veog_)(\d+)", option["source_path"]).group(1))
                                for option in options})
        aligned = all(len(numbers) == 1 for numbers in ids.values()) and len({numbers[0] for numbers in ids.values() if numbers}) == 1
        alignment.append({"npy_record": index, "publisher_ids": ids, "aligned_unique_source_id": aligned})
    save_json(output / "klados_export_alignment.json", alignment)
    save_json(output / "klados_provenance_check_summary.json", {"source": catalog["source"], "publisher_files_verified": len(verified),
              "clean_records_exactly_matched": matches["klados_pure_eeg.npy"]["matched_records"],
              "contaminated_records_exactly_matched": matches["klados_contaminated_eeg.npy"]["matched_records"],
              "heog_records_exactly_matched": matches["klados_heog.npy"]["matched_records"],
              "veog_records_exactly_matched": matches["klados_veog.npy"]["matched_records"],
              "all_four_exports_unique_and_aligned": all(row["aligned_unique_source_id"] for row in alignment),
              "anatomical_mapping": "unresolved until named source fields or explicit row-order documentation is reviewed",
              "subject_mapping": "unresolved; source cell indices are not automatically subject IDs", "victory": False})
