"""Kaggle-only source/cache diagnosis for the archived guarded VMD model.

Run inside the pinned legacy environment. No cache tolerance is relaxed and
no fitting, VMD computation or new anatomical assignment occurs here.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def diagnose(root, output, expected_checkpoint):
    if not Path("/kaggle/input").exists():
        raise RuntimeError("All numerical diagnosis must run on Kaggle")
    from eog_vmd_fcm_bgru.dataset_io import klados_arrays
    from eog_vmd_fcm_bgru.snr_target import unique_input, STARTS, WINDOW
    parent = unique_input("snr_target_protocol.json").parent
    protocol = json.loads((parent / "snr_target_protocol.json").read_text())
    checkpoint = parent / "eog_vmd_context_student/best_safe.pt"
    if digest(checkpoint) != expected_checkpoint:
        raise ValueError("Archived checkpoint digest differs")
    source = json.loads((parent / "run_config.json").read_text())
    if source["git_sha"] != "2ec0b3cd3da636cc3aed3b476ff890a6f56e2627":
        raise ValueError("Archived checkpoint source differs")
    dirty, clean, eyes = klados_arrays(root)
    details = []
    for part in ("train", "val", "test"):
        path = parent / f"{part}_vmd_cache.npz"
        expected = [(record, start) for record in protocol["split"][part] for start in STARTS]
        with np.load(path, allow_pickle=False) as cache:
            keys = [(int(record), int(start)) for record, start in cache["keys"]]
            if keys != expected:
                raise ValueError("Archived cache source/window keys differ")
            for name, values in (("raw", dirty), ("clean", clean), ("eyes", eyes)):
                current = np.stack([values[record, :, start:start+WINDOW] for record, start in expected])
                archived = cache[name]
                if current.shape != archived.shape or not np.isfinite(current).all() or not np.isfinite(archived).all():
                    raise ValueError("Invalid current/archive axes or samples")
                delta = current.astype(np.float64) - archived.astype(np.float64)
                energy = np.linalg.norm(archived.astype(np.float64))
                correlations = []
                for first, second in zip(current.reshape(-1, WINDOW), archived.reshape(-1, WINDOW)):
                    correlations.append(float(np.corrcoef(first, second)[0, 1]))
                details.append({"part": part, "array": name, "cache_sha256": digest(path),
                    "shape": list(current.shape), "current_dtype": str(current.dtype),
                    "archive_dtype": str(archived.dtype), "exact_equal": bool(np.array_equal(current, archived)),
                    "changed_samples": int(np.count_nonzero(delta)),
                    "relative_rms_difference": float(np.linalg.norm(delta)/max(energy, 1e-30)),
                    "maximum_absolute_difference": float(np.max(np.abs(delta))),
                    "minimum_channel_window_cc": min(correlations),
                    "current_range": [float(current.min()), float(current.max())],
                    "archive_range": [float(archived.min()), float(archived.max())]})
        print("GUARD_SOURCE_AUDIT", part, len(expected), flush=True)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch")}
    report = {"source_checkpoint_verified": True, "source_git_sha": source["git_sha"],
        "source_checkpoint_sha256": expected_checkpoint, "versions": versions,
        "source_files": {p.name: digest(p) for p in sorted((root / "klados").glob("*.npy"))},
        "comparison": details, "cache_mutated": False, "decompositions_computed": 0,
        "parameters_selected": False, "fresh_confirmation": False,
        "scope": "historical unknown-montage cache provenance diagnosis",
        "replay_authorized_by_diagnosis": False}
    (output / "guard17_source_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()
    diagnose(args.root, args.output, args.checkpoint_sha256)
