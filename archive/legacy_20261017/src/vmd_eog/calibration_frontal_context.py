"""Cross-fitted frontal VMD artifact context for nonfrontal calibration.

This is a bounded EOG-assisted adaptation. It uses at most four whole
1024-sample calibration chunks, computes fixed K3/alpha2000 frontal modes,
and cross-fits the frontal calibration transfer so each returned context chunk
was estimated without fitting on that same chunk. It never accepts targets.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from .contracts import canonical_hash
from .frontal_calibration_transfer import calibration_transfer
from .frontal_methods import FixedRecipe, decompose_frontal, frontal_indices
from .io import atomic_json, sha256_file
from .data import _hemisphere
from .reference import signed_context


CACHE_VERSION = 1
CHUNK_SAMPLES = 1024
MAX_CHUNKS = 4
TRIAL_TYPES = (1, 2, 3, 4)
CONTEXT_KEYS = ("left", "right", "midline", "common", "lateral")
TRANSFER_PARAMETERS = {
    "threshold": 0.4,
    "lags": [-10, 0, 10],
    "penalty": 0.01,
    "gain_mode": "shared",
}


def _plain(value):
    """Convert diagnostics to strict-JSON-compatible built-in values."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _boundaries(boundaries, length):
    values = np.asarray(boundaries)
    if values.size == 0:
        return ()
    if (values.ndim != 1 or not np.issubdtype(values.dtype, np.integer)
            or np.any(values <= 0) or np.any(values >= length)
            or np.any(np.diff(values) <= 0)):
        raise ValueError("Calibration boundaries must be ordered interior integer trial joins")
    return tuple(int(value) for value in values)


def _validate_calibration(calibration, channel_names, regions, mask, recipe=None):
    eeg = np.asarray(calibration["eeg"], dtype=np.float64)
    references = np.asarray(calibration["references"], dtype=np.float64)
    labels = np.asarray(calibration["trial_types"])
    names = np.asarray(channel_names)
    region_values = np.asarray(regions)
    valid_channels = np.asarray(mask, dtype=bool)
    if (eeg.ndim != 2 or references.shape != (2, eeg.shape[-1])
            or labels.shape != (eeg.shape[-1],)
            or names.shape != (eeg.shape[0],) or region_values.shape != names.shape
            or valid_channels.shape != names.shape):
        raise ValueError("Calibration EEG, references, labels, or channel metadata axes differ")
    if not np.isfinite(eeg).all() or not np.isfinite(references).all():
        raise ValueError("Calibration EEG and references must be finite")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Calibration trial types must be integer labels")
    joins = _boundaries(calibration.get("boundaries", ()), eeg.shape[-1])
    if recipe is not None and (recipe.K != 3 or recipe.alpha != 2000 or recipe.fs != 200
            or recipe.tolerance != 1e-6 or recipe.max_iterations != 2000
            or recipe.initialization != "uniform" or recipe.normalization != "input_RMS"):
        raise ValueError("Calibration context requires the pinned frontal K3/alpha2000 recipe")
    # This also rejects guessed frontal labels and duplicate/nonfinite montage rows.
    frontal_indices(eeg, names, region_values, valid_channels)
    if not np.any(valid_channels & (region_values == 0)):
        raise ValueError("No verified available frontal channels")
    hemispheres = np.asarray(calibration.get("hemispheres", ()))
    if hemispheres.shape != names.shape:
        raise ValueError("Calibration hemisphere metadata axis differs")
    for index in np.flatnonzero(valid_channels & (region_values == 0)):
        if int(hemispheres[index]) != _hemisphere(str(names[index])):
            raise ValueError("Verified frontal hemisphere metadata differs from channel name")
    return eeg, references, labels, names, region_values, valid_channels, joins


def select_calibration_chunks(trial_types, boundaries, *, length=None):
    """Select at most four deterministic whole chunks without crossing joins.

    The first eligible chunk for each present known trial type is prioritized
    in type order 1..4. Remaining slots are filled chronologically. Runs of
    changing trial labels are treated as additional boundaries.
    """
    labels = np.asarray(trial_types)
    if labels.ndim != 1 or (length is not None and labels.shape != (length,)):
        raise ValueError("Trial-type labels must be a one-dimensional sample axis")
    joins = _boundaries(boundaries, len(labels))
    trial_edges = (0, *joins, len(labels))
    candidates = []
    for trial_index, (left, right) in enumerate(zip(trial_edges[:-1], trial_edges[1:])):
        # A label transition inside a declared trial is also kept as a hard edge.
        local_edges = np.flatnonzero(labels[left + 1:right] != labels[left:right - 1]) + left + 1
        run_edges = (left, *local_edges.tolist(), right)
        for run_left, run_right in zip(run_edges[:-1], run_edges[1:]):
            label = int(labels[run_left])
            if label not in TRIAL_TYPES:
                continue
            for start in range(run_left, run_right - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
                candidates.append({
                    "start": int(start),
                    "stop": int(start + CHUNK_SAMPLES),
                    "trial_index": int(trial_index),
                    "trial_type": label,
                })

    selected = []
    selected_starts = set()
    for label in TRIAL_TYPES:
        first = next((chunk for chunk in candidates if chunk["trial_type"] == label), None)
        if first is not None:
            chosen = dict(first)
            chosen["selection_reason"] = "first_chunk_for_trial_type"
            selected.append(chosen)
            selected_starts.add(chosen["start"])
            if len(selected) == MAX_CHUNKS:
                break
    for chunk in candidates:
        if len(selected) == MAX_CHUNKS:
            break
        if chunk["start"] in selected_starts:
            continue
        chosen = dict(chunk)
        chosen["selection_reason"] = "chronological_fill"
        selected.append(chosen)
        selected_starts.add(chosen["start"])
    return sorted(selected, key=lambda item: item["start"])


def _cache_identity(calibration_sha256, recipe):
    if not re.fullmatch(r"[0-9a-f]{64}", str(calibration_sha256)):
        raise ValueError("A lowercase 64-character calibration SHA-256 is required")
    recipe_values = asdict(recipe)
    recipe_hash = recipe.identity()
    implementation_hash = _implementation_hash()
    identity = {
        "cache_version": CACHE_VERSION,
        "representation": "vmd",
        "calibration_sha256": calibration_sha256,
        "recipe": recipe_values,
        "recipe_hash": recipe_hash,
        "implementation_sha256": implementation_hash,
        "solver_settings": {
            "fs": recipe.fs,
            "tolerance": recipe.tolerance,
            "max_iterations": recipe.max_iterations,
            "initialization": recipe.initialization,
            "normalization": recipe.normalization,
        },
        "chunk_samples": CHUNK_SAMPLES,
        "chunk_policy": "nonoverlapping full chunks inside trial and label runs",
        "max_chunks": MAX_CHUNKS,
    }
    return identity, canonical_hash(identity)


def _raw_identity(calibration_sha256):
    """Identify raw-reference calibration without invoking or caching VMD."""
    if not re.fullmatch(r"[0-9a-f]{64}", str(calibration_sha256)):
        raise ValueError("A lowercase 64-character calibration SHA-256 is required")
    identity = {
        "cache_version": CACHE_VERSION,
        "representation": "raw",
        "calibration_sha256": calibration_sha256,
        "implementation_sha256": _implementation_hash(),
        "transfer_parameters": TRANSFER_PARAMETERS,
        "chunk_samples": CHUNK_SAMPLES,
        "chunk_policy": "nonoverlapping full chunks inside trial and label runs",
        "max_chunks": MAX_CHUNKS,
    }
    return identity, canonical_hash(identity)


def _implementation_hash():
    """Bind passive caches to the decomposition and transfer implementations."""
    module_folder = Path(__file__).resolve().parent
    source_names = (
        "calibration_frontal_context.py",
        "frontal_calibration_transfer.py",
        "frontal_methods.py",
        "frontal.py",
        "vmd.py",
    )
    digest = hashlib.sha256()
    for name in source_names:
        path = module_folder / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_or_build_modes(calibration, selected, eeg, names, regions, mask,
                         config, recipe, cache_folder, identity, cache_hash):
    cache_folder = Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)
    stem = f"calibration-frontal-k3-{identity['calibration_sha256'][:16]}-{cache_hash[:16]}"
    array_path = cache_folder / f"{stem}.npz"
    metadata_path = cache_folder / f"{stem}.json"
    frontal_ids = frontal_indices(eeg, names, regions, mask)

    if array_path.exists() or metadata_path.exists():
        if not array_path.is_file() or not metadata_path.is_file():
            raise ValueError("Incomplete calibration frontal-mode cache pair")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, expected in identity.items():
            if metadata.get(key) != _plain(expected):
                raise ValueError("Calibration frontal-mode cache identity differs: " + key)
        if metadata.get("cache_hash") != cache_hash or metadata.get("arrays_sha256") != sha256_file(array_path):
            raise ValueError("Calibration frontal-mode cache checksum differs")
        if metadata.get("selected_chunks") != _plain(selected):
            raise ValueError("Calibration frontal-mode cache chunk selection differs")
        with np.load(array_path, allow_pickle=False) as saved:
            modes = saved["modes"].copy()
            residuals = saved["residuals"].copy()
            valid = saved["valid"].copy()
            cached_ids = saved["frontal_ids"].copy()
        expected_shape = (len(selected), len(frontal_ids), recipe.K, CHUNK_SAMPLES)
        if (modes.shape != expected_shape or residuals.shape != expected_shape[:2] + (CHUNK_SAMPLES,)
                or valid.shape != expected_shape[:2] or not np.array_equal(cached_ids, frontal_ids)):
            raise ValueError("Calibration frontal-mode cache arrays differ from montage or chunks")
        return modes, residuals, valid, frontal_ids, metadata

    modes = np.zeros((len(selected), len(frontal_ids), recipe.K, CHUNK_SAMPLES), dtype=np.float32)
    residuals = np.zeros((len(selected), len(frontal_ids), CHUNK_SAMPLES), dtype=np.float32)
    valid = np.zeros((len(selected), len(frontal_ids)), dtype=bool)
    diagnostics = []
    for chunk_index, chunk in enumerate(selected):
        start, stop = chunk["start"], chunk["stop"]
        bundle = decompose_frontal(
            eeg[:, start:stop], names, regions, recipe, config, mask,
        )
        channel_diagnostics = []
        for local, channel_index in enumerate(frontal_ids):
            vector, residual, info = bundle[int(channel_index)]
            failed = bool(info.get("failure") or info.get("hit_iteration_limit"))
            vector = np.asarray(vector, dtype=np.float32)
            residual = np.asarray(residual, dtype=np.float32)
            if vector.shape != (recipe.K, CHUNK_SAMPLES) or residual.shape != (CHUNK_SAMPLES,):
                raise ValueError("Frontal VMD returned an unexpected calibration chunk shape")
            if not failed and (not np.isfinite(vector).all() or not np.isfinite(residual).all()):
                failed = True
                info = {**info, "failure": "nonfinite cached calibration decomposition"}
            if not failed:
                reconstructed = vector.sum(axis=0, dtype=np.float64) + residual.astype(np.float64)
                if not np.allclose(reconstructed, eeg[channel_index, start:stop], rtol=1e-5, atol=1e-11):
                    failed = True
                    info = {**info, "failure": "calibration modes plus residual fail reconstruction"}
            if failed:
                # Keep bad vectors out of any later shared-gain fit.
                vector = np.zeros_like(vector)
                residual = eeg[channel_index, start:stop].astype(np.float32, copy=True)
            else:
                valid[chunk_index, local] = True
            modes[chunk_index, local] = vector
            residuals[chunk_index, local] = residual
            channel_diagnostics.append({
                "channel_index": int(channel_index),
                "channel_name": str(names[channel_index]),
                "status": "usable" if not failed else "failed_pass_through",
                "diagnostic": _plain(info),
            })
        diagnostics.append({"chunk_index": chunk_index, "start": start, "stop": stop,
                            "trial_index": chunk["trial_index"], "trial_type": chunk["trial_type"],
                            "channels": channel_diagnostics})

    _atomic_npz(array_path, modes=modes, residuals=residuals, valid=valid,
                frontal_ids=frontal_ids.astype(np.int64))
    metadata = {
        **identity,
        "cache_hash": cache_hash,
        "selected_chunks": _plain(selected),
        "frontal_ids": frontal_ids.tolist(),
        "frontal_names": [str(names[index]) for index in frontal_ids],
        "decomposition_diagnostics": diagnostics,
        "arrays_sha256": sha256_file(array_path),
        "cache_status": "passive_fixed_recipe_calibration_modes",
    }
    atomic_json(metadata_path, _plain(metadata))
    return modes, residuals, valid, frontal_ids, metadata


def _unavailable(eeg_channels, metadata, reason):
    details = {**metadata, "status": "unavailable", "available": False,
               "context_available": False, "failure": str(reason),
               "chunk_join_boundaries": []}
    return {
        "eeg": np.empty((eeg_channels, 0), dtype=np.float64),
        "predictors": np.empty((len(CONTEXT_KEYS), 0), dtype=np.float64),
        "boundaries": np.empty(0, dtype=np.int64),
        "metadata": details,
    }


def score_context(frontal_artifact, channel_names, regions, mask, hemispheres):
    """Summarize only whitelisted frontal artifact rows into five signed rows.

    Returns ``(context, flags)`` where context is ordered left, right,
    midline, common, lateral. Nonfrontal values in the supplied artifact are a
    contract violation rather than silent predictor leakage.
    """
    artifact = np.asarray(frontal_artifact, dtype=np.float64)
    names = np.asarray(channel_names)
    region_values = np.asarray(regions)
    channel_mask = np.asarray(mask, dtype=bool)
    side_values = np.asarray(hemispheres)
    if (artifact.ndim != 2 or names.shape != (len(artifact),)
            or region_values.shape != names.shape or channel_mask.shape != names.shape
            or side_values.shape != names.shape or not np.isfinite(artifact).all()):
        raise ValueError("Frontal artifact or metadata axes are invalid")
    frontal_ids = frontal_indices(artifact, names, region_values, channel_mask)
    for index in frontal_ids:
        if int(side_values[index]) != _hemisphere(str(names[index])):
            raise ValueError("Verified frontal hemisphere metadata differs from channel name")
    allowed = np.zeros(len(artifact), dtype=bool)
    allowed[frontal_ids] = True
    if np.any(artifact[~allowed] != 0):
        raise ValueError("Frontal context artifact has nonzero values outside verified masked frontal rows")
    usable_regions = np.where(allowed, region_values, -1)
    usable_hemispheres = np.where(allowed, side_values, 3)
    context = signed_context(artifact, names, usable_regions, usable_hemispheres)
    features = np.stack([context[key] for key in CONTEXT_KEYS])
    available = context["available"]
    counts = {}
    for side, key in ((0, "left"), (1, "right"), (2, "midline")):
        indices = frontal_ids[side_values[frontal_ids] == side]
        counts[key] = int(len(indices))
    lateral_available = bool(available[0] and available[1])
    flags = {
        "feature_order": list(CONTEXT_KEYS),
        "left_available": bool(available[0]),
        "right_available": bool(available[1]),
        "midline_available": bool(available[2]),
        "common_available": bool(any(available)),
        "lateral_available": lateral_available,
        "frontal_channel_counts": counts,
        "context_nonzero": bool(np.any(features != 0)),
        "context_available": bool(len(frontal_ids) and np.any(features != 0)),
        "available": bool(len(frontal_ids) and np.any(features != 0)),
        "nonfrontal_input_used": False,
    }
    return features, flags


def fit_context_calibration(calibration, config, destination, cache_id, representation="vmd"):
    """Build OOF calibration EEG/context arrays for VMD or raw references.

    The returned dict has ``eeg [C,N]``, ``predictors [5,N]``,
    ``boundaries`` between selected chunks, and auditable ``metadata``. The
    default VMD representation uses a passive K3 mode cache; raw-reference
    context passes ``bundles=None`` and never creates or reads that cache.
    A failed fit makes the whole context unavailable and returns empty arrays;
    a valid zero artifact for an individual rest chunk remains a zero predictor
    block so downstream ridge fitting can learn an identity-preserving case.
    """
    if representation not in ("vmd", "raw"):
        try:
            eeg_channels = int(np.asarray(calibration.get("eeg", np.empty((0, 0)))).shape[0])
        except Exception:
            eeg_channels = 0
        return _unavailable(eeg_channels, {
            "cache_version": CACHE_VERSION,
            "representation": str(representation),
            "calibration_sha256": str(cache_id),
            "reserved_confirmation_opened": False,
            "paired_target_access": False,
            "scoring_eeg_access": False,
        }, "unknown calibration context representation")
    recipe = None
    try:
        if representation == "vmd":
            recipe = FixedRecipe(3, 2000, fs=int(config.get("fs", 200)))
        elif int(config.get("fs", 200)) != 200:
            raise ValueError("Calibration context requires the fixed 200 Hz project sampling rate")
        if int(config.get("window", -1)) != CHUNK_SAMPLES:
            raise ValueError("Calibration context requires the fixed 1024-sample project window")
        channel_names = calibration["channel_names"]
        regions = calibration["regions"]
        mask = calibration["mask"]
        hemispheres = calibration["hemispheres"]
        eeg, references, labels, names, region_values, channel_mask, joins = _validate_calibration(
            calibration, channel_names, regions, mask, recipe,
        )
        if representation == "vmd":
            identity, cache_hash = _cache_identity(cache_id, recipe)
        else:
            identity, cache_hash = _raw_identity(cache_id)
    except Exception as error:
        try:
            eeg_channels = int(np.asarray(calibration.get("eeg", np.empty((0, 0)))).shape[0])
        except Exception:
            eeg_channels = 0
        return _unavailable(eeg_channels, {
            "cache_version": CACHE_VERSION,
            "representation": representation,
            "calibration_sha256": str(cache_id),
            "reserved_confirmation_opened": False,
            "paired_target_access": False,
            "scoring_eeg_access": False,
        }, f"invalid calibration contract: {type(error).__name__}: {error}")
    selected = select_calibration_chunks(labels, joins, length=eeg.shape[-1])
    base_quality = {
        **identity,
        "cache_hash": cache_hash,
        "representation": representation,
        "transfer_parameters": TRANSFER_PARAMETERS,
        "crossfit_policy": "leave-one-selected-chunk-out; complement chunks concatenated with explicit joins",
        "max_chunk_adaptation": "at most four nonoverlapping 1024-sample chunks; selected types prioritized 1..4",
        "calibration_sha256": cache_id,
        "recipe_hash": recipe.identity() if recipe is not None else None,
        "selected_chunks": _plain(selected),
        "chunk_join_boundaries": [],
        "reserved_confirmation_opened": False,
        "paired_target_access": False,
        "scoring_eeg_access": False,
    }
    if len(selected) < 2:
        return _unavailable(eeg.shape[0], base_quality,
                            "fewer than two eligible whole calibration chunks")

    cache_metadata = None
    modes = residuals = valid_modes = frontal_ids = None
    if representation == "vmd":
        try:
            modes, residuals, valid_modes, frontal_ids, cache_metadata = _load_or_build_modes(
                calibration, selected, eeg, names, region_values, channel_mask,
                config, recipe, destination, identity, cache_hash,
            )
        except Exception as error:
            return _unavailable(eeg.shape[0], base_quality,
                                f"frontal calibration mode cache failed: {type(error).__name__}: {error}")
    else:
        frontal_ids = frontal_indices(eeg, names, region_values, channel_mask)

    chunks_eeg, chunks_artifact, chunks_quality = [], [], []
    for heldout_index, chunk in enumerate(selected):
        training_indices = [index for index in range(len(selected)) if index != heldout_index]
        training_eeg = np.concatenate([eeg[:, selected[index]["start"]:selected[index]["stop"]]
                                       for index in training_indices], axis=-1)
        training_refs = np.concatenate([references[:, selected[index]["start"]:selected[index]["stop"]]
                                        for index in training_indices], axis=-1)
        training_boundaries = np.arange(1, len(training_indices), dtype=np.int64) * CHUNK_SAMPLES
        training = {"eeg": training_eeg, "references": training_refs,
                    "boundaries": training_boundaries}
        start, stop = chunk["start"], chunk["stop"]
        bundle = None
        if representation == "vmd":
            bundle = {
                int(channel_index): (
                    modes[heldout_index, local].copy(),
                    residuals[heldout_index, local].copy(),
                    {"failure": "cached calibration VMD failure"} if not valid_modes[heldout_index, local] else {},
                )
                for local, channel_index in enumerate(frontal_ids)
            }
        try:
            _, artifact, details = calibration_transfer(
                eeg[:, start:stop], references[:, start:stop], training,
                names, region_values, channel_mask, bundles=bundle,
                threshold=TRANSFER_PARAMETERS["threshold"],
                lags=tuple(TRANSFER_PARAMETERS["lags"]),
                penalty=TRANSFER_PARAMETERS["penalty"],
                gain_mode=TRANSFER_PARAMETERS["gain_mode"],
            )
            if artifact.shape != (eeg.shape[0], CHUNK_SAMPLES) or not np.isfinite(artifact).all():
                raise ValueError("calibration transfer returned invalid artifact axes or values")
            effective_mask = channel_mask.copy()
            if representation == "vmd":
                effective_mask[frontal_ids[~valid_modes[heldout_index]]] = False
                if not np.any(effective_mask[frontal_ids]):
                    raise ValueError("held-out chunk has no usable frontal VMD channels")
            feature, feature_flags = score_context(
                artifact, names, region_values, effective_mask, hemispheres,
            )
        except Exception as error:
            return _unavailable(eeg.shape[0], {
                **base_quality,
                "cache_metadata": cache_metadata,
                "crossfit_chunks_completed": chunks_quality,
            }, f"cross-fit failed on chunk {heldout_index}: {type(error).__name__}: {error}")
        chunks_eeg.append(eeg[:, start:stop].copy())
        chunks_artifact.append(feature)
        chunks_quality.append({
            **chunk,
            "heldout_index": heldout_index,
            "training_chunk_indices": training_indices,
            "feature_flags": feature_flags,
            "frontal_transfer_decisions": _plain(details),
        })

    calibration_eeg = np.concatenate(chunks_eeg, axis=-1)
    context = np.concatenate(chunks_artifact, axis=-1)
    output_boundaries = np.arange(1, len(selected), dtype=np.int64) * CHUNK_SAMPLES
    if not np.isfinite(context).all() or not np.any(context != 0):
        return _unavailable(eeg.shape[0], {
            **base_quality,
            "cache_metadata": cache_metadata,
            "representation": representation,
            "crossfit_chunks_completed": chunks_quality,
        }, "cross-fitted frontal context is entirely zero")
    return {
        "eeg": calibration_eeg,
        "predictors": context,
        "boundaries": output_boundaries,
        "metadata": {
            **base_quality,
            "status": "available",
            "available": True,
            "context_available": True,
            "cache_metadata": cache_metadata,
            "selected_chunks": _plain(selected),
            "chunk_join_boundaries": output_boundaries.tolist(),
            "crossfit_chunks": chunks_quality,
            "context_feature_order": list(CONTEXT_KEYS),
            "context_shape": list(context.shape),
            "calibration_eeg_shape": list(calibration_eeg.shape),
            "context_nonzero": True,
        },
    }
