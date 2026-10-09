"""Kaggle-stage orchestration for the regional classical teacher.

This module consumes only materialized corpus-manifest entries.  It has no
knowledge of the source datasets and never creates a test-set search result.
The public entry point is intentionally narrow so the campaign runner can own
provenance, submission and final result assembly.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .channel_regions import region_ids
from .evaluation import modification_metrics, paired_metrics
from .provenance import save_json
from .regional_teacher import FrontalVMDConfig, frontal_vmd_estimate


DEFAULT_K = tuple(range(3, 11))
DEFAULT_ALPHA = (250, 500, 1000, 2000, 4000)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, default=_json_value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Corpus manifest is unavailable: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid corpus manifest JSON at line {line_number}") from error
    if not rows:
        raise ValueError("Corpus manifest contains no entries")
    return rows


def _resolve_array_path(entry: dict[str, Any], data_root: Path, manifest: Path) -> Path:
    value = entry.get("array_path")
    if not value:
        raise ValueError(f"Corpus entry {entry.get('example_id')} is recipe-only; materialized NPZ is required")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for root in (data_root, manifest.parent):
        path = root / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot locate corpus NPZ for {entry.get('example_id')}: {value}")


def _load_example(entry: dict[str, Any], data_root: Path, manifest: Path) -> dict[str, Any]:
    array_path = _resolve_array_path(entry, data_root, manifest)
    with np.load(array_path, allow_pickle=False) as archive:
        required = {"eeg", "mask", "regions", "hemispheres"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Corpus NPZ {array_path} is missing {sorted(missing)}")
        eeg = np.asarray(archive["eeg"], dtype=np.float64)
        if eeg.ndim != 2 or eeg.shape[1] < 32 or not np.isfinite(eeg).all():
            raise ValueError(f"Corpus EEG must be finite [channels, samples]: {array_path}")
        mask = np.asarray(archive["mask"], dtype=bool)
        regions = np.asarray(archive["regions"], dtype=np.int64)
        hemispheres = np.asarray(archive["hemispheres"])
        if mask.shape != (eeg.shape[0],) or regions.shape != (eeg.shape[0],):
            raise ValueError(f"Corpus metadata does not align with EEG: {array_path}")
        references = np.asarray(archive["references"], dtype=np.float64) if "references" in archive.files else None
        if references is not None and (references.shape != (2, eeg.shape[1]) or not np.isfinite(references).all()):
            raise ValueError(f"Corpus references must be finite ordered [HEOG,VEOG]: {array_path}")
        paired = np.asarray(archive["paired_reference"], dtype=np.float64) if "paired_reference" in archive.files else None
        if paired is not None and paired.shape != eeg.shape:
            raise ValueError(f"Paired reference does not align with EEG: {array_path}")
        stored_names = archive["channel_names"].tolist() if "channel_names" in archive.files else None
    names = entry.get("channel_names", stored_names)
    if names is None:
        # Region IDs may be valid without names, but VMD routing must not infer
        # names.  Synthetic stable names preserve the supplied region array
        # through the validated route below.
        names = [f"UNKNOWN_{index}" for index in range(eeg.shape[0])]
    if len(names) != eeg.shape[0]:
        raise ValueError(f"Channel names do not align with EEG: {array_path}")
    return {**entry, "array_file": array_path, "eeg": eeg, "mask": mask, "regions": regions,
            "hemispheres": hemispheres, "references": references, "paired_reference": paired, "names": list(names)}


def _named_regions(example: dict[str, Any]) -> list[str]:
    """Use names only when their region dictionary agrees with corpus metadata."""
    names = example["names"]
    supplied = example["regions"]
    computed = region_ids(names)
    if any(name.startswith("UNKNOWN_") for name in names):
        # Do not invent anatomy: unknown names are not eligible for a frontal
        # VMD route even if a corpus producer supplied a region integer.
        return names
    mismatch = (computed != supplied) & example["mask"]
    if mismatch.any():
        raise ValueError("Named channel regions conflict with corpus metadata")
    return names


def _grid(config: dict[str, Any]) -> list[tuple[int, float]]:
    search = config.get("vmd_grid", {})
    values_k = tuple(int(value) for value in search.get("K", DEFAULT_K))
    values_alpha = tuple(float(value) for value in search.get("alpha", DEFAULT_ALPHA))
    grid = [(k, alpha) for k in values_k for alpha in values_alpha]
    if len(grid) != 40 and not config.get("allow_nonstandard_grid", False):
        raise ValueError("Teacher VMD grid must contain the declared 40 K/alpha candidates")
    if any(k < 2 for k, _ in grid) or any(alpha <= 0 for _, alpha in grid):
        raise ValueError("Teacher VMD grid contains invalid K/alpha values")
    return grid


def _train_entries(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = [row for row in rows if row.get("split_role") == "train"]
    if not entries:
        raise ValueError("Teacher cache requires materialized train entries")
    return entries


def _cache_path(output: Path, example_id: str, k: int, alpha: float) -> Path:
    token = str(alpha).replace(".", "p")
    return output / "vmd_cache" / f"{example_id}__k{k}__a{token}.npz"


def teacher_cache_stage(data_root: Path, output: Path, config: dict[str, Any], parents: dict[str, Any]) -> dict[str, Any]:
    """Cache canonical RMS VMD modes for train entries only."""
    manifest = Path(parents["corpus_manifest"])
    rows = _load_jsonl(manifest)
    output.mkdir(parents=True, exist_ok=True)
    (output / "vmd_cache").mkdir(exist_ok=True)
    diagnostics: list[dict[str, Any]] = []
    for entry in _train_entries(rows):
        example = _load_example(entry, data_root, manifest)
        names = _named_regions(example)
        frontal = np.flatnonzero((region_ids(names) == 0) & example["mask"])
        for k, alpha in _grid(config):
            destination = _cache_path(output, str(entry["example_id"]), k, alpha)
            if destination.exists():
                diagnostics.append({"example_id": entry["example_id"], "K": k, "alpha": alpha,
                                    "cache_path": str(destination.relative_to(output)), "status": "reused"})
                continue
            modes, residuals, channel_diagnostics = [], [], []
            for channel in frontal:
                started = time.perf_counter()
                try:
                    from .vmd_rolling import rolling_decompose
                    vector, residual, detail = rolling_decompose(
                        example["eeg"][channel], k, alpha,
                        tolerance=float(config.get("tolerance", 1e-6)),
                        max_iterations=int(config.get("max_iterations", 2000)),
                    )
                    modes.append(vector)
                    residuals.append(residual)
                    channel_diagnostics.append({"channel": int(channel), **detail})
                except Exception as error:
                    channel_diagnostics.append({"channel": int(channel), "error": repr(error), "runtime_s": time.perf_counter() - started})
            np.savez_compressed(destination, channels=frontal, modes=np.asarray(modes, dtype=np.float32),
                                residuals=np.asarray(residuals, dtype=np.float32))
            diagnostics.extend({"example_id": entry["example_id"], "K": k, "alpha": alpha,
                                "cache_path": str(destination.relative_to(output)), **row}
                               for row in channel_diagnostics)
    frame = pd.DataFrame(diagnostics)
    frame.to_csv(output / "teacher_cache_diagnostics.csv", index=False)
    save_json(output / "teacher_cache_manifest.json", {
        "scope": "train-only VMD mode cache", "grid": [{"K": k, "alpha": alpha} for k, alpha in _grid(config)],
        "rows": int(len(diagnostics)), "corpus_manifest": str(manifest),
    })
    return {"cache_diagnostics": output / "teacher_cache_diagnostics.csv", "cache_root": output / "vmd_cache"}


def _candidate_configs(config: dict[str, Any]) -> Iterable[FrontalVMDConfig]:
    correction = config.get("candidate_grid", {})
    thresholds = correction.get("threshold", (0.2, 0.4, 0.6, 0.8))
    strengths = correction.get("strength", (0.25, 0.5, 0.75, 1.0))
    penalties = correction.get("projection_penalty", (0.01,))
    lags = tuple(int(value) for value in correction.get("lags", (0,)))
    for k, alpha in _grid(config):
        for threshold in thresholds:
            for strength in strengths:
                for penalty in penalties:
                    yield FrontalVMDConfig(
                        modes=k, alpha=alpha, tolerance=float(config.get("tolerance", 1e-6)),
                        max_iterations=int(config.get("max_iterations", 2000)),
                        correlation_threshold=float(threshold), strength=float(strength),
                        projection_penalty=float(penalty), lags=lags,
                    )


def teacher_search_stage(data_root: Path, output: Path, config: dict[str, Any], parents: dict[str, Any]) -> dict[str, Any]:
    """Search teacher candidates on development roles only; never final test."""
    manifest = Path(parents["corpus_manifest"])
    rows = _load_jsonl(manifest)
    development_roles = set(config.get("search_roles", ("val",)))
    entries = [entry for entry in rows if entry.get("split_role") in development_roles]
    if not entries:
        raise ValueError("Teacher search has no declared development entries")
    if any(entry.get("split_role") == "test" for entry in entries):
        raise ValueError("Final test examples are forbidden in teacher search")
    output.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []
    for entry in entries:
        example = _load_example(entry, data_root, manifest)
        if example["references"] is None:
            candidate_rows.append({"example_id": entry["example_id"], "status": "skipped_no_heog_veog"})
            continue
        names = _named_regions(example)
        for candidate in _candidate_configs(config):
            try:
                estimate = frontal_vmd_estimate(example["eeg"], example["references"], names, candidate,
                                                valid_mask=example["mask"])
                corrected = example["eeg"] - estimate.artifact
                row = {"example_id": entry["example_id"], "record_id": entry.get("record_id"),
                       "recipient_id": entry.get("recipient_id"), "donor_id": entry.get("donor_id"),
                       "split_role": entry.get("split_role"), **asdict(candidate), "status": "ok",
                       "vmd_converged_channels": int(sum(item.get("accepted", False) for item in estimate.diagnostics["channels"])),
                       "vmd_passthrough_channels": int(sum(not item.get("accepted", False) for item in estimate.diagnostics["channels"]))}
                if example["paired_reference"] is not None:
                    row.update(paired_metrics(corrected, example["paired_reference"]))
                    # Clean counterfactual uses EOG only to quantify privileged
                    # teacher risk; it never gives the teacher a clean target.
                    clean_estimate = frontal_vmd_estimate(example["paired_reference"], example["references"], names, candidate,
                                                         valid_mask=example["mask"])
                    clean_corrected = example["paired_reference"] - clean_estimate.artifact
                    row.update({f"clean_{key}": value for key, value in modification_metrics(clean_corrected, example["paired_reference"]).items()})
                    row["clean_alpha_error_db"] = paired_metrics(clean_corrected, example["paired_reference"])["alpha_error_db"]
                    row["clean_beta_error_db"] = paired_metrics(clean_corrected, example["paired_reference"])["beta_error_db"]
                candidate_rows.append(row)
            except Exception as error:
                candidate_rows.append({"example_id": entry["example_id"], **asdict(candidate), "status": "failed", "error": repr(error)})
    details = pd.DataFrame(candidate_rows)
    details.to_csv(output / "teacher_search_windows.csv", index=False)
    good = details[details.status == "ok"].copy() if "status" in details else pd.DataFrame()
    selection: dict[str, Any] = {"selected": None, "search_roles": sorted(development_roles), "final_test_scored": False}
    if not good.empty:
        keys = ["modes", "alpha", "correlation_threshold", "projection_penalty", "lags", "strength"]
        summary = good.groupby(keys, dropna=False, as_index=False).mean(numeric_only=True)
        if "rmse" in summary:
            feasible = summary.copy()
            if "clean_relative_change" in feasible:
                feasible = feasible[feasible.clean_relative_change <= float(config.get("max_clean_relative_change", 0.01))]
            if "clean_alpha_error_db" in feasible:
                feasible = feasible[np.abs(feasible.clean_alpha_error_db) <= float(config.get("max_clean_band_error_db", 0.5))]
            if "clean_beta_error_db" in feasible:
                feasible = feasible[np.abs(feasible.clean_beta_error_db) <= float(config.get("max_clean_band_error_db", 0.5))]
            if not feasible.empty:
                selected = feasible.sort_values(["rmse", "modes", "alpha", "strength"]).iloc[0].to_dict()
                selection["selected"] = selected
        summary.to_csv(output / "teacher_search_summary.csv", index=False)
    save_json(output / "teacher_selection.json", selection)
    return {"search_windows": output / "teacher_search_windows.csv", "selection": output / "teacher_selection.json"}


def _selection_config(parents: dict[str, Any], config: dict[str, Any]) -> FrontalVMDConfig:
    selection_path = parents.get("teacher_selection")
    selection = None
    if selection_path:
        selection = json.loads(Path(selection_path).read_text(encoding="utf-8")).get("selected")
    selection = selection or config.get("selected_teacher")
    if not selection:
        raise ValueError("Teacher OOF requires a frozen selected teacher recipe")
    allowed = {field for field in FrontalVMDConfig.__dataclass_fields__}
    return FrontalVMDConfig(**{key: value for key, value in selection.items() if key in allowed})


def teacher_oof_stage(data_root: Path, output: Path, config: dict[str, Any], parents: dict[str, Any]) -> dict[str, Any]:
    """Materialize EEG-aligned frontal teacher targets for permitted OOF roles."""
    manifest = Path(parents["corpus_manifest"])
    rows = _load_jsonl(manifest)
    candidate = _selection_config(parents, config)
    roles = set(config.get("oof_roles", ("train", "val")))
    fit_record_ids = set(map(str, config.get("fit_record_ids", ())))
    recipe = {"frontal_vmd": asdict(candidate), "fit_record_ids": sorted(fit_record_ids),
              "teacher_type": "EOG-assisted window-adaptive VMD projection"}
    recipe_hash = _canonical_hash(recipe)
    arrays_root = output / "teacher_arrays"
    arrays_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for entry in rows:
        if entry.get("split_role") not in roles:
            continue
        example = _load_example(entry, data_root, manifest)
        example_id = str(entry["example_id"])
        recipient = str(entry.get("recipient_id", entry.get("record_id", "")))
        donor = entry.get("donor_id")
        forbidden = {recipient}
        if donor is not None:
            forbidden.add(str(donor))
        if fit_record_ids.intersection(forbidden):
            raise ValueError(f"OOF teacher fit IDs overlap recipient/donor for example {example_id}")
        teacher_oof = example["references"] is not None
        if teacher_oof:
            names = _named_regions(example)
            estimate = frontal_vmd_estimate(example["eeg"], example["references"], names, candidate,
                                            valid_mask=example["mask"])
            artifact = estimate.artifact.astype(np.float32)
            weight = np.broadcast_to(estimate.confidence[:, None], artifact.shape).astype(np.float32)
            diagnostics = estimate.diagnostics
        else:
            artifact = np.zeros_like(example["eeg"], dtype=np.float32)
            weight = np.zeros_like(artifact)
            diagnostics = {"reason": "HEOG/VEOG unavailable; teacher disabled"}
        destination = arrays_root / f"{example_id}.npz"
        np.savez_compressed(destination, teacher_artifact=artifact, teacher_weight=weight,
                            teacher_oof=np.asarray(teacher_oof), valid_mask=example["mask"])
        row = {"example_id": example_id, "array_path": str(destination.relative_to(output)), "teacher_oof": bool(teacher_oof),
               "fit_record_ids": sorted(fit_record_ids), "recipe_hash": recipe_hash,
               "recipient_id": recipient, "donor_id": donor, "record_id": entry.get("record_id"),
               "split_role": entry.get("split_role"), "diagnostics": diagnostics}
        manifest_rows.append(row)
        metric = {"example_id": example_id, "teacher_oof": bool(teacher_oof)}
        if teacher_oof and example["paired_reference"] is not None:
            metric.update(paired_metrics(example["eeg"] - artifact, example["paired_reference"]))
        metric_rows.append(metric)
    with (output / "teacher_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, default=_json_value, sort_keys=True) + "\n")
    pd.DataFrame(metric_rows).to_csv(output / "teacher_oof_metrics.csv", index=False)
    save_json(output / "teacher_recipe.json", {**recipe, "recipe_hash": recipe_hash, "oof_roles": sorted(roles)})
    return {"teacher_manifest": output / "teacher_manifest.jsonl", "teacher_metrics": output / "teacher_oof_metrics.csv"}


def run_teacher_stage(stage, data_root, output, config, parents):
    """Campaign runner entry point for ``teacher-cache``, ``teacher-search`` and ``teacher-oof``.

    ``parents['corpus_manifest']`` must point to a materialized corpus manifest.
    ``teacher-oof`` additionally needs ``parents['teacher_selection']`` or a
    ``selected_teacher`` config block.  The function intentionally does not
    offer a final-test stage.
    """
    data_root, output = Path(data_root), Path(output)
    config = dict(config or {})
    parents = dict(parents or {})
    if "corpus_manifest" not in parents:
        raise ValueError("Teacher stage requires parents['corpus_manifest']")
    if stage == "teacher-cache":
        return teacher_cache_stage(data_root, output, config, parents)
    if stage == "teacher-search":
        return teacher_search_stage(data_root, output, config, parents)
    if stage == "teacher-oof":
        return teacher_oof_stage(data_root, output, config, parents)
    raise ValueError(f"Unknown teacher stage: {stage}")

