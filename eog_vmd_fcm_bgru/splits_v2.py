"""Fail-closed split and exposure governance for the campaign-v2 corpus."""
from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from .campaign_contracts import atomic_json, canonical_hash


def _stable_order(values: Iterable[str], seed: int) -> list[str]:
    return sorted(values, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())


def split_unit(record: Mapping[str, Any]) -> str:
    participant = record.get("participant_id")
    if participant and record.get("participant_verification") == "verified":
        return "participant:" + str(participant)
    group = record.get("clean_target_group")
    if group:
        return "clean-target-group:" + str(group)
    return "recording:" + str(record["record_id"])


def _assign_groups(groups: list[str], fractions: Mapping[str, float], seed: int) -> dict[str, str]:
    required = {"train", "val", "test"}
    if set(fractions) != required or any(float(fractions[name]) <= 0 for name in required):
        raise ValueError("fractions must contain positive train, val and test values")
    total = sum(float(fractions[name]) for name in required)
    ordered = _stable_order(groups, seed)
    counts = {name: int(round(len(ordered) * float(fractions[name]) / total)) for name in required}
    while sum(counts.values()) > len(ordered):
        counts[max(counts, key=counts.get)] -= 1
    while sum(counts.values()) < len(ordered):
        counts[min(counts, key=counts.get)] += 1
    assignment, position = {}, 0
    for partition in ("train", "val", "test"):
        for group in ordered[position:position + counts[partition]]:
            assignment[group] = partition
        position += counts[partition]
    return assignment


def validate_split_manifest(manifest: Mapping[str, Any], recordings: Iterable[Mapping[str, Any]]) -> None:
    assignment = manifest["assignment"]
    seen = set()
    for record in recordings:
        record_id = record["record_id"]
        if record_id not in assignment:
            raise ValueError("Split omits recording " + record_id)
        partition = assignment[record_id]
        if partition not in {"train", "val", "test"}:
            raise ValueError("Unknown partition")
        key = split_unit(record)
        previous = seen.intersection({record_id})
        if previous:
            raise ValueError("Duplicate recording in split")
        seen.add(record_id)
    by_group: dict[str, set[str]] = defaultdict(set)
    for record in recordings:
        by_group[split_unit(record)].add(assignment[record["record_id"]])
    leaked = {group: values for group, values in by_group.items() if len(values) != 1}
    if leaked:
        raise ValueError("Split leaks group(s): " + ", ".join(sorted(leaked)))


def freeze_splits(recordings: list[Mapping[str, Any]], config: Mapping[str, Any], output: Path,
                  exposure_path: Path | None = None) -> dict[str, Any]:
    """Freeze participant/clean-target groups before windows or donor recipes exist."""
    if not recordings:
        raise ValueError("Cannot split an empty recording manifest")
    forbidden = set(config.get("final_eval_record_ids", []))
    previous_exposure = []
    if exposure_path is not None and Path(exposure_path).exists():
        import json
        previous_exposure = [json.loads(line) for line in Path(exposure_path).read_text(encoding="utf-8").splitlines() if line]
    exposed = {row["record_id"] for row in previous_exposure
               if row.get("purpose") in {"teacher_fit", "student_train", "selection", "calibration"}}
    if forbidden.intersection(exposed):
        raise ValueError("Final-evaluation recording already exposed to development")
    groups = sorted({split_unit(record) for record in recordings})
    assignment_group = _assign_groups(groups, config.get("fractions", {"train": .70, "val": .15, "test": .15}),
                                      int(config.get("seed", 42)))
    assignment = {record["record_id"]: assignment_group[split_unit(record)] for record in recordings}
    if forbidden:
        for record_id in forbidden:
            if record_id not in assignment:
                raise ValueError("Unknown requested final evaluation record")
            assignment[record_id] = "test"
        # Final records must comprise full group(s), never force a participant apart.
        final_groups = {split_unit(record) for record in recordings if record["record_id"] in forbidden}
        for record in recordings:
            if split_unit(record) in final_groups:
                assignment[record["record_id"]] = "test"
    manifest = {"schema_version": 2, "kind": "split_manifest", "recording_manifest_hash": canonical_hash(recordings),
                "split_unit_policy": "verified participant, then duplicate clean target group, then recording",
                "independence_limits": "Klados has unverified participants and rows; its groups are engineering groups only.",
                "seed": int(config.get("seed", 42)), "fractions": config.get("fractions", {"train": .70, "val": .15, "test": .15}),
                "final_eval_record_ids": sorted(forbidden), "assignment": assignment}
    validate_split_manifest(manifest, recordings)
    manifest["split_hash"] = canonical_hash(manifest)
    atomic_json(Path(output) / "split_manifest.json", manifest)
    return manifest


def assert_oof_fit_ids(fit_record_ids: Iterable[str], scored_record_ids: Iterable[str],
                       *, source: str = "teacher") -> None:
    overlap = set(fit_record_ids).intersection(scored_record_ids)
    if overlap:
        raise ValueError(f"{source} OOF fit/scored record overlap: {sorted(overlap)}")


def append_exposure_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Append only new immutable exposure rows; duplicate ids are a hard error."""
    import json
    path = Path(path)
    old = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    keys = {(row["record_id"], row["purpose"], row["run_id"]) for row in old}
    additions = []
    for row in rows:
        required = {"record_id", "purpose", "run_id", "stage"}
        if required.difference(row):
            raise ValueError("Exposure row missing required fields")
        key = (row["record_id"], row["purpose"], row["run_id"])
        if key in keys:
            raise ValueError("Duplicate exposure row")
        keys.add(key)
        additions.append(dict(row))
    payload = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in old + additions)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
