"""Immutable campaign specifications and small provenance-safe file helpers.

This module intentionally has no signal-processing imports.  It is used by
Kaggle jobs to make configuration identity and parent-artifact checks explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


SCHEMA_VERSION = 2
STAGES = frozenset({
    "contracts",
    "provenance", "split-freeze", "corpus-fit", "corpus-build",
    "teacher-cache", "teacher-search", "teacher-oof", "student-paired",
    "student-distill", "select-freeze", "final-eval", "export",
})
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def canonical_json_bytes(value: Any) -> bytes:
    """Stable JSON encoding used for hashes; floats must be ordinary JSON values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> str:
    """Atomically publish JSON and return its canonical-content SHA256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload.rstrip(b"\n")).hexdigest()


def read_json_verified(path: Path, expected_hash: str | None = None) -> Any:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if expected_hash is not None and canonical_hash(value) != expected_hash:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return value


@dataclass(frozen=True)
class RunSpec:
    schema_version: int
    campaign_id: str
    run_id: str
    stage: str
    git_sha: str
    config: dict[str, Any]
    inputs: dict[str, Any]
    fold: int | None = None
    seed: int = 42
    shard: str | None = None
    config_hash: str | None = None
    data_hash: str | None = None
    split_hash: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunSpec":
        required = {"campaign_id", "run_id", "stage", "git_sha", "config", "inputs"}
        missing = required.difference(value)
        if missing:
            raise ValueError("RunSpec missing required fields: " + ", ".join(sorted(missing)))
        spec = cls(schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
                   campaign_id=str(value["campaign_id"]), run_id=str(value["run_id"]),
                   stage=str(value["stage"]), git_sha=str(value["git_sha"]),
                   config=dict(value["config"]), inputs=dict(value["inputs"]),
                   fold=value.get("fold"), seed=int(value.get("seed", 42)),
                   shard=value.get("shard"), config_hash=value.get("config_hash"),
                   data_hash=value.get("data_hash"), split_hash=value.get("split_hash"))
        spec.validate()
        return spec

    @classmethod
    def load(cls, path: Path) -> "RunSpec":
        return cls.from_dict(read_json_verified(path))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "campaign_id": self.campaign_id,
                "run_id": self.run_id, "stage": self.stage, "git_sha": self.git_sha,
                "config": self.config, "inputs": self.inputs, "fold": self.fold,
                "seed": self.seed, "shard": self.shard, "config_hash": self.config_hash,
                "data_hash": self.data_hash, "split_hash": self.split_hash}

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported RunSpec schema {self.schema_version}")
        for name, value in (("campaign_id", self.campaign_id), ("run_id", self.run_id)):
            if not _ID.fullmatch(value):
                raise ValueError(f"Invalid {name}")
        if self.stage not in STAGES:
            raise ValueError(f"Unsupported stage: {self.stage}")
        if not re.fullmatch(r"[0-9a-f]{40}", self.git_sha):
            raise ValueError("git_sha must be a lowercase 40-character SHA")
        if self.fold is not None and (not isinstance(self.fold, int) or self.fold < 0):
            raise ValueError("fold must be a nonnegative integer or null")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.shard is not None and not _ID.fullmatch(str(self.shard)):
            raise ValueError("Invalid shard")
        computed = canonical_hash(self.config)
        if self.config_hash is not None and self.config_hash != computed:
            raise ValueError("RunSpec config_hash does not match config")
        for name, value in (("data_hash", self.data_hash), ("split_hash", self.split_hash)):
            if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} must be a SHA256 or null")

    @property
    def resolved_config_hash(self) -> str:
        return canonical_hash(self.config)

    def write(self, path: Path) -> str:
        value = self.to_dict()
        value["config_hash"] = self.resolved_config_hash
        return atomic_json(path, value)


def artifact_manifest(kind: str, payload: Mapping[str, Any], *, config_hash: str,
                      parents: Mapping[str, Path] | None = None) -> dict[str, Any]:
    """Create a compact, hashable record of artifact ancestry without reading data."""
    if not _ID.fullmatch(kind):
        raise ValueError("Invalid artifact kind")
    parent_hashes = {}
    for name, path in (parents or {}).items():
        if not _ID.fullmatch(name):
            raise ValueError("Invalid parent name")
        parent_hashes[name] = sha256_file(Path(path)) if Path(path).is_file() else None
    result = {"schema_version": SCHEMA_VERSION, "kind": kind, "config_hash": config_hash,
              "parents": parent_hashes, "payload": dict(payload)}
    result["artifact_hash"] = canonical_hash(result)
    return result
