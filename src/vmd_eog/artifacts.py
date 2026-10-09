"""Verify immutable parent evidence before a dependent numerical stage."""
import json
from pathlib import Path
from .contracts import canonical_hash
from .io import atomic_json, sha256_file


def verify_parent(directory, required=()):
    directory = Path(directory)
    state = json.loads((directory / "execution_state.json").read_text())
    if state.get("status") != "complete":
        raise ValueError(f"Parent is incomplete: {directory}")
    manifest = json.loads((directory / "artifact_manifest.json").read_text())
    for name in required:
        candidate = (directory / name).resolve()
        if not candidate.is_relative_to(directory.resolve()):
            raise ValueError("Parent artifact path escapes its run directory")
        if name not in manifest or sha256_file(candidate) != manifest[name]:
            raise ValueError(f"Parent artifact checksum mismatch: {name}")
    return manifest


def record_parent_identities(input_root, output, config):
    identities = []
    for spec_path in sorted(Path(input_root).rglob("run_spec.json")):
        directory = spec_path.parent
        verify_parent(directory, ("run_spec.json", "configuration.json", "environment.json"))
        spec = json.loads(spec_path.read_text())
        if spec.get("campaign_id") != config["campaign_id"]:
            raise ValueError("Parent belongs to another campaign")
        previous = json.loads((directory / "configuration.json").read_text())
        if canonical_hash(previous) != canonical_hash(config):
            raise ValueError("Parent configuration differs; launch an explicit new campaign")
        identities.append({"run_id": spec["run_id"], "stage": spec["stage"],
            "git_sha": spec["git_sha"], "kernel_sources": spec.get("kernel_sources", []),
            "artifact_manifest_sha256": sha256_file(directory / "artifact_manifest.json")})
    atomic_json(Path(output) / "parent_identities.json", identities)
    return identities
