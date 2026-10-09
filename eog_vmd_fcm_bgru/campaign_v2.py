"""Versioned Kaggle campaign dispatcher with verified, immutable parents."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback
import zipfile

from .campaign_contracts import RunSpec, atomic_json, canonical_hash, sha256_file


DATA_STAGES = {"provenance", "split-freeze", "corpus-fit", "corpus-build"}
TEACHER_STAGES = {"teacher-cache", "teacher-search", "teacher-oof"}


def resolve_parents(inputs: dict, attachment_root: Path, campaign_id=None) -> dict[str, Path]:
    """Resolve exact run identities, never infer a parent from a kernel directory name."""
    manifests = list(Path(attachment_root).rglob("run_manifest.json"))
    result = {}
    for alias, expected in inputs.get("parents", {}).items():
        matches = []
        for path in manifests:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("run_id") == expected["run_id"]:
                matches.append((path, value))
        if len(matches) != 1:
            raise ValueError(f"Parent {alias}: expected one exact run {expected['run_id']}, found {len(matches)}")
        path, value = matches[0]
        if value.get("state") != "completed":
            raise ValueError(f"Parent {alias} did not complete")
        if campaign_id is not None and value.get("campaign_id") != campaign_id:
            raise ValueError(f"Parent {alias} belongs to another campaign")
        if expected.get("stage", alias) != value.get("stage"):
            raise ValueError(f"Parent {alias} stage mismatch")
        digest = expected.get("manifest_sha256")
        if not digest or sha256_file(path) != digest:
            raise ValueError(f"Parent {alias} manifest hash mismatch")
        for item in value.get("files", []):
            artifact = (path.parent / item["path"]).resolve()
            if not artifact.is_relative_to(path.parent.resolve()) or not artifact.is_file():
                raise ValueError(f"Parent {alias} has an invalid artifact path")
            if artifact.stat().st_size != item["bytes"] or sha256_file(artifact) != item["sha256"]:
                raise ValueError(f"Parent {alias} artifact content mismatch: {item['path']}")
        result[alias] = path.parent
    return result


def unpack_source(attachment_root: Path, extraction: Path, output: Path) -> tuple[Path, Path]:
    """Verify container and every declared file before using the original source."""
    manifests = list(Path(attachment_root).rglob("source_manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Attach exactly one complete source manifest")
    manifest = manifests[0]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    container = payload.get("archive")
    if container:
        archive = manifest.parent / container["name"]
        if archive.stat().st_size != container["bytes"] or sha256_file(archive) != container["sha256"]:
            raise ValueError("Source container checksum mismatch")
        extraction.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(archive) as stream:
            names = stream.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate source archive paths")
            for member in stream.infolist():
                target = (extraction / member.filename).resolve()
                if not target.is_relative_to(extraction.resolve()):
                    raise ValueError("Unsafe source archive path")
            stream.extractall(extraction)
        candidates = list(extraction.rglob("klados_contaminated_eeg.npy"))
    else:
        candidates = list(manifest.parent.rglob("klados_contaminated_eeg.npy"))
    if len(candidates) != 1:
        raise ValueError("Expected one original Klados source tree")
    root = candidates[0].parent.parent
    for item in payload["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing/unsafe source: {item['path']}")
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Source checksum mismatch: {item['path']}")
    supplements = list(Path(attachment_root).rglob("supplement_manifest.json"))
    if len(supplements) > 1:
        raise ValueError("Ambiguous restored-source supplement")
    if supplements:
        from .bundle_io import merge_supplement
        merge_supplement(root, supplements[0], output)
    atomic_json(output / "source_verification.json", {
        "source_manifest_sha256": sha256_file(manifest), "files_verified": len(payload["files"]),
        "supplement_verified": bool(supplements), "source_root": str(root)})
    return root, manifest


def environment_record(repository: Path) -> dict:
    import torch
    versions = {}
    for package in ("numpy", "scipy", "torch", "mne", "scikit-learn", "vmdpy", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "git_sha": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "tpu_detected": bool(os.environ.get("TPU_NAME") or os.environ.get("COLAB_TPU_ADDR")),
            "backend": "torch-cuda" if torch.cuda.is_available() else "torch-cpu",
            "tpu_training_supported": False}


def execute(spec_path: Path, output: Path, attachment_root: Path) -> dict:
    if not Path("/kaggle/input").exists() and os.environ.get("EOG_ALLOW_NON_KAGGLE") != "1":
        raise RuntimeError("Numerical campaign execution is authorized on Kaggle only")
    spec = RunSpec.load(spec_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_manifest.json").exists():
        raise FileExistsError("Run IDs are immutable; create a new run ID")
    repository = Path(__file__).resolve().parents[1]
    environment = environment_record(repository)
    if environment["git_sha"] != spec.git_sha:
        raise ValueError("Checkout Git SHA differs from RunSpec")
    spec.write(output / "run_spec.json")
    atomic_json(output / "environment.json", environment)
    record = {"schema_version": 2, "campaign_id": spec.campaign_id, "run_id": spec.run_id,
              "stage": spec.stage, "git_sha": spec.git_sha, "config_hash": spec.resolved_config_hash,
              "state": "running", "files": []}
    atomic_json(output / "run_manifest.json", record)
    try:
        parents = resolve_parents(spec.inputs, attachment_root, spec.campaign_id)
        config = {**spec.config, "seed": spec.seed, "fold": spec.fold, "shard": spec.shard,
                  "run_id": spec.run_id, "git_sha": spec.git_sha}
        if spec.stage == "contracts":
            test = subprocess.run([sys.executable, "-m", "pytest", str(repository / "tests"), "-q"],
                                  cwd=repository, capture_output=True, text=True)
            (output / "contract_tests.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
            print(test.stdout + test.stderr, flush=True)
            if test.returncode:
                raise RuntimeError("Numerical contract tests failed on Kaggle")
            summary = {"passed": True, "kind": "contract_validation", "cleaning_quality_claim": False}
        else:
            if spec.inputs.get("datasets"):
                data_root, source_manifest = unpack_source(attachment_root, Path(tempfile.mkdtemp(prefix="eog-source-")) / "data", output)
                config.update(data_root=str(data_root), source_manifest_path=str(source_manifest))
            else:
                data_root = Path(attachment_root)
            if spec.stage in DATA_STAGES:
                from .data_campaign import run_data_stage
                summary = run_data_stage(spec.stage, data_root, output, config, parents)
            elif spec.stage in TEACHER_STAGES:
                from .teacher_search_v2 import run_teacher_stage
                summary = run_teacher_stage(spec.stage, data_root, output, config, parents)
            else:
                from .regional_training import run_student_stage
                summary = run_student_stage(spec.stage, data_root, output, config, parents)
        atomic_json(output / "stage_summary.json", summary)
        record["state"] = "completed"
    except BaseException as error:
        record.update(state="failed", error_type=type(error).__name__, error=str(error))
        (output / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        record["files"] = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                            "sha256": sha256_file(path)} for path in sorted(output.rglob("*"))
                           if path.is_file() and path.name != "run_manifest.json"]
        record["artifact_hash"] = canonical_hash({key: value for key, value in record.items() if key != "artifact_hash"})
        atomic_json(output / "run_manifest.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attachment-root", type=Path, default=Path("/kaggle/input"))
    args = parser.parse_args()
    print(json.dumps(execute(args.run_spec, args.output, args.attachment_root), indent=2), flush=True)


if __name__ == "__main__":
    main()
