"""Local-only orchestration: package/hash files, invoke Kaggle, retrieve outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
WORK = REPOSITORY / ".kaggle-work"
OWNER = "satyapaladugu"
DATASET = f"{OWNER}/lateral-eye-complete-dataset"


def run_cli(arguments: list[str]) -> None:
    # A stale legacy config previously shadowed valid OAuth credentials. OAuth
    # remains in ~/.kaggle/credentials.json; this empty per-task config is safe.
    config = REPOSITORY / ".kaggle-config"
    config.mkdir(exist_ok=True)
    process_environment = os.environ.copy()
    process_environment["KAGGLE_CONFIG_DIR"] = str(config)
    subprocess.run([shutil.which("kaggle") or "kaggle", *arguments], env=process_environment, check=True)


def prepare(source: Path) -> None:
    source = source.resolve(strict=True)
    staging = WORK / "dataset"
    staging.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if path.stat().st_size == 0:
            raise ValueError(f"Empty source file: {path}")
        relative = path.relative_to(source)
        destination = staging / "complete_dataset" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(path, destination)
        if not os.path.samefile(path, destination):
            raise ValueError(f"Staging file is stale: {destination}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        entries.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    manifest = {"source": str(source), "files": entries}
    (staging / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metadata = {"id": DATASET, "title": "Lateral Eye Complete Dataset", "isPrivate": True,
                "licenses": [{"name": "other"}],
                "description": "Private research copy of the complete local bundle. Individual upstream terms apply; no public relicensing. Includes Klados, OSF originals, derivatives and EyeTrack. SHA256 manifest included."}
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(entries), "bytes": sum(item["bytes"] for item in entries), "dataset": DATASET}))


def submit(phase: str, sha: str) -> None:
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise ValueError("Use an explicit 40-character Git SHA")
    from build_research_notebook import build
    stage = WORK / phase
    stage.mkdir(parents=True, exist_ok=True)
    build(stage / "Region_Aware_EOG_Kaggle.ipynb", phase=phase, sha=sha)
    metadata = {"id": f"{OWNER}/region-aware-eog-{phase}", "title": f"Region Aware EOG {phase.title()}",
                "code_file": "Region_Aware_EOG_Kaggle.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": phase == "train", "enable_internet": True,
                "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": []}
    if phase == "train":
        metadata["machine_shape"] = "NvidiaTeslaT4"
    (stage / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    run_cli(["kernels", "push", "-p", str(stage)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["auth-check", "prepare", "upload", "dataset-status", "submit", "status", "retrieve"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--phase", choices=["audit", "benchmark", "train"], default="audit")
    parser.add_argument("--sha")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        prepare(args.source)
    elif args.action == "auth-check":
        run_cli(["kernels", "list", "--mine", "--page-size", "1"])
    elif args.action == "upload":
        run_cli(["datasets", "create", "-p", str(WORK / "dataset"), "--dir-mode", "zip", "--keep-tabular", "--quiet"])
    elif args.action == "dataset-status":
        run_cli(["datasets", "status", DATASET])
        run_cli(["datasets", "files", DATASET])
    elif args.action == "submit":
        submit(args.phase, args.sha or "")
    elif args.action == "status":
        run_cli(["kernels", "status", f"{OWNER}/region-aware-eog-{args.phase}"])
    elif args.action == "retrieve":
        run_cli(["kernels", "output", f"{OWNER}/region-aware-eog-{args.phase}", "-p", str(REPOSITORY / "results" / args.phase), "--force"])


if __name__ == "__main__":
    main()
