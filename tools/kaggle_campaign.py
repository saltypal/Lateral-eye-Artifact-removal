"""Local-only orchestration: package/hash files, invoke Kaggle, retrieve outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
WORK = REPOSITORY / ".kaggle-work"
OWNER = "satyapaladugu"
DATASET = f"{OWNER}/lateral-eye-complete-dataset"


def refresh_oauth_if_due() -> None:
    """Avoid the installed SDK's 30-minute delay after access-token expiry."""
    credentials_path = Path.home() / ".kaggle" / "credentials.json"
    if not credentials_path.exists():
        return
    metadata = json.loads(credentials_path.read_text())
    expiry = metadata.get("access_token_expiration")
    if not expiry or datetime.fromisoformat(expiry) > datetime.now(timezone.utc) + timedelta(minutes=1):
        return
    # The official SDK refreshes/saves credentials; token values never enter
    # project logs, artifacts, shell arguments or Git.
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kaggle_creds import KaggleCredentials
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        credentials = KaggleCredentials.load(client)
        if credentials is None:
            raise RuntimeError("Kaggle OAuth credentials are unavailable")
        credentials.refresh_access_token()
    print("Refreshed Kaggle OAuth access; credentials were not printed")


def run_cli(arguments: list[str], capture=False) -> str:
    # A stale legacy config previously shadowed valid OAuth credentials. OAuth
    # remains in ~/.kaggle/credentials.json; this empty per-task config is safe.
    config = REPOSITORY / ".kaggle-config"
    config.mkdir(exist_ok=True)
    process_environment = os.environ.copy()
    process_environment["KAGGLE_CONFIG_DIR"] = str(config)
    process_environment["PYTHONUTF8"] = "1"
    os.environ["KAGGLE_CONFIG_DIR"] = str(config)
    refresh_oauth_if_due()
    result = subprocess.run([shutil.which("kaggle") or "kaggle", *arguments], env=process_environment,
                            check=True, capture_output=capture, text=True, encoding="utf-8")
    return result.stdout if capture else ""


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


def package_opaque_bundle() -> None:
    """Preserve nested archives and hidden source files without Kaggle parsing.

    A ZIP container uses .eogbundle so Kaggle stores it as an opaque research
    file. The notebook explicitly verifies and extracts the ZIP on Kaggle.
    """
    stage = WORK / "opaque"
    stage.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((WORK / "dataset" / "source_manifest.json").read_text())
    bundle = stage / "complete_dataset.eogbundle"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for item in manifest["files"]:
            path = WORK / "dataset" / "complete_dataset" / item["path"]
            archive.write(path, arcname="complete_dataset/" + item["path"])
    digest = hashlib.sha256()
    with bundle.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    manifest["archive"] = {"name": bundle.name, "bytes": bundle.stat().st_size, "sha256": digest.hexdigest(),
                           "format": "zip; opaque extension prevents backend recursive expansion"}
    (stage / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copyfile(WORK / "dataset" / "dataset-metadata.json", stage / "dataset-metadata.json")
    print(json.dumps({"packed_source_files": len(manifest["files"]), **manifest["archive"]}))


def recover_uploaded_archive(cache_path: Path) -> None:
    """Finish a version after the Windows CLI fails on a manifest cache path.

    The completed archive's upload token stays in memory; never print or save
    tokens, signed upload URLs or authentication details into project outputs.
    """
    import importlib
    os.environ["KAGGLE_CONFIG_DIR"] = str(REPOSITORY / ".kaggle-config")
    refresh_oauth_if_due()
    sdk = importlib.import_module("kaggle.api.kaggle_api_extended")
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    if not cached.get("upload_complete") or time.time() - cached["timestamp"] > 24 * 3600:
        raise ValueError("Archive upload is incomplete or stale")
    request_metadata = cached["start_blob_upload_request"]
    if not request_metadata["name"].endswith("complete_dataset.zip"):
        raise ValueError("Upload cache does not describe this EEG archive")
    config = REPOSITORY / ".kaggle-config"
    os.environ["KAGGLE_CONFIG_DIR"] = str(config)
    api = sdk.KaggleApi()
    api.authenticate()
    body = sdk.ApiCreateDatasetVersionRequestBody()
    body.version_notes = "Complete original 375-file EEG bundle; every file verified by attached SHA256 manifest"
    archive = sdk.ApiDatasetNewFile()
    archive.token = cached["start_blob_upload_response"]["token"]
    body.files = [archive]
    manifest = (WORK / "dataset" / "source_manifest.json").resolve(strict=True)
    with sdk.ResumableUploadContext() as context:
        uploaded = api._upload_file("source_manifest.json", str(manifest), sdk.ApiBlobType.DATASET,
                                    context, quiet=True, resources=None)
        if uploaded is None:
            raise RuntimeError("Manifest upload failed; no version created")
        body.files.append(api._new_file(uploaded))
        request = sdk.ApiCreateDatasetVersionRequest()
        request.owner_slug = OWNER
        request.dataset_slug = DATASET.split("/", 1)[1]
        request.body = body
        with api.build_kaggle_client() as client:
            response = client.datasets.dataset_api_client.create_dataset_version(request)
        public_response = {"ref": response.ref, "url": response.url, "status": response.status, "error": response.error}
        print(json.dumps(public_response))
        if response.status.lower() != "ok":
            raise RuntimeError("Kaggle rejected recovered archive version")
    (WORK / "upload_recovery_response.json").write_text(json.dumps(public_response, indent=2), encoding="utf-8")


def verify_remote_inventory() -> dict:
    """Require the current published version to contain the entire upload.

    Publication may lag behind a successful create-version response. An older
    ready manifest-only version is insufficient for an experiment attachment.
    """
    import importlib
    os.environ["KAGGLE_CONFIG_DIR"] = str(REPOSITORY / ".kaggle-config")
    refresh_oauth_if_due()
    sdk = importlib.import_module("kaggle.api.kaggle_api_extended")
    api = sdk.KaggleApi()
    api.authenticate()
    status = json.loads(api.dataset_status(DATASET, format="json"))
    if status["status"] != "ready":
        raise RuntimeError(f"Dataset is not ready: {status['status']}")
    remote, next_page = {}, None
    while True:
        page = api.dataset_list_files(DATASET, page_token=next_page, page_size=100)
        if page.error_message:
            raise RuntimeError(page.error_message)
        for item in page.files:
            remote[item.name] = item.total_bytes
        next_page = page.next_page_token
        if not next_page:
            break
    manifest = json.loads((WORK / "dataset" / "source_manifest.json").read_text())
    opaque_manifest = WORK / "opaque" / "source_manifest.json"
    if "complete_dataset.eogbundle" in remote and opaque_manifest.exists():
        archive = json.loads(opaque_manifest.read_text())["archive"]
        if remote[archive["name"]] != archive["bytes"] or "source_manifest.json" not in remote:
            raise RuntimeError("Published opaque bundle size/manifest mismatch")
        result = {**status, "packed_source_files": len(manifest["files"]), "archive_size_matches": True,
                  "remote_files": len(remote), "source_sha256_verification": "pending Kaggle audit"}
        print(json.dumps(result))
        return result
    mismatches = [item["path"] for item in manifest["files"]
                  if remote.get("complete_dataset/" + item["path"]) != item["bytes"]]
    if mismatches or "source_manifest.json" not in remote:
        raise RuntimeError(f"Current dataset v{status['current_version_number']} is incomplete: {len(mismatches)} missing/size-mismatched source files")
    result = {**status, "source_files_verified_by_size": len(manifest["files"]), "remote_files": len(remote)}
    print(json.dumps(result))
    return result


def submit(phase: str, sha: str) -> None:
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise ValueError("Use an explicit 40-character Git SHA")
    subprocess.run(["git", "-C", str(REPOSITORY), "cat-file", "-e", sha + "^{commit}"], check=True)
    # A syntactically valid SHA can still be absent from the remote checkout.
    published = subprocess.check_output(["git", "-C", str(REPOSITORY), "branch", "-r", "--contains", sha], text=True)
    if not any(line.strip().startswith("origin/") for line in published.splitlines()):
        raise ValueError("Push this commit to origin before launching Kaggle")
    # Kaggle may accept a kernel push while dropping an invalid attachment.
    # Fail before submission until the private data is actually accessible.
    if phase not in {"contracts", "report", "klados-group-audit", "vmd-engine"}:
        verify_remote_inventory()
    from build_research_notebook import build
    stage = WORK / phase
    stage.mkdir(parents=True, exist_ok=True)
    build(stage / "Region_Aware_EOG_Kaggle.ipynb", phase=phase, sha=sha)
    metadata = {"id": f"{OWNER}/region-aware-eog-{phase}", "title": f"Region Aware EOG {phase.title()}",
                "code_file": "Region_Aware_EOG_Kaggle.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": phase in {"train", "neural-search"}, "enable_internet": True,
                "dataset_sources": [] if phase in {"contracts", "report", "klados-group-audit", "vmd-engine"} else [DATASET], "competition_sources": [],
                "kernel_sources": ([f"{OWNER}/region-aware-eog-klados-source"] if phase in {"klados-group-audit", "vmd-engine"} else
                                   [f"{OWNER}/region-aware-eog-{item}" for item in ["neural-search", "calibration", "vmd-convergence"]] if phase == "report" else
                                   [f"{OWNER}/region-aware-eog-benchmark", f"{OWNER}/region-aware-eog-restore"] if phase in {"train", "calibration", "neural-search", "vmd-convergence", "vmd-robust-grid"}
                                   else [f"{OWNER}/region-aware-eog-restore"] if phase == "benchmark" else [])}
    if phase in {"train", "neural-search"}:
        metadata["machine_shape"] = "NvidiaTeslaT4"
    (stage / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    response = run_cli(["kernels", "push", "-p", str(stage)], capture=True)
    print(response)
    if "not valid dataset sources" in response or "could not be added" in response:
        raise RuntimeError("Kaggle rejected the dataset attachment; kernel execution is invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["auth-check", "prepare", "upload", "package-opaque", "upload-opaque", "recover-upload", "inventory-check", "dataset-status", "submit", "status", "retrieve"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--phase", choices=["contracts", "audit", "restore", "benchmark", "calibration", "klados-source", "klados-group-audit", "vmd-engine", "train", "neural-search", "vmd-convergence", "vmd-robust-grid", "report"], default="audit")
    parser.add_argument("--sha")
    parser.add_argument("--upload-cache", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        prepare(args.source)
    elif args.action == "auth-check":
        run_cli(["kernels", "list", "--mine", "--page-size", "1"])
    elif args.action == "upload":
        run_cli(["datasets", "create", "-p", str(WORK / "dataset"), "--dir-mode", "zip", "--keep-tabular", "--quiet"])
    elif args.action == "package-opaque":
        package_opaque_bundle()
    elif args.action == "upload-opaque":
        run_cli(["datasets", "version", "-p", str((WORK / "opaque").resolve()), "--keep-tabular",
                 "-m", "All 375 original files preserved in a checksummed opaque research bundle"])
    elif args.action == "recover-upload":
        if args.upload_cache is None:
            parser.error("recover-upload requires an explicit --upload-cache")
        recover_uploaded_archive(args.upload_cache)
    elif args.action == "inventory-check":
        verify_remote_inventory()
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
