"""Download and verify a completed Kaggle AutoVMD cache locally.

Kaggle's stock output command downloads every cache array serially. This helper
uses the same signed output URLs with bounded concurrency, then verifies each
file against the SHA-256 recorded by the Kaggle-produced cache index.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time

import requests


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".kaggle-work"


def safe_relative_path(file_name, campaign_id, run_id):
    remote = PurePosixPath(file_name)
    if remote.is_absolute() or ".." in remote.parts:
        raise ValueError("Unsafe Kaggle output path")
    parts = remote.parts
    prefix = ("results", campaign_id, run_id)
    for index in range(len(parts) - 2):
        if tuple(parts[index:index + 3]) == prefix:
            parts = parts[index + 3:]
            break
    if not parts:
        raise ValueError("Kaggle output path has no artifact name")
    return Path(*parts)


def list_output_files(kernel, page_size=200):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    api = KaggleApi()
    api.authenticate()
    owner, slug = kernel.split("/", 1)
    found = []
    token = None
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            request.page_size = page_size
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            found.extend(response.files or [])
            token = response.next_page_token
            if not token:
                break
    return found


def download_one(item, destination, attempts=4):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    for attempt in range(attempts):
        try:
            with requests.get(item.url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            temporary.replace(destination)
            return destination.stat().st_size
        except Exception as error:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                # Do not print signed URLs or request exception text.
                raise RuntimeError(f"Download failed for {item.file_name}: {type(error).__name__}") from None
            time.sleep(2 ** attempt)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_cache(artifact_root, run_spec):
    summary_path = artifact_root / "autovmd_cache_summary.json"
    index_path = artifact_root / "mode_cache_index.jsonl"
    manifest_path = artifact_root / "artifact_manifest.json"
    for required in (summary_path, index_path, manifest_path):
        if not required.is_file():
            raise RuntimeError(f"Missing required downloaded artifact: {required.name}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("reserved_confirmation_opened") is not False:
        raise RuntimeError("Cache metadata indicates confirmation data was opened")
    expected = int(summary["entries"])
    records = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != expected:
        raise RuntimeError(f"Index has {len(records)} entries; cache summary declares {expected}")

    for row in records:
        relative = Path(*PurePosixPath(row["path"]).parts)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unsafe path in Kaggle cache index")
        path = artifact_root / relative
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"Cache hash mismatch or missing file: {row['path']}")

    actual = list((artifact_root / "mode_cache").rglob("*.npz"))
    if len(actual) != expected:
        raise RuntimeError(f"Found {len(actual)} NPZ files; cache summary declares {expected}")
    return {"entries": expected, "npz_files": len(actual), "settings": summary["settings"],
            "scope": summary["scope"], "configuration_hash": summary["configuration_hash"],
            "run_id": run_spec["run_id"], "git_sha": run_spec["git_sha"],
            "kernel": run_spec["kernel"], "kernel_version": run_spec["kernel_version"],
            "sha256_verified": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")

    spec_path = WORK / args.run_id / "run_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("stage") != "autovmd-cache" or not spec.get("kernel_version"):
        raise RuntimeError("Run is not a version-pinned AutoVMD cache")
    os.environ["KAGGLE_CONFIG_DIR"] = str(ROOT / ".kaggle-config")
    cli = shutil.which("kaggle") or "kaggle"
    status = subprocess.check_output([cli, "kernels", "status", spec["kernel"]],
                                     text=True, encoding="utf-8", env=os.environ.copy())
    if "COMPLETE" not in status:
        raise RuntimeError("Kaggle cache kernel is not complete; no download started")

    campaign = spec["campaign_id"]
    run_id = spec["run_id"]
    artifact_root = ROOT / "results" / campaign / run_id / "kaggle_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    files = list_output_files(spec["kernel"])
    if not files:
        raise RuntimeError("Kaggle returned no output files")

    tasks = []
    for item in files:
        relative = safe_relative_path(item.file_name, campaign, run_id)
        tasks.append((item, artifact_root / relative))
    print(f"Downloading {len(tasks)} output files with {args.workers} workers", flush=True)
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, item, path) for item, path in tasks]
        for number, future in enumerate(as_completed(futures), 1):
            total_bytes += future.result()
            if number % 200 == 0 or number == len(futures):
                print(f"Downloaded {number}/{len(futures)} files", flush=True)

    report = verify_cache(artifact_root, spec)
    report["downloaded_bytes"] = total_bytes
    report_path = ROOT / "results" / campaign / run_id / "local_cache_verification.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Verified local cache: {report_path}", flush=True)


if __name__ == "__main__":
    main()
