"""Recover pinned Kaggle outputs with bounded HTTP waits and verified hashes.

This tool performs orchestration and byte/provenance checks only. Signed download
URLs and credentials are never printed or persisted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ["KAGGLE_CONFIG_DIR"] = str(ROOT / ".kaggle-config")

import requests
from kaggle.api.kaggle_api_extended import KaggleApi


def download(item, destination):
    path = (destination / item.file_name).resolve()
    if not path.is_relative_to(destination.resolve()):
        raise ValueError("Output path escapes destination")
    for attempt in range(3):
        try:
            response = requests.get(item.url, timeout=(15, 45))
            response.raise_for_status()
            content = response.content
            if path.suffix == ".json":
                json.loads(content)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".download")
            temporary.write_bytes(content)
            os.replace(temporary, path)
            return item.file_name
        except (requests.RequestException, json.JSONDecodeError) as error:
            # URLs may contain credentials; expose only exception type and filename.
            print("DOWNLOAD_RETRY", item.file_name, attempt + 1, type(error).__name__, flush=True)
            if attempt == 2:
                raise RuntimeError("Output download unavailable: " + item.file_name) from None
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pattern", default=".*")
    args = parser.parse_args()
    spec = json.loads((ROOT / ".kaggle-work" / args.run_id / "run_spec.json").read_text())
    destination = ROOT / "results" / spec["campaign_id"] / args.run_id
    destination.mkdir(parents=True, exist_ok=True)
    original_send = requests.sessions.Session.send

    def bounded_send(session, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (15, 45)
        return original_send(session, request, **kwargs)

    requests.sessions.Session.send = bounded_send
    api = KaggleApi()
    api.authenticate()
    Request = KaggleApi.kernels_output.__globals__["ApiListKernelSessionOutputRequest"]
    owner, slug = spec["kernel"].split("/")
    selected, token, seen = [], None, set()
    pattern = re.compile(args.pattern)
    with api.build_kaggle_client() as client:
        while True:
            request = Request()
            request.user_name, request.kernel_slug = owner, slug
            api._set_paging(request, 200, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                if pattern.search(item.file_name) or Path(item.file_name).name in ("artifact_manifest.json", "run_spec.json", "execution_state.json"):
                    if item.file_name in seen:
                        raise ValueError("Duplicate output filename")
                    seen.add(item.file_name)
                    selected.append(item)
            token = response.next_page_token
            if not token:
                break
    print("SELECTED_OUTPUTS", len(selected), flush=True)
    # Recover identity and compact review tables before large prediction bundles.
    # This changes download order only; every selected byte is still verified.
    critical = {"artifact_manifest.json", "run_spec.json", "execution_state.json",
                "tests.txt", "nonfrontal_controlled_gates.json"}
    def priority(item):
        name = Path(item.file_name).name
        if name in critical or name.endswith("_summary.json") or name.endswith("_means.csv"):
            return 0, item.file_name
        if name.endswith(".npz"):
            return 2, item.file_name
        return 1, item.file_name
    selected.sort(key=priority)
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = list(pool.map(lambda item: download(item, destination), selected))
    specs = list(destination.rglob("run_spec.json"))
    if len(specs) != 1:
        raise ValueError("Missing or ambiguous recovered identity")
    parent = specs[0].parent
    actual = json.loads(specs[0].read_text())
    for key in ("run_id", "git_sha", "stage", "campaign_id"):
        if actual.get(key) != spec.get(key):
            raise ValueError("Recovered identity mismatch: " + key)
    hashes = json.loads((parent / "artifact_manifest.json").read_text())
    verified = []
    for path in parent.rglob("*"):
        if not path.is_file():
            continue
        if path.name in ("artifact_manifest.json", "execution_state.json"):
            continue
        relative = path.relative_to(parent).as_posix()
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(relative):
            raise ValueError("Recovered checksum mismatch: " + relative)
        verified.append(relative)
    receipt = {"run_id": args.run_id, "git_sha": spec["git_sha"], "verified_payloads": verified,
               "manifest_payload_count": len(hashes),
               "all_manifest_payloads_verified": set(verified) == set(hashes),
               "downloaded_outputs": len(downloaded),
               "utc": datetime.now(timezone.utc).isoformat(), "retrieval_method": "bounded HTTP; four download workers",
               "completion_verified": json.loads((parent / "execution_state.json").read_text()).get("status") == "complete"}
    (destination / "retrieval_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("VERIFIED_OUTPUTS", len(verified), "COMPLETE", receipt["completion_verified"], flush=True)


if __name__ == "__main__":
    main()
