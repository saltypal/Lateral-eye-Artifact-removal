"""Resume four parallel Kaggle downloads; verify the exact run and all bytes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
os.environ["KAGGLE_CONFIG_DIR"] = str(ROOT/".kaggle-config")
from kaggle.api.kaggle_api_extended import KaggleApi


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(item, destination, expected_hash=None):
    path = (destination/item.file_name).resolve()
    if not path.is_relative_to(destination.resolve()):
        raise ValueError("Output path escapes result directory")
    if expected_hash and path.is_file() and digest(path) == expected_hash:
        return
    for attempt in range(3):
        try:
            with requests.get(item.url, timeout=(10, 25)) as response:
                response.raise_for_status()
                content = response.content
            if expected_hash and hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError("Downloaded bytes differ")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix+".download")
            temporary.write_bytes(content)
            temporary.replace(path)
            return
        except (requests.RequestException, ValueError) as error:
            print("DOWNLOAD_RETRY", item.file_name, attempt+1, type(error).__name__, flush=True)
            if attempt == 2:
                raise RuntimeError("Download failed: "+item.file_name) from None
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT/".kaggle-work"/args.run_id/"run_spec.json").read_text())
    destination = ROOT/"results"/spec["campaign_id"]/args.run_id
    destination.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    Request = KaggleApi.kernels_output.__globals__["ApiListKernelSessionOutputRequest"]
    owner, slug = spec["kernel"].split("/")
    files, token = [], None
    with api.build_kaggle_client() as client:
        while True:
            request = Request()
            request.user_name, request.kernel_slug = owner, slug
            api._set_paging(request, 200, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            files.extend(response.files or [])
            token = response.next_page_token
            if not token:
                break
    manifests = [item for item in files if Path(item.file_name).name == "artifact_manifest.json"]
    if len(manifests) != 1:
        raise ValueError("No unique completed result manifest")
    download(manifests[0], destination)
    manifest = json.loads((destination/manifests[0].file_name).read_text())
    if manifest["run_id"] != args.run_id or manifest["git_sha"] != spec["git_sha"]:
        raise ValueError("Available notebook version differs from requested run")
    prefix = Path(manifests[0].file_name).parent
    available = {item.file_name: item for item in files}
    jobs = []
    for entry in manifest["files"]:
        name = str(prefix/entry["path"]).replace("\\", "/")
        if name not in available:
            raise ValueError("Manifest payload absent from Kaggle listing: "+name)
        jobs.append((available[name], entry))
    jobs.sort(key=lambda pair: (Path(pair[0].file_name).suffix == ".npz", pair[0].file_name))

    def recover(pair):
        item, entry = pair
        download(item, destination, entry["sha256"])
        if (destination/item.file_name).stat().st_size != entry["bytes"]:
            raise ValueError("Payload length differs")
        return item.file_name

    with ThreadPoolExecutor(max_workers=4) as pool:
        for count, name in enumerate(pool.map(recover, jobs), 1):
            if count%10 == 0 or count == len(jobs):
                print("VERIFIED_PAYLOADS", count, "/",len(jobs), flush=True)
    receipt = {"run_id": args.run_id, "git_sha": spec["git_sha"],
        "payloads_verified": len(jobs), "all_manifest_payloads_verified": True,
        "accuracy_qualification_claimed": False}
    (destination/"retrieval_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
