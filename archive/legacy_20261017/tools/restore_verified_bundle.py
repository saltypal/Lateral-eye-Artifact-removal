"""Restore exact source bytes from a fully recovered Kaggle bundle, no EEG math."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

from prepare_broader_stage import ROOT, digest, read, recovered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-run", required=True)
    parser.add_argument("--source-run", required=True)
    args = parser.parse_args()
    bundle, _, _, _ = recovered(args.bundle_run, "bundle_summary.json")
    index = read(bundle / "bundle_index.json")
    submitted = read(ROOT / ".kaggle-work" / args.source_run / "run_spec.json")
    if any(index["source"][key] != submitted[key] for key in ("run_id", "campaign_id", "git_sha")):
        raise ValueError("Bundle source differs from requested original run")
    destination = ROOT / "results" / submitted["campaign_id"] / submitted["run_id"]
    parent = destination / "results" / submitted["campaign_id"] / submitted["run_id"]
    parent.mkdir(parents=True, exist_ok=True)
    expected = read(bundle / "source_payload_manifest.json")
    seen = set()
    for shard in index["shards"]:
        path = (bundle / shard["path"]).resolve()
        if not path.is_relative_to(bundle.resolve()) or digest(path) != shard["sha256"]:
            raise ValueError("Bundle shard path/checksum differs")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if names != shard["entries"] or len(set(names)) != len(names):
                raise ValueError("Archive entries differ from verified index")
            for name in names:
                target = (parent / name).resolve()
                if name not in expected or name in seen or not target.is_relative_to(parent.resolve()):
                    raise ValueError("Duplicate, unexpected or escaping archive entry")
                seen.add(name)
                if target.is_file() and digest(target) == expected[name]:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".download")
                value = hashlib.sha256()
                with archive.open(name) as source, temporary.open("wb") as output:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        value.update(chunk)
                        output.write(chunk)
                if value.hexdigest() != expected[name]:
                    raise ValueError("Restored source byte checksum differs: " + name)
                os.replace(temporary, target)
    if seen != set(expected) or digest(parent / "artifact_manifest.json") != index["source"]["manifest_sha256"]:
        raise ValueError("Bundle does not cover the complete original source")
    original_manifest = read(parent / "artifact_manifest.json")
    for name, checksum in original_manifest.items():
        target = (parent / name).resolve()
        if not target.is_relative_to(parent.resolve()) or digest(target) != checksum:
            raise ValueError("Final original payload verification failed: " + name)
    present = {path.relative_to(parent).as_posix() for path in parent.rglob("*")
               if path.is_file() and path.name not in ("artifact_manifest.json", "execution_state.json")
               and not path.name.endswith(".download")}
    if present != set(original_manifest):
        raise ValueError("Unexpected or missing files in restored original parent")
    spec = read(parent / "run_spec.json")
    if any(spec[key] != submitted[key] for key in ("run_id", "campaign_id", "git_sha", "stage")):
        raise ValueError("Restored original source identity differs")
    complete = read(parent / "execution_state.json").get("status") == "complete"
    if not complete:
        raise ValueError("Restored original source was not complete")
    receipt = {"run_id": submitted["run_id"], "git_sha": submitted["git_sha"],
        "verified_payloads": sorted(original_manifest), "manifest_payload_count": len(original_manifest),
        "all_manifest_payloads_verified": True, "completion_verified": True,
        "retrieval_method": "fully verified Kaggle byte bundle", "bundle_run_id": args.bundle_run,
        "utc": datetime.now(timezone.utc).isoformat()}
    (destination / "retrieval_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("RESTORED_AND_VERIFIED_ORIGINAL_PAYLOADS", len(original_manifest), flush=True)


if __name__ == "__main__":
    main()
