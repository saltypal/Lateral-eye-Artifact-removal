"""Non-numerical 45-second Kaggle watcher with verified output retrieval."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from kaggle_campaign import ROOT, WORK


def command(*arguments):
    return subprocess.check_output([sys.executable, str(ROOT / "tools/kaggle_campaign.py"),
                                    *arguments], cwd=ROOT, text=True, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    requested = json.loads((WORK / args.run_id / "run_spec.json").read_text())
    destination = ROOT / "results" / requested["campaign_id"] / args.run_id
    destination.mkdir(parents=True, exist_ok=True)
    while True:
        response = command("status", "--run-id", args.run_id)
        message = {"utc": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
                   "status": response.strip()}
        print(json.dumps(message), flush=True)
        with (destination / "watch_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message) + "\n")
        if "KernelWorkerStatus.COMPLETE" in response or "KernelWorkerStatus.ERROR" in response:
            print(command("retrieve", "--run-id", args.run_id,
                "--file-pattern", "guard17_source_audit|execution_state|artifact_manifest|contract_tests|environment|configuration"), flush=True)
            identities = list(destination.rglob("run_spec.json"))
            if len(identities) != 1:
                raise RuntimeError("Missing or ambiguous retrieved run identity")
            parent = identities[0].parent
            actual = json.loads(identities[0].read_text())
            for field in ("run_id", "git_sha", "orchestrator_git_sha", "stage", "campaign_id"):
                if actual.get(field) != requested.get(field):
                    raise RuntimeError("Retrieved source identity differs: " + field)
            manifest_path = parent / "artifact_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                for path in parent.rglob("*"):
                    if not path.is_file() or path.name in ("execution_state.json", "artifact_manifest.json"):
                        continue
                    relative = path.relative_to(parent).as_posix()
                    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get(relative):
                        raise RuntimeError("Retrieved file checksum differs: " + relative)
            print("PRIMARY_REVIEW_REQUIRED; no algorithm changes or dependent launch", flush=True)
            return
        if not any("KernelWorkerStatus." + state in response for state in ("RUNNING", "QUEUED")):
            raise RuntimeError("Unknown/API status; do not infer numerical failure")
        time.sleep(45)


if __name__ == "__main__":
    main()
