"""Persistent non-numerical Kaggle monitoring and bounded artifact retrieval.

No method changes or dependent launches occur here. The primary agent reviews
scientific evidence after complete, identity-checked retrieval.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from kaggle_campaign import ROOT, WORK


STATES = {"RUNNING": "running", "QUEUED": "queued", "COMPLETE": "complete",
          "ERROR": "error", "CANCELED": "cancelled", "CANCELLED": "cancelled",
          "CANCEL_REQUESTED": "running"}


def classify_status(response):
    tokens = set(re.findall(r"KernelWorkerStatus\.([A-Z_]+)", response))
    return STATES.get(next(iter(tokens))) if len(tokens) == 1 else None


def save_status(destination, message):
    path = destination / "watch_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(message, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def retrieve(run_id, destination, requested, pattern):
    log = destination / "watch_retrieval.log"
    with log.open("a", encoding="utf-8") as handle:
        result = subprocess.run([sys.executable, str(ROOT / "tools/retrieve_bounded.py"),
            "--run-id", run_id, "--pattern", pattern], cwd=ROOT,
            stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Bounded retrieval failed; inspect watch_retrieval.log")
    receipt = json.loads((destination / "retrieval_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("run_id") != requested["run_id"] or receipt.get("git_sha") != requested["git_sha"]:
        raise ValueError("Retrieved run/source differs from monitored identity")
    if pattern == ".*" and receipt.get("all_manifest_payloads_verified") is not True:
        raise ValueError("Full retrieval did not verify the complete manifest")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--file-pattern", default=".*")
    parser.add_argument("--interval", type=int, choices=(30, 45, 60), default=45)
    parser.add_argument("--once", action="store_true", help="Inspect one status without starting retrieval")
    args = parser.parse_args()
    requested = json.loads((WORK / args.run_id / "run_spec.json").read_text(encoding="utf-8"))
    destination = ROOT / "results" / requested["campaign_id"] / args.run_id
    destination.mkdir(parents=True, exist_ok=True)
    last_state, api_errors = None, 0

    def report(state, **details):
        nonlocal last_state
        message = {"utc": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
                   "git_sha": requested["git_sha"], "pid": os.getpid(), "monitor_state": state,
                   "interval_s": args.interval, **details}
        save_status(destination, message)
        if state != last_state or state in ("api_attention", "retrieval_attention"):
            print(json.dumps(message), flush=True)
            with (destination / "watch_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(message) + "\n")
        last_state = state

    while True:
        try:
            result = subprocess.run([sys.executable, str(ROOT / "tools/kaggle_campaign.py"),
                "status", "--run-id", args.run_id], cwd=ROOT, text=True, encoding="utf-8",
                capture_output=True, check=True, timeout=60)
            state = classify_status(result.stdout)
            if state is None:
                raise ValueError("Unrecognized or ambiguous API status")
            api_errors = 0
        except (subprocess.SubprocessError, ValueError) as error:
            api_errors += 1
            report("api_attention", error_type=type(error).__name__, consecutive_api_errors=api_errors,
                   kernel_failure_inferred=False, primary_review_required=api_errors >= 3)
            if args.once:
                return
            time.sleep(args.interval)
            continue
        report("watching", kernel_state=state, primary_review_required=False)
        if args.once:
            return
        if state not in ("complete", "error", "cancelled"):
            time.sleep(args.interval)
            continue
        for attempt in range(1, 6):
            report("retrieving", kernel_state=state, retrieval_attempt=attempt,
                   primary_review_required=state != "complete")
            try:
                receipt = retrieve(args.run_id, destination, requested, args.file_pattern)
                report("retrieved", kernel_state=state, primary_review_required=True,
                       completion_verified=receipt["completion_verified"],
                       all_manifest_payloads_verified=receipt["all_manifest_payloads_verified"],
                       manifest_payload_count=receipt["manifest_payload_count"])
                return
            except (OSError, ValueError, RuntimeError) as error:
                report("retrieval_attention", kernel_state=state, retrieval_attempt=attempt,
                       error_type=type(error).__name__, primary_review_required=True,
                       kernel_failure_inferred=state == "error")
                if attempt == 5:
                    raise RuntimeError("Retrieval needs primary review; no dependent launch") from None
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
