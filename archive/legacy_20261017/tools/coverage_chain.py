"""Watch, recover and verify bounded coverage experiments; no numerical work."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import time

from diagnostic_chain import call, manifest_with_retry
from kaggle_campaign import ROOT, WORK


PATTERN = "coverage_(summary|method_means|condition_means|controlled_gates|calibration_fits|decisions|recipient_means|channel_scores|example_scores|whole|central_shared|uncertainty|paired)|stage_tests|tests.txt|environment|configuration|resource_summary|parent_identities"


def monitor(run_id):
    spec = json.loads((WORK / run_id / "run_spec.json").read_text())
    destination = ROOT / "results" / spec["campaign_id"] / run_id
    destination.mkdir(parents=True, exist_ok=True)
    api_errors = 0
    while True:
        result = call("status", "--run-id", run_id)
        event = {"utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                 "status": result.stdout.strip(), "api_returncode": result.returncode}
        if result.returncode:
            api_errors += 1
            event["monitoring_error"] = result.stderr[-2000:]
        else:
            api_errors = 0
        with (destination / "chain_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
        if api_errors >= 5:
            raise RuntimeError("Repeated API failures; kernel outcome unconfirmed")
        if result.returncode:
            time.sleep(45)
            continue
        if any("KernelWorkerStatus." + state in result.stdout for state in ("COMPLETE", "ERROR")):
            result = subprocess.run([sys.executable, str(ROOT / "tools/retrieve_bounded.py"),
                "--run-id", run_id, "--pattern", PATTERN], cwd=ROOT, text=True,
                encoding="utf-8", capture_output=True)
            print(result.stdout, result.stderr, flush=True)
            if result.returncode:
                raise RuntimeError("Retrieval failed; no dependent launch")
            specs = list(destination.rglob("run_spec.json"))
            if len(specs) != 1:
                raise RuntimeError("Ambiguous retrieved identity")
            parent = specs[0].parent
            actual = json.loads(specs[0].read_text())
            if any(actual.get(key) != spec.get(key) for key in ("run_id", "git_sha", "stage", "campaign_id")):
                raise RuntimeError("Retrieved run identity differs")
            hashes = manifest_with_retry(parent, destination, run_id)
            verified = []
            for path in parent.rglob("*"):
                if not path.is_file() or path.name in ("artifact_manifest.json", "execution_state.json"):
                    continue
                name = path.relative_to(parent).as_posix()
                if hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(name):
                    raise RuntimeError("Retrieved checksum differs: " + name)
                verified.append(name)
            state = json.loads((parent / "execution_state.json").read_text())
            if state.get("status") != "complete":
                raise RuntimeError("Coverage kernel failed; retain evidence and diagnose")
            summary = json.loads((parent / "coverage_summary.json").read_text())
            tests = json.loads((parent / "stage_tests_summary.json").read_text())
            if tests.get("passed") is not True or summary.get("frozen_branch_parity_checks", 0) <= 0:
                raise RuntimeError("Numerical tests/frozen branch contracts failed")
            for key in ("autovmd_executed", "weights_trained", "reserved_confirmation_opened", "scoring_target_fit"):
                if summary.get(key) is not False:
                    raise RuntimeError("Unauthorized scope: " + key)
            receipt = {"run_id": run_id, "git_sha": spec["git_sha"], "verified_payloads": verified,
                       "utc": datetime.now(timezone.utc).isoformat(), "completion_verified": True}
            (destination / "retrieval_receipt.json").write_text(json.dumps(receipt, indent=2))
            return spec, summary
        if not any("KernelWorkerStatus." + state in result.stdout for state in ("RUNNING", "QUEUED")):
            raise RuntimeError("Unknown API status; do not infer numerical failure")
        time.sleep(45)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    monitor(args.run_id)
    print("COVERAGE_VERIFIED; primary review before dependent full run or qualification", flush=True)


if __name__ == "__main__":
    main()
