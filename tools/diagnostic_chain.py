"""Monitor and verify the diagnostic pilot before submitting the full run.

Only orchestration/metadata/checksum work runs locally. API errors are retained
as monitoring errors, not relabelled as numerical failures.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from kaggle_campaign import ROOT, WORK

PATTERN = "diagnostic_|execution_state|artifact_manifest|stage_tests|tests.txt|environment|configuration|resource_summary|parent_identities"


def manifest_with_retry(parent, destination, run_id):
    """Kaggle downloads can return an HTML 502 body with a successful CLI exit."""
    path = parent / "artifact_manifest.json"
    for attempt in range(4):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not value:
                raise ValueError("Missing artifact hashes")
            return value
        except (json.JSONDecodeError, ValueError, OSError) as error:
            evidence = destination / "retrieval_failures"
            evidence.mkdir(exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            if path.exists():
                (evidence / ("manifest-invalid-" + timestamp + ".txt")).write_bytes(path.read_bytes())
            print("MANIFEST_RETRIEVAL_RETRY", run_id, attempt + 1, str(error), flush=True)
            if attempt == 3:
                raise RuntimeError("Manifest retrieval failed repeatedly; kernel outcome not relabelled") from error
            retry = call("retrieve", "--run-id", run_id, "--file-pattern", "artifact_manifest.json$")
            if retry.returncode:
                print(retry.stderr[-2000:], flush=True)
            time.sleep(5)


def call(*arguments):
    return subprocess.run([sys.executable, str(ROOT / "tools/kaggle_campaign.py"), *arguments],
                          cwd=ROOT, text=True, encoding="utf-8", capture_output=True)


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
            raise RuntimeError("Five consecutive API errors; kernel outcome unconfirmed, no dependent launch")
        if result.returncode:
            time.sleep(45)
            continue
        if "KernelWorkerStatus.COMPLETE" in result.stdout or "KernelWorkerStatus.ERROR" in result.stdout:
            retrieved = call("retrieve", "--run-id", run_id, "--file-pattern", PATTERN)
            print(retrieved.stdout, retrieved.stderr, flush=True)
            if retrieved.returncode:
                raise RuntimeError("Output retrieval failed; no dependent launch")
            files = list(destination.rglob("run_spec.json"))
            if len(files) != 1:
                raise RuntimeError("Missing/ambiguous output identity")
            parent = files[0].parent
            actual = json.loads(files[0].read_text())
            for key in ("run_id", "git_sha", "stage", "campaign_id"):
                if actual.get(key) != spec.get(key):
                    raise RuntimeError("Retrieved identity differs: " + key)
            state = json.loads((parent / "execution_state.json").read_text())
            manifest = manifest_with_retry(parent, destination, run_id)
            verified = []
            for path in parent.rglob("*"):
                if not path.is_file() or path.name in ("execution_state.json", "artifact_manifest.json"):
                    continue
                name = path.relative_to(parent).as_posix()
                if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get(name):
                    raise RuntimeError("Downloaded checksum differs: " + name)
                verified.append(name)
            if state.get("status") != "complete":
                raise RuntimeError("Diagnostic kernel failed; retained outputs require diagnosis")
            tests = json.loads((parent / "stage_tests_summary.json").read_text())
            summary = json.loads((parent / "diagnostic_summary.json").read_text())
            if tests.get("passed") is not True or summary.get("channel_metric_parity_checks", 0) <= 0 or summary.get("whole_pooled_parity_checks", 0) <= 0:
                raise RuntimeError("Diagnostic tests/parity did not pass")
            for key in ("autovmd_executed", "weights_trained", "correction_changed", "reserved_confirmation_opened"):
                if summary.get(key) is not False:
                    raise RuntimeError("Diagnostic scope differs: " + key)
            receipt = {"run_id": run_id, "git_sha": spec["git_sha"], "verified_payloads": verified,
                       "utc": datetime.now(timezone.utc).isoformat(), "completion_verified": True}
            (destination / "retrieval_receipt.json").write_text(json.dumps(receipt, indent=2))
            return spec, summary
        if not any("KernelWorkerStatus." + state in result.stdout for state in ("RUNNING", "QUEUED")):
            raise RuntimeError("Unknown API status; kernel failure not inferred")
        time.sleep(45)


def main():
    pilot, summary = monitor("regional-diagnostic-pilot-001")
    if summary["profile"] != "pilot":
        raise RuntimeError("Expected pilot profile")
    run_id = "regional-diagnostic-full-001"
    if not (WORK / run_id).exists():
        arguments = ("submit", "--stage", "regional-diagnostic", "--run-id", run_id,
                     "--profile", "full", "--config", "configs/regional_diagnostic.json",
                     "--kernel-slug", "vmd-regional-error-full", "--parent-run", pilot["run_id"],
                     "--source-sha", pilot["git_sha"],
                     "--kernel-source", "satyapaladugu/vmd-regional-composition-full/1")
        submitted = call(*arguments)
        print(submitted.stdout, submitted.stderr, flush=True)
        if submitted.returncode:
            raise RuntimeError("Full diagnostic submission failed")
    monitor(run_id)
    print("DIAGNOSTIC_COMPLETE; primary review required; no challenger/model automatically launched", flush=True)


if __name__ == "__main__":
    main()
