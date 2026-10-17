"""Monitor the regional pilot, then launch the same full comparison if verified.

No numerical imports/evaluation. Stops on failed contracts, parity or API status.
Does not launch neural work or native/confirmation evaluation.
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle_campaign import ROOT, WORK
from run_watch import command

PATTERN = "regional_|posterior_oof_review|posterior_metric_parity|inherited_|execution_state|artifact_manifest|stage_tests|tests.txt|environment|configuration"


def monitor(run_id):
    spec = json.loads((WORK / run_id / "run_spec.json").read_text())
    destination = ROOT / "results" / spec["campaign_id"] / run_id
    destination.mkdir(parents=True, exist_ok=True)
    while True:
        status = command("status", "--run-id", run_id)
        event = {"utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id, "status": status.strip()}
        print(json.dumps(event), flush=True)
        with (destination / "chain_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        if "KernelWorkerStatus.COMPLETE" in status or "KernelWorkerStatus.ERROR" in status:
            # Existing watcher validates run identity and downloaded payload hashes.
            subprocess.run([sys.executable, str(ROOT / "tools/run_watch.py"), "--run-id", run_id,
                            "--file-pattern", PATTERN], cwd=ROOT, check=True)
            summaries = list(destination.rglob("regional_summary.json"))
            if len(summaries) != 1:
                raise RuntimeError("No unique completed regional summary; diagnose before another launch")
            parent = summaries[0].parent
            state = json.loads((parent / "execution_state.json").read_text())
            tests = json.loads((parent / "stage_tests_summary.json").read_text())
            summary = json.loads(summaries[0].read_text())
            if state.get("status") != "complete" or tests.get("passed") is not True:
                raise RuntimeError("Regional execution/contracts failed")
            if summary.get("posterior_parity_checks", 0) <= 0 or summary.get("autovmd_executed") is not False:
                raise RuntimeError("Regional prediction/source contract differs")
            return summary
        if not any("KernelWorkerStatus." + value in status for value in ("RUNNING", "QUEUED")):
            raise RuntimeError("API/unknown status; no inference of kernel failure or dependent launch")
        time.sleep(45)


def main():
    monitor("regional-pilot-001")
    run_id = "regional-full-001"
    if not (WORK / run_id).exists():
        args = ["submit", "--stage", "regional-integration", "--run-id", run_id,
                "--profile", "full", "--config", "configs/regional_integration.json",
                "--kernel-slug", "vmd-regional-composition-full", "--parent-run", "regional-pilot-001"]
        for source in ("satyapaladugu/vmd-eog-corpus/2", "satyapaladugu/vmd-eog-research-ready/1",
                       "satyapaladugu/vmd-frontal-monotonic-axis-events/1", "satyapaladugu/vmd-band-posterior-full/1"):
            args += ["--kernel-source", source]
        print(command(*args), flush=True)
    monitor(run_id)
    print("REGIONAL_COMPLETE_PRIMARY_REVIEW_REQUIRED; no model/confirmation authorization", flush=True)


if __name__ == "__main__":
    main()
