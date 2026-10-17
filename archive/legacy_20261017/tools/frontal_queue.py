"""Local Kaggle orchestration only: no numerical imports or local EEG execution.

This bounded queue ends at the native frontal pilot. It cannot authorize a
teacher, open confirmation sources, integrate posterior code or train models.
Run from the project root: python tools/frontal_queue.py
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "vmd-frontal-replication-20261011"
OUTPUT = ROOT / "results" / CAMPAIGN / "orchestration"
BASE = ["satyapaladugu/vmd-eog-corpus/2",
        "satyapaladugu/vmd-eog-research-ready/1",
        "satyapaladugu/vmd-frontal-review-contracts/1"]
CACHES = ["satyapaladugu/vmd-band-cache-k3/1", "satyapaladugu/vmd-band-cache-k4/1"]
FULL_RUN = "frontal-full-002"  # 001 is preserved failed evidence, never overwritten.
REPORT_FILES = (r"(?:summary|execution_state|tests|resource_summary|environment|artifact_manifest|"
                r"frontal_fold_selections|frontal_detector_folds|frontal_oof|frontal_uncertainty|"
                r"frontal_progress|frontal_search|frontal_failures|frontal_mechanism_diagnostics|"
                r"frontal_controlled|native_condition|native_frontal_preservation|\.png$)")


def record(event, **details):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **details}
    with (OUTPUT / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def command(*args):
    result = subprocess.run([sys.executable, str(ROOT / "tools/kaggle_campaign.py"), *args],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr[-6000:] + result.stdout[-6000:])
    return result.stdout


def status(run_id):
    response = command("status", "--run-id", run_id)
    # An HTTP/API exception never becomes a scientific or kernel failure.
    for state in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "RUNNING", "QUEUED"):
        if 'KernelWorkerStatus.' + state in response:
            record("kernel_status", run_id=run_id, state=state)
            return state
    raise RuntimeError("Unrecognized kernel status: " + response[-1000:])


def retrieved(run_id, summary_name):
    directory = ROOT / "results" / CAMPAIGN / run_id
    summaries = list(directory.rglob(summary_name))
    if len(summaries) != 1:
        return None
    parent = summaries[0].parent
    manifest = json.loads((parent / "artifact_manifest.json").read_text())
    for path in parent.rglob("*"):
        if not path.is_file() or path.name in ("artifact_manifest.json", "execution_state.json"):
            continue
        relative = path.relative_to(parent).as_posix()
        if relative not in manifest:
            raise RuntimeError("Retrieved file is missing its manifest identity: " + relative)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest[relative]:
            raise RuntimeError("Retrieved checksum differs: " + relative)
    requested = json.loads((ROOT / ".kaggle-work" / run_id / "run_spec.json").read_text())
    actual = json.loads((parent / "run_spec.json").read_text())
    if any(actual.get(key) != requested.get(key) for key in ("run_id", "git_sha", "stage", "campaign_id")):
        raise RuntimeError("Retrieved run identity differs: " + run_id)
    state = json.loads((parent / "execution_state.json").read_text())
    tests = json.loads((parent / "stage_tests_summary.json").read_text())
    if state.get("status") != "complete" or tests.get("passed") is not True:
        raise RuntimeError("Retrieved stage or numerical tests failed: " + run_id)
    summary = json.loads(summaries[0].read_text())
    if summary.get("reserved_confirmation_opened") is not False:
        raise RuntimeError("Reserved-source closure is not verified: " + run_id)
    if summary.get("scope") != "verified_frontal_only" or summary.get("autovmd_executed", False):
        raise RuntimeError("Frontal scope or stopped AutoVMD policy differs: " + run_id)
    return summary


def kernel_parent(run_id):
    spec = json.loads((ROOT / ".kaggle-work" / run_id / "run_spec.json").read_text())
    if not spec.get("kernel_version"):
        raise RuntimeError("Exact parent kernel version is unavailable: " + run_id)
    return spec["kernel"] + "/" + str(spec["kernel_version"])


def collect(run_id, summary_name):
    summary = retrieved(run_id, summary_name)
    if summary is not None:
        return summary
    print(command("retrieve", "--run-id", run_id, "--file-pattern", REPORT_FILES), flush=True)
    summary = retrieved(run_id, summary_name)
    if summary is None:
        raise RuntimeError("Completed stage lacks its required summary: " + run_id)
    record("retrieved_and_identity_verified", run_id=run_id, summary=summary_name)
    return summary


def submit(run_id, stage, slug, parents, profile="pilot", parent_run=None):
    if (ROOT / ".kaggle-work" / run_id).exists():
        return
    args = ["submit", "--config", "configs/frontal_replication.json", "--stage", stage,
            "--run-id", run_id, "--profile", profile, "--kernel-slug", slug]
    if parent_run:
        args.extend(["--parent-run", parent_run])
    for parent in parents:
        args.extend(["--kernel-source", parent])
    print(command(*args), flush=True)
    record("dependent_notebook_submitted", run_id=run_id, stage=stage, parents=parents)


def step(full_run=FULL_RUN, skip_superseded_pilot=False):
    repair = None
    repair_id = "frontal-cache-audit-002"
    if (ROOT / ".kaggle-work" / repair_id).exists():
        repair = retrieved(repair_id, "frontal_cache_audit_summary.json")
        if repair is None:
            state = status(repair_id)
            if state in ("ERROR", "CANCEL_ACKNOWLEDGED"):
                raise RuntimeError("Repair audit stopped: " + state)
            if state != "COMPLETE":
                record("waiting_for_repair_audit", run_id=repair_id)
                return False
            print(command("retrieve", "--run-id", repair_id,
                          "--file-pattern", REPORT_FILES + "|frontal_cache_audit"), flush=True)
            repair = collect(repair_id, "frontal_cache_audit_summary.json")
        if (repair.get("archived_reader_passed") is not True or
                repair.get("float64_reconstruction_failures") != 0 or
                repair.get("examples") != 320 or repair.get("channel_recipe_checks") != 8320):
            raise RuntimeError("Repair audit lacks full unchanged-tolerance evidence")
    active = []
    for run_id, summary_name in (
            ("frontal-pilot-002", "frontal_summary.json"),
            ("frontal-screen-001", "frontal_summary.json"),
            ("frontal-evidence-001", "frontal_evidence_summary.json"),
            (full_run, "frontal_summary.json"),
            ("frontal-native-pilot-001", "native_frontal_summary.json")):
        if not (ROOT / ".kaggle-work" / run_id).exists():
            continue
        if run_id == "frontal-pilot-002" and skip_superseded_pilot:
            # The completed full-bank screen and mechanism report are the
            # actual parents. Keep the optional older pilot as historical work.
            continue
        summary = retrieved(run_id, summary_name)
        if summary is None:
            state = status(run_id)
            if state in ("ERROR", "CANCEL_ACKNOWLEDGED"):
                raise RuntimeError("Dependent stages stopped: " + run_id + " is " + state)
            if state != "COMPLETE":
                active.append(run_id)
                continue
            summary = collect(run_id, summary_name)
        if run_id == "frontal-pilot-002":
            submit("frontal-evidence-001", "frontal-evidence", "vmd-frontal-result-inspection",
                   BASE + CACHES + [kernel_parent(run_id)], parent_run=run_id)
        elif run_id == "frontal-screen-001":
            if summary.get("evaluation_complete") is not False or summary.get("selection_subset") != "source-balanced pilot recipients":
                raise RuntimeError("Screen interface differs; stop before full evaluation")
            if repair is None:
                raise RuntimeError("Repaired full evaluation requires completed cache audit")
            submit(full_run, "frontal-evaluation", "vmd-frontal-full-evaluation",
                   BASE + CACHES + [kernel_parent(run_id), kernel_parent(repair_id)], "full", run_id)
            # A completed screen supports a stronger frozen-recipe mechanism
            # audit without waiting for a separately delayed smaller pilot.
            submit("frontal-evidence-001", "frontal-evidence", "vmd-frontal-result-inspection",
                   BASE + CACHES + [kernel_parent(run_id)], parent_run=run_id)
        elif run_id == full_run:
            if summary.get("evaluation_complete") is not True:
                raise RuntimeError("Full evaluation is incomplete; native validation blocked")
            submit("frontal-native-pilot-001", "frontal-native", "vmd-frontal-native-pilot",
                   BASE + [kernel_parent(run_id),
                           "satyapaladugu/region-aware-eog-restore/1"], parent_run=run_id)
        elif run_id == "frontal-native-pilot-001":
            if retrieved("frontal-evidence-001", "frontal_evidence_summary.json") is not None:
                record("frontal_review_checkpoint", run_id=run_id,
                       note="Primary-agent review required; no posterior/neural/full-native auto-launch",
                       native_failure_count=summary.get("failure_count"))
                return True
    record("waiting", active_runs=active)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="One orchestration poll, including eligible submissions")
    parser.add_argument("--watch-run", help="Monitor and retrieve one diagnostic run without launching dependents")
    parser.add_argument("--summary-name", default="frontal_cache_audit_summary.json")
    parser.add_argument("--full-run",default=FULL_RUN,help="New run identity for a repaired full-evaluation version")
    parser.add_argument("--skip-superseded-pilot",action="store_true",
                        help="Use completed full-bank screen/evidence; do not wait on the older optional pilot")
    args = parser.parse_args()
    record("queue_started", scope="frontal only; no AutoVMD or neural/posterior launch")
    while True:
        try:
            if args.watch_run:
                state = status(args.watch_run)
                if state == "COMPLETE":
                    print(command("retrieve", "--run-id", args.watch_run,
                                  "--file-pattern", REPORT_FILES + "|frontal_cache_audit"), flush=True)
                    collect(args.watch_run, args.summary_name)
                    record("diagnostic_review_checkpoint", run_id=args.watch_run)
                    return
                if state in ("ERROR", "CANCEL_ACKNOWLEDGED"):
                    raise RuntimeError("Diagnostic run stopped: " + args.watch_run + " is " + state)
                if args.once:
                    return
                time.sleep(45)
                continue
            if step(args.full_run,args.skip_superseded_pilot) or args.once:
                return
        except Exception as error:
            record("queue_stopped_for_diagnosis", error=str(error))
            raise
        time.sleep(45)


if __name__ == "__main__":
    main()
