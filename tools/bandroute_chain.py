"""Persistent CLI-only scheduling; all numerical work remains in Kaggle.

This controller never repairs scientific failures or changes acceptance gates.
It saves exact accepted notebook versions and verifies completed identities
before attaching a parent to a dependent job.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace
from kaggle_campaign import ROOT,WORK,cli,submit
from kaggle_log import fetch_log

GPU_STAGES = {"router-train","student-paired","student-distill","neural-fixture","autovmd-search"}


def load_spec(run_id):
    return json.loads((WORK/run_id/"run_spec.json").read_text())


def pinned_parent(run_id):
    spec = load_spec(run_id)
    if not spec.get("kernel_version"):
        raise RuntimeError("Parent has no accepted version: " + run_id)
    return spec["kernel"]+"/"+str(spec["kernel_version"])


def verify_log(run_id):
    spec = load_spec(run_id)
    text = fetch_log(run_id)
    events = json.loads(text)
    stdout = ''.join(event.get("data","") for event in events if event.get("stream_name") == "stdout")
    if spec["git_sha"] not in stdout or '"status": "complete"' not in stdout:
        raise RuntimeError("Completed output lacks exact source/completion identity: " + run_id)
    return {"run_id":run_id,"git_sha":spec["git_sha"],"parent":pinned_parent(run_id)}


def save(path,state):
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2),encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue",type=Path,required=True)
    parser.add_argument("--interval",type=int,default=45)
    parser.add_argument("--state-file",type=Path,default=WORK/"bandroute-controller.json",
                        help="Separate state for an independently monitored CPU/GPU queue")
    args = parser.parse_args()
    if not 30 <= args.interval <= 60:
        raise ValueError("Polling must be 30–60 seconds")
    queue = json.loads(args.queue.read_text())
    path = args.state_file
    path.parent.mkdir(parents=True,exist_ok=True)
    state = json.loads(path.read_text()) if path.exists() else {"verified":{},"statuses":{},"api_errors":{}}
    jobs = queue["jobs"]
    required = set(queue["external_parents"])|{job["run_id"] for job in jobs}
    if len({job["run_id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate queued run IDs")
    if state.get("error"):
        state.setdefault("prior_stops",[]).append({
            "error":state.pop("error"),"updated_utc":state.get("updated_utc"),
            "note":"Archived on explicit coordinator restart; failed run evidence remains preserved"})
    state.update(status="monitoring",queue=str(args.queue),scientific_qualification=False)
    save(path,state)
    try:
        while not all(job["run_id"] in state["verified"] for job in jobs):
            active = {"cpu":0,"gpu":0}
            for run_id in sorted(required):
                spec_path = WORK/run_id/"run_spec.json"
                if run_id in state["verified"] or not spec_path.exists():
                    continue
                spec = load_spec(run_id)
                if not spec.get("kernel_version"):
                    raise RuntimeError("Submission was not accepted; inspect before resubmission: " + run_id)
                try:
                    status = cli(["kernels","status",spec["kernel"]]).strip()
                    state["api_errors"][run_id] = 0
                except Exception as error:
                    state["api_errors"][run_id] = state["api_errors"].get(run_id,0)+1
                    state["statuses"][run_id] = "API unavailable; kernel state unknown: " + str(error)
                    save(path,state)
                    if state["api_errors"][run_id] >= 20:
                        raise RuntimeError("Persistent API outage; no numerical failure inferred")
                    active["gpu" if spec["stage"] in GPU_STAGES else "cpu"] += 1
                    continue
                state["statuses"][run_id] = status
                if "COMPLETE" in status:
                    state["verified"][run_id] = verify_log(run_id)
                    print("VERIFIED",run_id,flush=True)
                elif any(label in status for label in ("ERROR","CANCELLED","CANCELED")):
                    try:
                        fetch_log(run_id)
                    except Exception:
                        pass
                    raise RuntimeError("Kaggle job failed; primary engineer must repair: " + run_id)
                else:
                    active["gpu" if spec["stage"] in GPU_STAGES else "cpu"] += 1
                save(path,state)
            for job in jobs:
                run_id = job["run_id"]
                if (WORK/run_id/"run_spec.json").exists() or not all(parent in state["verified"] for parent in job["parents"]):
                    continue
                kind = "gpu" if job["stage"] in GPU_STAGES else "cpu"
                if active[kind] >= queue["max_active"][kind]:
                    continue
                experiment = WORK/(run_id+"-experiment.json")
                experiment.write_text(json.dumps(job.get("experiment",{}),indent=2))
                submit(SimpleNamespace(stage=job["stage"],run_id=run_id,profile=job.get("profile","pilot"),
                    parent_run=None,source_sha=job.get("source_sha"),config="configs/bandroute.json",
                    experiment_spec=str(experiment),kernel_slug=job["kernel_slug"],
                    kernel_source=[pinned_parent(parent) for parent in job["parents"]],dataset_source=None))
                active[kind] += 1
                state["statuses"][run_id] = "submitted"
                save(path,state)
            time.sleep(args.interval)
        state["status"] = "queued numerical stages complete; scientific review required"
        save(path,state)
    except Exception as error:
        state.update(status="stopped for primary diagnosis",error=str(error))
        save(path,state)
        raise


if __name__ == "__main__":
    main()
