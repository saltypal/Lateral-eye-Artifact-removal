"""CLI-only orchestration of reviewed Phase A stages; no EEG calculations locally."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace
from kaggle_campaign import ROOT,WORK,cli,submit
from kaggle_log import fetch_log


def specification(run_id):
    return json.loads((WORK/run_id/"run_spec.json").read_text())


def pinned_kernel(run_id):
    spec=specification(run_id)
    version=spec.get("kernel_version")
    if not version:
        raise RuntimeError(f"Saved parent version absent: {run_id}")
    return spec["kernel"]+"/"+str(version)


def verify_completed_log(run_id):
    spec=specification(run_id)
    log=fetch_log(run_id)
    try:
        events=json.loads(log)
        text="".join(event.get("data","") for event in events if event.get("stream_name")=="stdout")
    except (TypeError,json.JSONDecodeError):
        text=log
    if spec["git_sha"] not in text or run_id not in text or '"status": "complete"' not in text:
        raise RuntimeError(f"Completed kernel log does not verify source/run/completion: {run_id}")
    return {"run_id":run_id,"git_sha":spec["git_sha"],"kernel_source":pinned_kernel(run_id)}


def save_state(path,state):
    state["updated_utc"]=datetime.now(timezone.utc).isoformat()
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state,indent=2),encoding="utf-8")
    temporary.replace(path)


def wait_for(run_ids,interval,state,path):
    verified={}; errors={}
    while len(verified)<len(run_ids):
        for run_id in run_ids:
            if run_id in verified: continue
            try:
                status=cli(["kernels","status",specification(run_id)["kernel"]]).strip()
                errors[run_id]=0
            except Exception as error:
                errors[run_id]=errors.get(run_id,0)+1
                state["statuses"][run_id]="API status unavailable: "+str(error)
                save_state(path,state)
                if errors[run_id]>=3: raise RuntimeError("Repeated API error; numerical job status remains unknown") from error
                continue
            state["statuses"][run_id]=status
            print(datetime.now(timezone.utc).isoformat(),run_id,status,flush=True)
            save_state(path,state)
            if "COMPLETE" in status:
                verified[run_id]=verify_completed_log(run_id)
                state["verified_parents"][run_id]=verified[run_id]
                save_state(path,state)
            elif any(label in status for label in ("ERROR","CANCELLED","CANCELED")):
                try: fetch_log(run_id)
                except Exception: pass
                raise RuntimeError(f"Kernel failed; preserve and diagnose before dependent launch: {run_id}")
        if len(verified)<len(run_ids): time.sleep(interval)
    return verified


def launch(stage,run_id,parents):
    if (WORK/run_id/"run_spec.json").exists():
        spec=specification(run_id)
        if spec["stage"]!=stage or spec["kernel_sources"]!=parents:
            raise RuntimeError("Existing chained run differs; choose a new run ID")
        return
    submit(SimpleNamespace(stage=stage,run_id=run_id,profile="full",parent_run=None,
        kernel_source=parents,dataset_source=None,source_sha=None))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--vmd-run",required=True)
    parser.add_argument("--posterior-run",required=True)
    parser.add_argument("--corpus-run",required=True)
    parser.add_argument("--contracts-run",required=True)
    parser.add_argument("--fixture-run",required=True)
    parser.add_argument("--regional-run",required=True)
    parser.add_argument("--review-run",required=True)
    parser.add_argument("--interval",type=int,default=45)
    args=parser.parse_args()
    if not 30<=args.interval<=60: raise ValueError("Monitoring interval must be 30–60 seconds")
    path=WORK/(args.review_run+"-chain.json")
    state={"phase":"classical-only","status":"monitoring","statuses":{},"verified_parents":{},
        "models_implemented_or_launched":False,"runs":vars(args)}
    save_state(path,state)
    try:
        parents=[args.corpus_run,args.contracts_run,args.fixture_run,args.vmd_run,args.posterior_run]
        wait_for(parents,args.interval,state,path)
        regional_sources=[pinned_kernel(r) for r in (args.corpus_run,args.vmd_run,args.posterior_run)]
        launch("regional",args.regional_run,regional_sources)
        wait_for([args.regional_run],args.interval,state,path)
        review_sources=[pinned_kernel(r) for r in (args.contracts_run,args.corpus_run,args.vmd_run,args.posterior_run,args.regional_run)]
        launch("review",args.review_run,review_sources)
        wait_for([args.review_run],args.interval,state,path)
        state["status"]="review completed; primary agent must inspect scientific gate"
        save_state(path,state)
        print(state["status"],flush=True)
    except Exception as error:
        state.update(status="stopped for diagnosis",error=f"{type(error).__name__}: {error}")
        save_state(path,state)
        raise


if __name__=="__main__": main()
