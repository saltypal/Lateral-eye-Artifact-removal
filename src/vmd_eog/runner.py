"""Kaggle stage entrypoint, with durable failure status and immutable run identity."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import traceback
import time
import resource
from .contracts import require_kaggle, require_approval
from .io import atomic_json, sha256_file
from .artifacts import record_parent_identities


def execute(stage, input_root, output, profile, config_path="configs/campaign.json", experiment=None):
    require_kaggle()
    root=Path(__file__).resolve().parents[2]
    config=json.loads((root/config_path).read_text())
    if experiment:
        atomic_json(output/"experiment.json", experiment)
    atomic_json(output/"configuration.json",config)
    record_parent_identities(input_root,output,config)
    require_approval(stage,input_root,config)
    packages={}
    for name in ("numpy","scipy","mne","pandas","scikit-learn","vmdpy","torch","python-picard"):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    hardware=subprocess.run(["bash","-lc","nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true"],capture_output=True,text=True).stdout.strip()
    atomic_json(output/"environment.json",{"python":sys.version,"platform":platform.platform(),"packages":packages,"gpu":hardware or None,
        "git_sha":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()})
    result=subprocess.run([sys.executable,"-m","pytest",str(root/"tests"),"-q"],cwd=root,capture_output=True,text=True)
    (output/"tests.txt").write_text(result.stdout+result.stderr)
    print(result.stdout,result.stderr,flush=True)
    atomic_json(output/("contracts_summary.json" if stage=="contracts" else "stage_tests_summary.json"),
        {"passed":result.returncode==0,"tests_sha256":sha256_file(output/"tests.txt")})
    if result.returncode: raise RuntimeError("Numerical contracts failed")
    if stage=="contracts": return
    from . import campaign
    if config.get("schema_version") == 2 and stage in ("research-ready", "neural-fixture", "autovmd-cache", "autovmd-search", "router-train", "student-paired", "neural-review"):
        from .neural_campaign import execute as neural_execute
        neural_execute(stage,Path(input_root),output,config,profile,experiment or {})
    else:
        campaign.execute(stage,Path(input_root),output,config,profile)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",required=True); parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); parser.add_argument("--profile",default="pilot")
    parser.add_argument("--config",default="configs/campaign.json")
    parser.add_argument("--experiment",type=Path)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    atomic_json(args.output/"execution_state.json",{"stage":args.stage,"status":"running"})
    try:
        experiment=json.loads(args.experiment.read_text()) if args.experiment else None
        execute(args.stage,args.input,args.output,args.profile,args.config,experiment)
    except Exception as error:
        atomic_json(args.output/"execution_state.json",{"stage":args.stage,"status":"failed","error":str(error),"traceback":traceback.format_exc()})
        raise
    atomic_json(args.output/"resource_summary.json",{"wall_time_s":time.perf_counter()-started,
        "peak_resident_memory_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    artifacts={str(path.relative_to(args.output)):sha256_file(path) for path in args.output.rglob("*") if path.is_file() and path.name not in ("execution_state.json","artifact_manifest.json")}
    atomic_json(args.output/"artifact_manifest.json",artifacts)
    atomic_json(args.output/"execution_state.json",{"stage":args.stage,"status":"complete","artifacts":len(artifacts)})


if __name__=="__main__":
    main()
