"""Local authoring/Git/Kaggle CLI orchestration; never imports EEG numerical modules."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from build_notebooks import build

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "research/vmd-eog-removal-notebooks"
OWNER = "satyapaladugu"
WORK = ROOT/".kaggle-work"


def cli(args, capture=True):
    config = ROOT/".kaggle-config"
    config.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(KAGGLE_CONFIG_DIR=str(config),PYTHONUTF8="1")
    credential_path = Path.home()/".kaggle/credentials.json"
    if credential_path.exists():
        expiry = json.loads(credential_path.read_text()).get("access_token_expiration")
        if expiry and datetime.fromisoformat(expiry) < datetime.now(timezone.utc)+timedelta(minutes=1):
            os.environ["KAGGLE_CONFIG_DIR"] = str(config)
            from kaggle.api.kaggle_api_extended import KaggleApi
            from kagglesdk.kaggle_creds import KaggleCredentials
            api = KaggleApi(); api.authenticate()
            with api.build_kaggle_client() as client:
                credentials = KaggleCredentials.load(client)
                if credentials is None:
                    raise RuntimeError("OAuth unavailable")
                credentials.refresh_access_token()
    result = subprocess.run([shutil.which("kaggle") or "kaggle",*args],env=env,
                            text=True,encoding="utf-8",capture_output=capture,check=True)
    return result.stdout if capture else ""


def git(*args):
    return subprocess.check_output(["git","-C",str(ROOT),*args],text=True).strip()


def submit(args):
    sha = git("ls-remote","origin","refs/heads/"+BRANCH).split()[0]
    if git("rev-parse","HEAD") != sha:
        raise RuntimeError("Push current implementation before submitting latest-SHA experiment")
    if git("status","--porcelain","--untracked-files=no"):
        raise RuntimeError("Tracked edits are uncommitted")
    run_id = args.run_id or args.stage+"-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    stage = WORK/run_id
    stage.mkdir(parents=True,exist_ok=False)
    spec = {"run_id":run_id,"stage":args.stage,"git_sha":sha,"campaign_id":"vmd-eog-20261010",
            "profile":args.profile,"parent_run":args.parent_run,
            "kernel_sources":args.kernel_source or [],"dataset_sources":args.dataset_source or []}
    if args.stage not in ("contracts",) and not spec["dataset_sources"]:
        spec["dataset_sources"] = [OWNER+"/lateral-eye-complete-dataset"]
    kernel = OWNER+"/vmd-eog-"+args.stage
    spec["kernel"] = kernel
    notebook = stage/"experiment.ipynb"
    build(notebook,args.stage,spec)
    metadata = {"id":kernel,"title":"VMD EOG "+args.stage.title(),"code_file":notebook.name,
                "language":"python","kernel_type":"notebook","is_private":True,"enable_internet":True,
                "enable_gpu":args.stage in ("student-paired","student-distill"),
                "dataset_sources":spec["dataset_sources"],"kernel_sources":spec["kernel_sources"],"competition_sources":[]}
    (stage/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2))
    (stage/"run_spec.json").write_text(json.dumps(spec,indent=2))
    print(cli(["kernels","push","-p",str(stage)]))
    print(json.dumps({"run_id":run_id,"kernel":kernel,"git_sha":sha}))


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    submit_parser=sub.add_parser("submit")
    submit_parser.add_argument("--stage",required=True)
    submit_parser.add_argument("--run-id")
    submit_parser.add_argument("--profile",choices=("pilot","full"),default="pilot")
    submit_parser.add_argument("--parent-run")
    submit_parser.add_argument("--kernel-source",action="append")
    submit_parser.add_argument("--dataset-source",action="append")
    for command in ("status","retrieve"):
        item=sub.add_parser(command); item.add_argument("--run-id",required=True)
    sub.add_parser("auth-check")
    args=parser.parse_args()
    if args.command=="submit":
        submit(args)
    elif args.command=="auth-check":
        print(cli(["kernels","list","--mine","--page-size","5"]))
    else:
        spec=json.loads((WORK/args.run_id/"run_spec.json").read_text())
        if args.command=="status":
            print(cli(["kernels","status",spec["kernel"]]))
        else:
            out=ROOT/"results"/spec["campaign_id"]/args.run_id
            out.mkdir(parents=True,exist_ok=True)
            print(cli(["kernels","output",spec["kernel"],"-p",str(out),"--force"]))
            identities=list(out.rglob("run_spec.json"))
            if len(identities)!=1 or json.loads(identities[0].read_text()).get("run_id")!=args.run_id:
                raise RuntimeError("Retrieved output identity differs from requested run; retain evidence")
            print("Retrieved:",out)


if __name__=="__main__":
    main()
