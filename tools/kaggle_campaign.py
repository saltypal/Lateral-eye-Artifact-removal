"""Local authoring/Git/Kaggle CLI orchestration; never imports EEG numerical modules."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil
import re
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
    if args.source_sha:
        if not re.fullmatch(r"[0-9a-f]{40}",args.source_sha):
            raise ValueError("Reproduction needs an exact archived 40-character SHA")
        git("merge-base","--is-ancestor",args.source_sha,sha)
        sha=args.source_sha
    elif git("rev-parse","HEAD") != sha:
        raise RuntimeError("Push current implementation before submitting latest-SHA experiment")
    if git("status","--porcelain","--untracked-files=no"):
        raise RuntimeError("Tracked edits are uncommitted")
    run_id = args.run_id or args.stage+"-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    stage = WORK/run_id
    stage.mkdir(parents=True,exist_ok=False)
    config_path = getattr(args,"config",None) or "configs/campaign.json"
    configuration = json.loads((ROOT/config_path).read_text())
    experiment_path = getattr(args,"experiment_spec",None)
    experiment = json.loads(Path(experiment_path).read_text()) if experiment_path else {}
    spec = {"run_id":run_id,"stage":args.stage,"git_sha":sha,"campaign_id":configuration["campaign_id"],
            "config_path":config_path, "experiment":experiment,
            "profile":args.profile,"parent_run":args.parent_run,
            "source_resolution":"archived reproduction" if args.source_sha else "latest published at launch",
            "kernel_sources":args.kernel_source or [],"dataset_sources":args.dataset_source or []}
    if args.stage not in ("contracts",) and not spec["dataset_sources"]:
        spec["dataset_sources"] = [OWNER+"/lateral-eye-complete-dataset"]
    prefix = "vmd-band-" if configuration.get("schema_version") == 2 else "vmd-eog-"
    slug = getattr(args,"kernel_slug",None) or prefix+args.stage
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,49}",slug):
        raise ValueError("Invalid Kaggle kernel slug")
    kernel = OWNER+"/"+slug
    spec["kernel"] = kernel
    notebook = stage/"experiment.ipynb"
    build(notebook,args.stage,spec)
    metadata = {"id":kernel,"title":"VMD EOG "+args.stage.title(),"code_file":notebook.name,
                "language":"python","kernel_type":"notebook","is_private":True,"enable_internet":True,
                "enable_gpu":args.stage in ("student-paired","student-distill","router-train","autovmd-search"),
                "dataset_sources":spec["dataset_sources"],"kernel_sources":spec["kernel_sources"],"competition_sources":[]}
    (stage/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2))
    (stage/"run_spec.json").write_text(json.dumps(spec,indent=2))
    message=cli(["kernels","push","-p",str(stage)])
    print(message)
    version=re.search(r"Kernel version (\d+) successfully pushed",message)
    if version:
        spec["kernel_version"]=int(version.group(1))
        (stage/"run_spec.json").write_text(json.dumps(spec,indent=2))
    print(json.dumps({"run_id":run_id,"kernel":kernel,"git_sha":sha}))


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    submit_parser=sub.add_parser("submit")
    submit_parser.add_argument("--stage",required=True)
    submit_parser.add_argument("--run-id")
    submit_parser.add_argument("--profile",choices=("pilot","full"),default="pilot")
    submit_parser.add_argument("--parent-run")
    submit_parser.add_argument("--source-sha",help="Exact archived SHA for an explicitly reproducible rerun")
    submit_parser.add_argument("--kernel-source",action="append")
    submit_parser.add_argument("--dataset-source",action="append")
    submit_parser.add_argument("--config",default="configs/campaign.json")
    submit_parser.add_argument("--experiment-spec",help="JSON experiment/fold/cache specification")
    submit_parser.add_argument("--kernel-slug",help="Separate sharded notebooks without overwriting active kernels")
    for command in ("status","retrieve"):
        item=sub.add_parser(command); item.add_argument("--run-id",required=True)
        if command=="retrieve": item.add_argument("--file-pattern")
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
            command=["kernels","output",spec["kernel"],"-p",str(out),"--force","--page-size","200"]
            if args.file_pattern:
                # Identity must accompany even a targeted plot/log download.
                # Otherwise a correct new retrieval misleadingly fails simply
                # because its user-specified pattern excluded run_spec.json.
                pattern="(?:"+args.file_pattern+")|(?:run_spec\\.json$)"
                command += ["--file-pattern",pattern]
            print(cli(command))
            identities=list(out.rglob("run_spec.json"))
            if len(identities)!=1:
                raise RuntimeError("Retrieved output has missing or ambiguous run identity; retain evidence")
            retrieved=json.loads(identities[0].read_text())
            if any(retrieved.get(key)!=spec.get(key) for key in ("run_id","git_sha","stage","campaign_id")):
                raise RuntimeError("Retrieved output identity differs from requested run; retain evidence")
            print("Retrieved:",out)


if __name__=="__main__":
    main()
