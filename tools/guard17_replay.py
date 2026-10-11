"""Author/submit the archived guarded recipe; numerical work is Kaggle-only."""
import argparse
import ast
import json
import re
from pathlib import Path

from kaggle_campaign import ROOT, WORK, BRANCH, cli, git


def submit(run_id, diagnose=False):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]+", run_id):
        raise ValueError("Invalid replay run ID")
    config = json.loads((ROOT/"configs/guard17_replay.json").read_text())
    author_sha = git("ls-remote", "origin", "refs/heads/"+BRANCH).split()[0]
    if git("rev-parse", "HEAD") != author_sha or git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Commit and push the replay notebook/configuration before launch")
    spec = {"run_id": run_id, "stage": config["stage"], "campaign_id": config["campaign_id"],
            "git_sha": config["legacy_git_sha"], "orchestrator_git_sha": author_sha,
            "kernel": "satyapaladugu/guarded-vmd-k3-alpha2000-archived-reproduction", "profile": "archived-replay",
            "kernel_sources": config["kernel_sources"], "dataset_sources": config["dataset_sources"],
            "recipe": config, "fresh_confirmation": False, "teacher_qualified": False}
    if diagnose:
        spec["stage"] = "guard17-source-audit"
        spec["kernel"] = "satyapaladugu/guarded-vmd-archived-source-audit"
    notebook_name = "31_Guarded_VMD_Source_Audit.ipynb" if diagnose else "23_Guarded_K3_Alpha2000_Reproduction.ipynb"
    notebook = json.loads((ROOT/"notebooks"/notebook_name).read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = ''.join(cell["source"])
            source = source.replace("RUN_SPEC = None  # injected by guard17_replay.py", "RUN_SPEC = "+repr(spec))
            ast.parse(source)
            cell["source"] = source.splitlines(keepends=True)
            cell["execution_count"], cell["outputs"] = None, []
    folder = WORK/run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder/"experiment.ipynb").write_text(json.dumps(notebook, indent=2))
    title = "Guarded VMD Archived Source Audit" if diagnose else "Guarded VMD K3 Alpha2000 Archived Reproduction"
    metadata = {"id": spec["kernel"], "title": title,
                "code_file": "experiment.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_internet": True, "enable_gpu": False,
                "kernel_sources": spec["kernel_sources"], "dataset_sources": spec["dataset_sources"],
                "competition_sources": []}
    (folder/"kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    (folder/"run_spec.json").write_text(json.dumps(spec, indent=2))
    message = cli(["kernels", "push", "-p", str(folder)])
    (folder/"submission.txt").write_text(message)
    version = re.search(r"Kernel version (\d+) successfully pushed", message)
    if not version:
        raise RuntimeError("Replay submission was not confirmed; inspect before retry")
    spec["kernel_version"] = int(version.group(1))
    (folder/"run_spec.json").write_text(json.dumps(spec, indent=2))
    print(message)
    print(json.dumps({"run_id": run_id, "kernel": spec["kernel"],
                      "legacy_sha": spec["git_sha"], "orchestrator_sha": author_sha}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    submit(args.run_id, args.diagnose)

