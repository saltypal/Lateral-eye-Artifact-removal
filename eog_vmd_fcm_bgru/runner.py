"""Fresh-interpreter experiment entry point; notebooks only orchestrate."""
import argparse
from pathlib import Path
import subprocess
import torch
from .provenance import audit, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["audit", "restore", "benchmark", "calibration", "klados-source", "train", "neural-search", "vmd-convergence", "report"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sha = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    save_json(args.output / "run_config.json", {"phase": args.phase, "git_sha": sha, "seed": 42,
              "data_root": str(args.data_root), "device": device, "gpu_count": torch.cuda.device_count(),
              "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]})
    if args.phase == "report":
        from .campaign_report import build_report
        build_report(args.output, repository)
        return
    if args.phase == "restore":
        from .restore_sources import restore_study04
        restore_study04(args.data_root, repository / "data_provenance/osf_study04_catalog.json", args.output)
    summary = audit(args.data_root, args.manifest, args.output, repository)
    if not summary["proceed_to_classical_gate"]:
        raise RuntimeError("Dataset audit gate failed; inspect saved exclusions")
    if args.phase == "benchmark":
        from .experiment import benchmark
        benchmark(args.data_root, args.output, profile="kaggle_smoke")
    elif args.phase == "train":
        from .training import train_campaign
        train_campaign(args.data_root, args.output, device=device, profile="kaggle_smoke")
    elif args.phase == "neural-search":
        from .neural_search import search_student
        search_student(args.data_root, args.output, device=device)
    elif args.phase == "vmd-convergence":
        from .vmd_convergence_check import check_vmd_convergence
        check_vmd_convergence(args.data_root, args.output)
    elif args.phase == "calibration":
        from .calibration_check import check_calibration
        check_calibration(args.data_root, args.output)
    elif args.phase == "klados-source":
        from .klados_source_check import inspect_originals
        inspect_originals(args.data_root, args.output, repository / "data_provenance/klados_mendeley_v4_catalog.json")
    print("Phase completed", args.phase, "at", sha, flush=True)


if __name__ == "__main__":
    main()
