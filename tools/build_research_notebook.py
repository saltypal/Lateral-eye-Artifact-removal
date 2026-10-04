"""Build the portable notebook from modular source; no experiments run locally."""
from pathlib import Path
import textwrap
import nbformat


def build(path: Path, phase="audit", sha="") -> None:
    notebook = nbformat.v4.new_notebook()
    md = nbformat.v4.new_markdown_cell
    code = lambda source: nbformat.v4.new_code_cell(textwrap.dedent(source).strip())
    notebook.cells = [md("# Region-aware blink and lateral-eye artifact removal\n\nThis notebook orchestrates reusable Git modules. Audit precedes classical benchmarking, which precedes neural training. No superiority result is assumed. Outputs persist when the completed Kaggle kernel version is saved; retrieve them before resetting the runtime."),
        md("## 1. Clean reset and exact source sync\nA fresh temporary checkout prevents stale code. Local/Colab/Kaggle paths and accelerators are detected; local execution requires an explicit opt-in because this campaign runs experiments on Kaggle."),
        code(f'''
            import os, sys, tempfile, subprocess, pathlib, json, platform, zipfile
            from pathlib import Path
            IN_KAGGLE = Path("/kaggle/input").exists()
            IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
            ENVIRONMENT = "kaggle" if IN_KAGGLE else "colab" if IN_COLAB else "local"
            if not IN_KAGGLE and os.environ.get("EOG_ALLOW_NON_KAGGLE") != "1":
                raise RuntimeError("Kaggle-only campaign. Portable non-Kaggle execution needs explicit EOG_ALLOW_NON_KAGGLE=1.")
            PHASE = {phase!r}
            COMMIT = {sha!r} or os.environ.get("EOG_COMMIT", "research/region-aware-eog-kaggle")
            REPOSITORY_URL = "https://github.com/saltypal/Lateral-eye-Artifact-removal.git"
            # A new clone replaces transient project state without deleting user files.
            CHECKOUT = Path(tempfile.mkdtemp(prefix="eog-source-")) / "repository"
            subprocess.check_call(["git", "clone", "--quiet", REPOSITORY_URL, str(CHECKOUT)])
            subprocess.check_call(["git", "-C", str(CHECKOUT), "checkout", "--quiet", COMMIT])
            GIT_SHA = subprocess.check_output(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True).strip()
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "-r", str(CHECKOUT / "requirements-research.txt")])
            sys.path.insert(0, str(CHECKOUT))
            for module in list(sys.modules):
                if module == "eog_vmd_fcm_bgru" or module.startswith("eog_vmd_fcm_bgru."):
                    del sys.modules[module]
            import torch
            DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
            TPU_AVAILABLE = os.environ.get("TPU_NAME") is not None or os.environ.get("COLAB_TPU_ADDR") is not None
            print(dict(environment=ENVIRONMENT, sha=GIT_SHA, device=DEVICE, gpu_count=torch.cuda.device_count(), tpu_available=TPU_AVAILABLE))
            if TPU_AVAILABLE and not torch.cuda.is_available():
                print("TPU detected. PyTorch-XLA backend not validated; CPU path is explicit.")
            ROOT_WORK = Path("/kaggle/working") if IN_KAGGLE else Path(os.environ.get("EOG_PERSISTENT_RESULTS", "./eog-results")).resolve()
            RUN_ID = PHASE + "-" + GIT_SHA[:12]
            OUT = ROOT_WORK / "results" / RUN_ID
            OUT.mkdir(parents=True, exist_ok=True)
        '''),
        md("## 2. Attach and verify all source data\nThe private upload includes the complete tree. ZIP members are checked before extraction. Every source file is verified against its local SHA256; derivatives do not count as independent recordings."),
        code('''
            if IN_KAGGLE:
                manifests = list(Path("/kaggle/input").rglob("source_manifest.json"))
                if len(manifests) != 1:
                    raise RuntimeError("Attach exactly one complete dataset with source_manifest.json")
                MANIFEST = manifests[0]
                attached = MANIFEST.parent
                archives = list(attached.glob("*.zip")) + list(attached.glob("*.eogbundle"))
                if archives:
                    from eog_vmd_fcm_bgru.provenance import sha256_file
                    container = json.loads(MANIFEST.read_text()).get("archive")
                    if container:
                        expected_archive = attached / container["name"]
                        if expected_archive.stat().st_size != container["bytes"] or sha256_file(expected_archive) != container["sha256"]:
                            raise RuntimeError("Outer research archive failed SHA256 verification")
                    DATA_ROOT = Path(tempfile.mkdtemp(prefix="eog-data-"))
                    for archive_path in archives:
                        with zipfile.ZipFile(archive_path) as archive:
                            for member in archive.infolist():
                                target = (DATA_ROOT / member.filename).resolve()
                                if not target.is_relative_to(DATA_ROOT.resolve()):
                                    raise ValueError("Unsafe ZIP path")
                            archive.extractall(DATA_ROOT)
                    candidates = list(DATA_ROOT.rglob("klados_contaminated_eeg.npy"))
                    if len(candidates) != 1:
                        raise RuntimeError("Ambiguous Klados dataset")
                    DATA_ROOT = candidates[0].parent.parent
                else:
                    candidates = list(attached.rglob("klados_contaminated_eeg.npy"))
                    if len(candidates) != 1:
                        raise RuntimeError("Missing or ambiguous Klados dataset")
                    DATA_ROOT = candidates[0].parent.parent
            else:
                DATA_ROOT = Path(os.environ["EOG_DATA_ROOT"]).resolve()
                MANIFEST = Path(os.environ["EOG_SOURCE_MANIFEST"]).resolve()
            print("Input", DATA_ROOT, "Output", OUT)
        '''),
        md("## 3. Audit gate\nRead original labels without EEG unit conversion. Preserve trial boundaries, original participant identities, and all valid EEG channels. Unknown Klados electrode and subject metadata cannot support anatomical or subject-independent claims."),
        code('''
            from eog_vmd_fcm_bgru.provenance import audit, save_json
            summary = audit(DATA_ROOT, MANIFEST, OUT, CHECKOUT)
            save_json(OUT / "run_config.json", {"phase": PHASE, "git_sha": GIT_SHA, "seed": 42, "data_root": str(DATA_ROOT), "device": DEVICE})
            if not summary["proceed_to_classical_gate"]:
                raise RuntimeError("Audit gate failed; inspect exclusions before running experiments")
        '''),
        md("## 4. Phase execution"),
        code('''
            if PHASE == "benchmark":
                from eog_vmd_fcm_bgru.experiment import benchmark
                benchmark(DATA_ROOT, OUT, profile="kaggle_smoke")
            elif PHASE == "train":
                from eog_vmd_fcm_bgru.training import train_campaign
                train_campaign(DATA_ROOT, OUT, device=DEVICE, profile="kaggle_smoke")
            else:
                print("Audit complete. Review the saved manifest before submitting the benchmark phase.")
        '''),
        md("## 5. Results and persistence\nNo clean EEG exists for OSF. OSF outputs are suppression/preservation proxies. A smoke run cannot pass the full-study victory gate. Save the completed kernel version and retrieve outputs with the campaign controller."),
        code('''
            import pandas as pd
            sessions = json.loads((OUT / "osf_sessions.json").read_text())
            usable = pd.DataFrame([item for item in sessions if item["status"] == "usable"])
            if not usable.empty:
                display(usable.groupby("study").agg(sessions=("session", "count"), participants=("participant", "nunique"), sampling_hz=("sampling_hz", "first")))
            for filename in ("benchmark_summary.csv", "student_metrics.csv"):
                if (OUT / filename).exists():
                    display(pd.read_csv(OUT / filename).round(4))
            from IPython.display import Image
            for name in ("vmd_center_frequency_sweep", "vmd_preservation_tradeoff", "vmd_mode_vectors"):
                if (OUT / (name + ".png")).exists():
                    display(Image(filename=str(OUT / (name + ".png"))))
            print("Saved", len(list(OUT.rglob("*"))), "artifacts in", OUT)
        ''')]
    contract_cell = code('''
        test = subprocess.run([sys.executable, "-m", "pytest", str(CHECKOUT / "tests"), "-q"],
                              cwd=CHECKOUT, capture_output=True, text=True)
        (OUT / "contract_tests.txt").write_text(test.stdout + test.stderr)
        print(test.stdout)
        print(test.stderr)
        if test.returncode:
            raise RuntimeError("Numeric contract tests failed on Kaggle")
    ''')
    if phase == "contracts":
        notebook.cells = notebook.cells[:3] + [md("## Kaggle numeric contract tests\nSynthetic fixtures validate implementation invariants; they are not dataset performance evidence."), contract_cell]
    else:
        notebook.cells.insert(3, contract_cell)
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
    nbformat.validate(notebook)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1] / "notebooks" / "Region_Aware_EOG_Kaggle.ipynb")
