"""Portable Git-backed notebook; numerical work uses a fresh isolated interpreter."""
from pathlib import Path
import textwrap
import nbformat


def build(path: Path, phase="audit", sha=""):
    notebook = nbformat.v4.new_notebook()
    md = nbformat.v4.new_markdown_cell
    def code(source):
        return nbformat.v4.new_code_cell(textwrap.dedent(source).strip())
    settings = f"PHASE = {phase!r}\nCOMMIT = {sha!r} or os.environ.get('EOG_COMMIT', 'research/region-aware-eog-kaggle')\n"
    setup = '''
import os, sys, tempfile, subprocess, json, hashlib, zipfile
from pathlib import Path
''' + settings + '''
IN_KAGGLE = Path("/kaggle/input").exists()
IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
ENVIRONMENT = "kaggle" if IN_KAGGLE else "colab" if IN_COLAB else "local"
if not IN_KAGGLE and os.environ.get("EOG_ALLOW_NON_KAGGLE") != "1":
    raise RuntimeError("Kaggle-only campaign; portable execution needs explicit EOG_ALLOW_NON_KAGGLE=1")
TEMP_ROOT = Path(tempfile.mkdtemp(prefix="eog-run-"))
CHECKOUT = TEMP_ROOT / "repository"
subprocess.check_call(["git", "clone", "--quiet", "https://github.com/saltypal/Lateral-eye-Artifact-removal.git", str(CHECKOUT)])
subprocess.check_call(["git", "-C", str(CHECKOUT), "checkout", "--quiet", COMMIT])
GIT_SHA = subprocess.check_output(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True).strip()
ENV_ROOT = TEMP_ROOT / "environment"
subprocess.check_call([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(ENV_ROOT)])
ENV_PY = ENV_ROOT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
# Kaggle's OS Python omits ensurepip; host pip can install into a pip-free venv.
subprocess.check_call([sys.executable, "-m", "pip", "--python", str(ENV_PY), "install", "--quiet", "--progress-bar", "off", "-r", str(CHECKOUT / "requirements-research.txt")])
PROCESS_ENV = os.environ.copy()
PROCESS_ENV["PYTHONPATH"] = str(CHECKOUT)
PROCESS_ENV["PYTHONUNBUFFERED"] = "1"
hardware = subprocess.check_output([str(ENV_PY), "-c", "import torch,json; print(json.dumps(dict(device='cuda:0' if torch.cuda.is_available() else 'cpu',gpu_count=torch.cuda.device_count())))"], text=True, env=PROCESS_ENV)
print("Environment", ENVIRONMENT, "Commit", GIT_SHA, "Hardware", hardware.strip())
if os.environ.get("TPU_NAME") or os.environ.get("COLAB_TPU_ADDR"):
    print("TPU detected. PyTorch-XLA is unvalidated; explicit CPU path applies without CUDA.")
ROOT_WORK = Path("/kaggle/working") if IN_KAGGLE else Path(os.environ.get("EOG_PERSISTENT_RESULTS", "./eog-results")).resolve()
OUT = ROOT_WORK / "results" / (PHASE + "-" + GIT_SHA[:12])
OUT.mkdir(parents=True, exist_ok=True)
'''
    notebook.cells = [
        md("# Region-aware blink and lateral-eye artifact removal\n\nAudit, VMD/ICA/ASR/ARMBR development search, regional comparison, and one EEG-only student. The feasibility profile does not establish complete removal or final superiority. Source modules and exact commits are the authority."),
        md("## 1. Clean source reset and isolated runtime\nA fresh Git checkout and Python environment prevent stale code and already-loaded NumPy binary conflicts. Runtime Torch/accelerators remain available through system packages; pinned analysis dependencies install into the isolated environment."),
        code(setup),
        md("## 2. Numeric contract tests in a fresh interpreter\nSynthetic fixtures test alignment, masking/permutation, checkpoint reload and annotation handling. They are not cleaning-quality evidence."),
        code('''
            test = subprocess.run([str(ENV_PY), "-m", "pytest", str(CHECKOUT / "tests"), "-q"], cwd=CHECKOUT,
                                  env=PROCESS_ENV, capture_output=True, text=True)
            (OUT / "contract_tests.txt").write_text(test.stdout + test.stderr)
            print(test.stdout)
            print(test.stderr)
            if test.returncode:
                raise RuntimeError("Kaggle numeric contract tests failed")
        ''')]
    if phase != "contracts":
        notebook.cells += [
            md("## 3. Verify and unpack the complete private input\nThe opaque ZIP preserves original nested archives and hidden metadata. Container and individual-file SHA256 checks prevent silently omitted source data."),
            code('''
                if IN_KAGGLE:
                    manifests = list(Path("/kaggle/input").rglob("source_manifest.json"))
                    if len(manifests) != 1:
                        raise RuntimeError("Attach exactly one complete research dataset")
                    MANIFEST = manifests[0]
                    attached = MANIFEST.parent
                    payload = json.loads(MANIFEST.read_text())
                    archives = list(attached.glob("*.eogbundle")) + list(attached.glob("*.zip"))
                    if archives:
                        container = payload.get("archive")
                        if container:
                            print("Verifying outer archive", container["name"], flush=True)
                            expected_archive = attached / container["name"]
                            digest = hashlib.sha256()
                            with expected_archive.open("rb") as handle:
                                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                                    digest.update(block)
                            if expected_archive.stat().st_size != container["bytes"] or digest.hexdigest() != container["sha256"]:
                                raise RuntimeError("Outer research archive SHA256 mismatch")
                        extraction = TEMP_ROOT / "data"
                        extraction.mkdir()
                        print("Extracting complete research input", flush=True)
                        for archive_path in archives:
                            with zipfile.ZipFile(archive_path) as archive:
                                for member in archive.infolist():
                                    target = (extraction / member.filename).resolve()
                                    if not target.is_relative_to(extraction.resolve()):
                                        raise ValueError("Unsafe ZIP path")
                                archive.extractall(extraction)
                        attached = extraction
                    candidates = list(attached.rglob("klados_contaminated_eeg.npy"))
                    if len(candidates) != 1:
                        raise RuntimeError("Missing or ambiguous Klados dataset")
                    DATA_ROOT = candidates[0].parent.parent
                else:
                    DATA_ROOT = Path(os.environ["EOG_DATA_ROOT"]).resolve()
                    MANIFEST = Path(os.environ["EOG_SOURCE_MANIFEST"]).resolve()
                supplements = list(Path("/kaggle/input").rglob("supplement_manifest.json")) if IN_KAGGLE else []
                if len(supplements) > 1:
                    raise RuntimeError("Attach at most one verified OSF restoration output")
                if PHASE == "train" and len(supplements) != 1:
                    raise RuntimeError("Training requires the completed study04 restoration output")
                if supplements:
                    subprocess.check_call([str(ENV_PY), "-m", "eog_vmd_fcm_bgru.bundle_io", "--root", str(DATA_ROOT),
                                           "--manifest", str(supplements[0]), "--output", str(OUT / "attached_supplement.json")],
                                          cwd=CHECKOUT, env=PROCESS_ENV)
                print("Input", DATA_ROOT, "Output", OUT)
            '''),
            md("## 4. Dataset audit and gated execution\nSource modules run in the isolated interpreter. Every-file integrity, aligned paired arrays and usable original OSF sessions are required. OSF labels stay integer annotations, trials remain separate, and unknown Klados channel/subject mapping cannot support anatomical/subject-independent claims."),
            code('''
                subprocess.check_call([str(ENV_PY), "-m", "eog_vmd_fcm_bgru.runner", "--phase", PHASE,
                                       "--data-root", str(DATA_ROOT), "--manifest", str(MANIFEST), "--output", str(OUT)],
                                      cwd=CHECKOUT, env=PROCESS_ENV)
            '''),
            md("## 5. Saved evidence and persistence\nOSF results are suppression/preservation proxies because clean targets do not exist. A smoke run cannot satisfy the full five-fold/three-seed victory gate. Save the completed Kaggle kernel version and retrieve outputs; /kaggle/working alone is temporary."),
            code('''
                from IPython.display import Image, HTML, display
                import csv, html
                for filename in ("audit_summary.json", "classical_gate.json", "selected_vmd.json", "training_summary.json"):
                    if (OUT / filename).exists():
                        print(filename, (OUT / filename).read_text())
                for filename in ("benchmark_summary.csv", "student_metrics.csv"):
                    if (OUT / filename).exists():
                        with (OUT / filename).open() as handle:
                            rows = list(csv.reader(handle))
                        markup = "<table>" + "".join("<tr>" + "".join("<td>" + html.escape(value) + "</td>" for value in row) + "</tr>" for row in rows[:25]) + "</table>"
                        display(HTML(markup))
                for name in ("vmd_center_frequency_sweep", "vmd_preservation_tradeoff", "vmd_mode_vectors", "student_heldout_waveforms", "student_channel_scaling"):
                    if (OUT / (name + ".png")).exists():
                        display(Image(filename=str(OUT / (name + ".png"))))
                print("Saved artifacts", len(list(OUT.rglob("*"))), "in", OUT)
            ''')]
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
    nbformat.validate(notebook)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1] / "notebooks" / "Region_Aware_EOG_Kaggle.ipynb")
