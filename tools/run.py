"""Small local notebook author/submission tool. It never imports EEG modules."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import nbformat


ROOT = Path(__file__).resolve().parents[1]
BRANCH = "research/vmd-eog-removal-notebooks"
KERNEL = "satyapaladugu/klados-osf-clean-comparison"


def git(*arguments):
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


def cli(*arguments):
    ensure_auth()
    environment = os.environ.copy()
    environment.update(KAGGLE_CONFIG_DIR=str(ROOT/".kaggle-config"), PYTHONUTF8="1")
    result = subprocess.run(["kaggle", *arguments], env=environment, text=True,
                            capture_output=True, encoding="utf-8", timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def ensure_auth():
    """Refresh short-lived OAuth before calls; never print credential values."""
    credential_path = Path.home()/'.kaggle/credentials.json'
    if not credential_path.is_file():
        return
    expiration = json.loads(credential_path.read_text()).get('access_token_expiration')
    if not expiration or datetime.fromisoformat(expiration) >= datetime.now(timezone.utc)+timedelta(minutes=1):
        return
    os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT/'.kaggle-config')
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kaggle_creds import KaggleCredentials
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        credentials = KaggleCredentials.load(client)
        if credentials is None:
            raise RuntimeError('Kaggle OAuth credentials unavailable')
        credentials.refresh_access_token()


def notebook(spec):
    markdown, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [markdown("""# Klados and OSF: paired comparison and regional validation

This notebook runs the small source package under `src/eog_clean/`.
It consumes Klados and original OSF only. No generated recipient EEG mixtures,
VMD–SOBI, anchored corrections or AutoVMD search enter this run.

**Questions:** Can the historical EOG-assisted VMD checkpoint be reproduced?
How do regression, ICA and MWF compare on paired Klados data? What does fixed
frontal VMD plus separate nonfrontal correction do on named-channel native OSF?

Klados electrode order is unverified: no new frontal/posterior claim is made
there. OSF has no paired clean target: no native reconstruction SNR is invented.
The historical replay is development-exposed and requires HEOG/VEOG at inference.
"""), markdown("## 1. Exact source, fresh checkout and dependencies"),
        code("RUN_SPEC = "+repr(spec)+"\n"+'''import json, os, pathlib, platform, subprocess, sys, tempfile
from datetime import datetime, timezone
if not pathlib.Path('/kaggle/input').exists():
    raise RuntimeError('Numerical work for this campaign must run on Kaggle')
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
print('BOOTSTRAP_START', RUN_SPEC['run_id'], RUN_SPEC['git_sha'], flush=True)
CHECKOUT = pathlib.Path(tempfile.mkdtemp(prefix='klados-osf-source-'))
subprocess.run(['git', 'clone', '--quiet', RUN_SPEC['git_url'], str(CHECKOUT)], check=True)
subprocess.run(['git', '-C', str(CHECKOUT), 'checkout', '--quiet', RUN_SPEC['git_sha']], check=True)
actual = subprocess.check_output(['git','-C',str(CHECKOUT),'rev-parse','HEAD'], text=True).strip()
assert actual == RUN_SPEC['git_sha']
print('BOOTSTRAP_DEPENDENCIES', flush=True)
# Use the existing coherent Kaggle numerical stack; install only small readers
# and test/report tools. The actual complete environment is recorded below.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'pymatreader==1.0.0', 'pytest==8.3.5', 'tabulate==0.9.0'], check=True, timeout=180)
sys.path.insert(0, str(CHECKOUT/'src'))
OUTPUT = pathlib.Path('/kaggle/working/results')/RUN_SPEC['campaign_id']/RUN_SPEC['run_id']
OUTPUT.mkdir(parents=True, exist_ok=False)
import importlib.metadata
import torch
torch.set_num_threads(2)
from eog_clean.data import save_json
save_json(OUTPUT/'run_spec.json', RUN_SPEC)
save_json(OUTPUT/'environment.json', {
    'actual_git_sha':actual, 'python':sys.version, 'platform':platform.platform(),
    'packages':{name:importlib.metadata.version(name) for name in
        ['numpy','scipy','scikit-learn','pandas','matplotlib','pymatreader','pytest','torch','tabulate']},
    'hardware':'Kaggle CPU', 'torch_cuda_available':torch.cuda.is_available(),
    'utc':datetime.now(timezone.utc).isoformat()})
CONFIG = json.loads((CHECKOUT/'configs/experiment.json').read_text())
CONFIG.update(RUN_SPEC.get('config_overrides', {}))
save_json(OUTPUT/'configuration.json', CONFIG)
print('SOURCE_VERIFIED', actual, flush=True)
'''), markdown("""## 2. Numerical contracts

These tests check reconstruction, reference axes, signed lateral projection,
calibration boundaries, identity initialization and overlap-add boundaries.
They validate implementation contracts, not denoising accuracy.
"""), code('''environment = os.environ.copy()
environment['PYTHONPATH'] = str(CHECKOUT/'src')
tests = subprocess.run([sys.executable,'-m','pytest','-q',str(CHECKOUT/'tests'),
    '--junitxml='+str(OUTPUT/'contract_tests.xml')], text=True, capture_output=True, env=environment)
(OUTPUT/'contract_tests.txt').write_text(tests.stdout+'\\n'+tests.stderr)
print(tests.stdout, flush=True)
if tests.returncode:
    save_json(OUTPUT/'execution_state.json', {'state':'failed_contracts','returncode':tests.returncode})
    raise RuntimeError('Numerical contracts failed; inspect saved test report')
'''), markdown("""## 3. Verify original sources and named channel groups

The source container and selected original files must match their hashes.
Only Klados and OSF members are extracted. OSF confirmation participants stay
excluded using the existing frozen participant manifest. Annotation channels are
never scaled or filtered as EEG. Each trial is filtered independently.
"""), code('''from eog_clean.data import prepare_sources, osf_sessions, osf_trials
source_work = pathlib.Path(tempfile.mkdtemp(prefix='klados-osf-data-'))
ROOT_DATA = prepare_sources('/kaggle/input', source_work/'originals', OUTPUT, CONFIG['source_supplement_sha256'])
PARTITION = json.loads((CHECKOUT/CONFIG['confirmation_manifest']).read_text())['partitions']
selected = osf_sessions(ROOT_DATA, PARTITION, CONFIG['osf_max_sessions'])
if not selected:
    raise ValueError('No eligible named-channel OSF development sessions')
first_trials, first_metadata = osf_trials(selected[0])
save_json(OUTPUT/'first_osf_audit.json', first_metadata)
print('OSF_SOURCE_READY', first_metadata['record'], 'trials',len(first_trials),
      'regions', {group:first_metadata['regions'].count(group) for group in set(first_metadata['regions'])}, flush=True)
del first_trials
'''), markdown("""## 4. Execute matched comparisons and save each record

Klados: calibration uses the first 2,000 samples; scoring uses the declared three
1,024-sample windows. Calibration and scoring are filtered separately. No clean
target is used to fit the classical methods. MWF calibration uses EOG activity
quantiles as an approximate ocular/rest split, explicitly labelled.

OSF: block 1 calibrates; block 2 scores, when declared blocks are available.
Otherwise a chronological whole-trial prefix is used. Region-local ICA/MWF
models avoid hidden montage support choices. Frontal VMD uses K=3, alpha=2000;
every mode is checked against both HEOG and VEOG. Posterior/central channels
receive their own estimates. Regional masks are disjoint, then one subtraction
is performed relative to original EEG. Unknown anatomy passes through.

Direct-window regression is the reference-access-matched VMD control. Native
EEG–EOG correlation is a suppression proxy; resting change guards preservation.
"""), code('''from eog_clean.experiment import run
run(ROOT_DATA, '/kaggle/input', OUTPUT, CONFIG, PARTITION)
from eog_clean.report import write_review
write_review(OUTPUT)
'''), markdown("""## 5. Result tables

Paper metrics: MSE, RMSE, temporal RRMSE and ordinary Pearson CC. Project output
SNR is 10 log10(sum(target²)/sum(error²)); improvement is a separate column.
The pooled-window SNR and mean channel SNR use different aggregation and are
reported separately. Records/participants receive equal weight within the
declared summary. Native OSF receives no paired reconstruction score.
"""), code('''import pandas as pd
from IPython.display import display
display(pd.read_csv(OUTPUT/'klados_record_macro.csv').round(5))
display(pd.read_csv(OUTPUT/'osf_participant_macro.csv').round(5))
display(pd.read_csv(OUTPUT/'klados_clean_preservation.csv').groupby('method').mean(numeric_only=True).round(5))
print(json.dumps(json.loads((OUTPUT/'historical_replay.json').read_text()), indent=2))
'''), markdown("""## 6. Diagnostic figures

Figures retain source amplitude scales. Error bars labelled SD are variability,
not confidence intervals. Mode indices are ordered by measured frequency; no
mode is labelled ocular solely because it is the lowest-frequency component.
"""), code('''from eog_clean.plots import make_figures
from IPython.display import Image, display
FIGURES = make_figures(OUTPUT)
for path in FIGURES:
    display(Image(filename=path))
'''), markdown("""## 7. Recovery manifest and interpretation boundary

Every payload below is hashed for retrieval. Completion means the notebook ran;
accuracy, clean preservation and generalization must be assessed from the saved
tables. A 17.9 dB archived replay cannot qualify fresh regional performance.
"""), code('''from eog_clean.data import sha256
save_json(OUTPUT/'execution_state.json', {'state':'complete', 'run_id':RUN_SPEC['run_id'],
    'git_sha':actual, 'research_qualified':False, 'plots_saved':len(FIGURES)})
files = [{'path':str(path.relative_to(OUTPUT)), 'bytes':path.stat().st_size, 'sha256':sha256(path)}
         for path in sorted(OUTPUT.rglob('*')) if path.is_file() and path.name != 'artifact_manifest.json']
save_json(OUTPUT/'artifact_manifest.json', {'run_id':RUN_SPEC['run_id'], 'git_sha':actual,
    'files':files, 'payload_count':len(files), 'complete':True, 'research_qualified':False})
print('RUN_COMPLETE', RUN_SPEC['run_id'], 'payloads',len(files), flush=True)
''')]
    result = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "name": "python3", "display_name": "Python 3", "language": "python"}})
    nbformat.validate(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "submit", "status", "retrieve"])
    parser.add_argument("--run-id", default="klados-osf-pilot-001")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.action == "status":
        print(cli("kernels", "status", KERNEL))
        return
    if args.action == "retrieve":
        destination = ROOT/"results/klados-osf-20261017"/args.run_id
        destination.mkdir(parents=True, exist_ok=True)
        print(cli("kernels", "output", KERNEL, "-p", str(destination), "--file-pattern", "results/|log|ipynb", "--page-size", "200"))
        manifests = list(destination.rglob("artifact_manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError("No unique completed artifact manifest; partial/failed outputs remain saved")
        manifest = json.loads(manifests[0].read_text())
        expected = json.loads((ROOT/".kaggle-work"/args.run_id/"run_spec.json").read_text())
        if manifest["run_id"] != args.run_id or manifest["git_sha"] != expected["git_sha"]:
            raise ValueError("Retrieved notebook version does not match the requested run")
        for entry in manifest["files"]:
            path = (manifests[0].parent/entry["path"]).resolve()
            if not path.is_relative_to(manifests[0].parent.resolve()):
                raise ValueError("Artifact path escapes its result directory")
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise ValueError("Artifact missing or truncated: "+entry["path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError("Artifact checksum mismatch: "+entry["path"])
        receipt = {"run_id": args.run_id, "git_sha": manifest["git_sha"],
            "payloads_verified": len(manifest["files"]), "all_manifest_payloads_verified": True,
            "accuracy_or_production_qualification_claimed": False}
        (destination/"retrieval_receipt.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt, indent=2))
        return
    config = json.loads((ROOT/"configs/experiment.json").read_text())
    sha = git("rev-parse", "HEAD")
    spec = {"run_id": args.run_id, "campaign_id": config["campaign_id"], "git_sha": sha,
            "git_url": git("remote", "get-url", "origin"), "kernel": KERNEL,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_overrides": {"klados_record_limit": 8, "osf_max_sessions": 1,
                                 "osf_max_scoring_trials_per_condition": 1} if args.pilot else {}}
    if args.action == "build":
        spec["git_sha"] = "RESOLVED_TO_PUBLISHED_SHA_AT_SUBMISSION"
        destination = ROOT/"notebooks/Frontal_Comparison.ipynb"
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook(spec), destination)
        print(destination)
        return
    published = git("ls-remote", "origin", "refs/heads/"+BRANCH).split()[0]
    if sha != published or git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Commit and push tracked changes before exact-SHA submission")
    folder = ROOT/".kaggle-work"/args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    nbformat.write(notebook(spec), folder/"experiment.ipynb")
    metadata = {"id": KERNEL, "title": "Klados OSF Clean Comparison", "code_file": "experiment.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_internet": True, "enable_gpu": False, "dataset_sources": config["dataset_sources"],
        "kernel_sources": config["kernel_sources"], "competition_sources": []}
    (folder/"kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    (folder/"run_spec.json").write_text(json.dumps(spec, indent=2))
    message = cli("kernels", "push", "-p", str(folder))
    (folder/"submission.txt").write_text(message)
    print(message)


if __name__ == "__main__":
    main()
