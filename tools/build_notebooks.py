"""Author notebook JSON without executing numerical code locally."""
import json
from pathlib import Path

STAGES = [
    ("00_Campaign_Control", "contracts", "Validate the reproducible execution environment and numerical contracts."),
    ("01_Dataset_Audit", "audit", "Verify source bytes, EEG/EOG/label semantics, identities and exposure."),
    ("02_Contracts_and_Splits", "contracts", "Validate lag signs, finite boundaries, VMD parity, masks and metrics."),
    ("03_Controlled_Corpus", "corpus", "Freeze participant/donor/recipient roles and generate coherent mixtures."),
    ("04_Frontal_VMD_Search", "vmd", "Compare K=3..10 and five alpha values; VMD must add value beyond regression."),
    ("05_Posterior_Experts", "posterior", "Validate GEVD-MWF and ICA with raw frontal support on unscored calibration."),
    ("06_Regional_Comparison", "regional", "Compare disjoint regional correction, shared processing and context ablations."),
    ("07_Approach_Review", "review", "Review all candidates and preservation; authorize models only on passing gates."),
    ("08_Teacher_Crossfit", "teacher-oof", "After passing gates, generate source-isolated teacher targets."),
    ("09_EEG_Only_Student", "student-paired", "After passing gates, train one EEG-only baseline."),
    ("10_Distillation_Ablations", "student-distill", "Evaluate teacher supervision and declared loss/architecture additions."),
    ("11_Model_Freeze", "freeze", "Freeze development-selected weights and analysis before confirmation access."),
    ("12_Reserved_Evaluation", "final-eval", "Evaluate untouched sources with the frozen cleaner; native data have no paired SNR."),
    ("13_Export_and_Report", "export", "Verify fresh-process deployment and assemble reproducible research evidence.")
]


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(path, stage, run_spec=None):
    spec = run_spec or {"stage": stage, "run_id": stage+"-manual", "git_sha": "", "campaign_id": "vmd-eog-20261010", "profile": "pilot"}
    setup = '''import os, sys, json, tempfile, subprocess
from pathlib import Path
IN_KAGGLE = Path('/kaggle/input').is_dir()
if not IN_KAGGLE:
    raise RuntimeError('This numerical campaign runs on Kaggle only')
TEMP_ROOT = Path(tempfile.mkdtemp(prefix='vmd-eog-'))
CHECKOUT = TEMP_ROOT / 'repository'
REPOSITORY_URL = 'https://github.com/saltypal/Lateral-eye-Artifact-removal.git'
BRANCH = 'research/vmd-eog-removal-notebooks'
''' + 'RUN_SPEC = ' + repr(spec) + '''
# Manual launches resolve latest once; CLI launches already have its exact SHA.
if not RUN_SPEC['git_sha']:
    line = subprocess.check_output(['git','ls-remote',REPOSITORY_URL,'refs/heads/'+BRANCH], text=True).strip()
    RUN_SPEC['git_sha'] = line.split()[0]
subprocess.check_call(['git','clone','--quiet','--branch',BRANCH,'--single-branch',REPOSITORY_URL,str(CHECKOUT)])
subprocess.check_call(['git','-C',str(CHECKOUT),'checkout','--quiet',RUN_SPEC['git_sha']])
ACTUAL_SHA = subprocess.check_output(['git','-C',str(CHECKOUT),'rev-parse','HEAD'],text=True).strip()
if ACTUAL_SHA != RUN_SPEC['git_sha']:
    raise RuntimeError('Source SHA mismatch')
ENV_ROOT = TEMP_ROOT / 'environment'
subprocess.check_call([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(ENV_ROOT)])
ENV_PY = ENV_ROOT / 'bin/python'
subprocess.check_call([sys.executable,'-m','pip','--python',str(ENV_PY),'install','--quiet','--progress-bar','off','-r',str(CHECKOUT/'requirements.txt')])
PROCESS_ENV = os.environ.copy()
PROCESS_ENV['PYTHONPATH'] = str(CHECKOUT/'src')
PROCESS_ENV['PYTHONUNBUFFERED'] = '1'
PROCESS_ENV['OMP_NUM_THREADS'] = '2'
OUT = Path('/kaggle/working/results') / RUN_SPEC['campaign_id'] / RUN_SPEC['run_id']
OUT.mkdir(parents=True,exist_ok=False)
(OUT/'run_spec.json').write_text(json.dumps(RUN_SPEC,indent=2))
print('Exact Git SHA:', ACTUAL_SHA, 'Stage:', RUN_SPEC['stage'], 'Output:',OUT,flush=True)
'''
    execute = '''command = [str(ENV_PY),'-m','vmd_eog.runner','--stage',RUN_SPEC['stage'],
           '--input','/kaggle/input','--output',str(OUT),'--profile',RUN_SPEC['profile']]
process = subprocess.run(command,cwd=CHECKOUT,env=PROCESS_ENV)
if process.returncode:
    raise RuntimeError('Stage failed; inspect saved execution_state.json and kernel logs')
print((OUT/'execution_state.json').read_text())
'''
    inspect = '''from IPython.display import display, Image
for figure in sorted(OUT.glob('*.png')):
    display(Image(filename=str(figure)))
for report in sorted(OUT.glob('*summary.json')):
    print(report.name, report.read_text()[:16000])
print('Persist this completed kernel version and retrieve outputs before launching a dependent stage.')
'''
    descriptions = {item[1]: item[2] for item in STAGES}
    notebook = {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
        "cells":[cell("markdown", "# Regional VMD EOG removal — "+stage+"\n\n"+descriptions.get(stage,stage)),
                 cell("markdown", "## Fresh source and environment\nResolve latest published code once and pin it. New temporary clones avoid stale Python imports; original inputs remain immutable."),
                 cell("code",setup),
                 cell("markdown", "## Execute reusable modules\nHEOG/VEOG are teacher/evaluation information. Clean targets cannot enter inference. Failed numerical or scientific gates stop dependent work."),
                 cell("code",execute),
                 cell("markdown", "## Inspect evidence and persist outputs\nNative real EEG has no clean-reference SNR. Pass-through failures remain counted. Parameters and physiological frequency bands are distinct."),
                 cell("code",inspect)]}
    for index, item in enumerate(notebook["cells"]):
        item["id"] = f"cell-{index}"
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(notebook,indent=2),encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for name, stage, _ in STAGES:
        build(root/"notebooks"/(name+".ipynb"),stage)
    print("Authored",len(STAGES),"notebook workflows; no numerical execution.")
