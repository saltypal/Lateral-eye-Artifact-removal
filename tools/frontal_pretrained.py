"""Build, submit and inspect the pretrained-only frontal closeout notebook."""
import argparse
from datetime import datetime, timezone
import json
import nbformat
from run import ROOT, BRANCH, git, cli
from frontal import launcher

KERNEL = 'satyapaladugu/frontal-pretrained-closeout'


def notebook(spec):
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md('''# 04 — Frontal pretrained-model eligibility and closeout

## Summary

EEG-WNet and Diff-ADN have no verified compatible pretrained release in the
recorded public-source audit. Neither is run. No training or guessed replica.
This notebook verifies immutable completed experiment parents and presents
their existing frontal evidence alongside the explicit release limitations.
Completion of this audit is not deployment qualification.
'''), md('## 1. Exact source and fresh environment'),
    code('RUN_SPEC = ' + repr(spec) + '\n' + '''import os,sys,pathlib,tempfile,subprocess,json,platform
assert pathlib.Path('/kaggle/input').exists(), 'Execute numerical reporting on Kaggle'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
print('BOOTSTRAP_START',RUN_SPEC['run_id'],RUN_SPEC['git_sha'],flush=True)
CHECKOUT=pathlib.Path(tempfile.mkdtemp(prefix='frontal-pretrained-source-'))
subprocess.run(['git','clone','--quiet',RUN_SPEC['git_url'],str(CHECKOUT)],check=True)
subprocess.run(['git','-C',str(CHECKOUT),'checkout','--quiet',RUN_SPEC['git_sha']],check=True)
actual=subprocess.check_output(['git','-C',str(CHECKOUT),'rev-parse','HEAD'],text=True).strip()
assert actual==RUN_SPEC['git_sha']
subprocess.run([sys.executable,'-m','pip','install','-q','pytest==8.3.5','pymatreader==1.0.0',
    'tabulate==0.9.0','torchinfo==1.8.0','PySDKit==0.5.0','condat-tv==0.0.5'],check=True,timeout=300)
sys.path.insert(0,str(CHECKOUT/'src'))
import torch,importlib.metadata
torch.set_num_threads(2)
from eog_clean.data import save_json
OUTPUT=pathlib.Path(RUN_SPEC['output']);OUTPUT.mkdir(parents=True,exist_ok=True)
save_json(OUTPUT/'run_spec.json',RUN_SPEC)
save_json(OUTPUT/'environment.json',{'git_sha':actual,'python':sys.version,'platform':platform.platform(),
    'cuda':torch.cuda.is_available(),'versions':{name:importlib.metadata.version(name) for name in
    ['numpy','scipy','pandas','torch','matplotlib']}})
'''), md('''## 2. Contracts and versioned evidence

Require complete source/checksum-matched parents. A new release cannot be
treated as an evaluated model without a verified checkpoint and inference
adapter. The discovery snapshot records HTTP failures separately from empty
successful searches. No online availability search runs implicitly here.
'''), code('''environment=os.environ.copy();environment['PYTHONPATH']=str(CHECKOUT/'src')
tests=subprocess.run([sys.executable,'-m','pytest','-q',str(CHECKOUT/'tests'),
    '--junitxml='+str(OUTPUT/'contract_tests.xml')],text=True,capture_output=True,env=environment)
(OUTPUT/'contract_tests.txt').write_text(tests.stdout+'\\n'+tests.stderr)
print(tests.stdout,flush=True)
if tests.returncode:raise RuntimeError('Numerical contracts failed')
from eog_clean.pretrained_review import run
FIGURES=run('/kaggle/input',CHECKOUT,OUTPUT)
'''), md('''## 3. Findings and remaining limitations

The new candidates have no measured scores. All charts below carry forward
completed Klados/OSF results; they do not represent new inference. Ordinary
clean-EEG Pearson CC and reconstruction SNR are paired Klados measures only.
Native before/after correlations use unchanged HEOG/VEOG. Large resting-input
modification prevents a deployment claim even when ocular correlation falls.

All processing represented here is frontal-only. There is no central/shared
correction or front-to-back communication. The next regional discussion must
retain this unresolved preservation limitation.
'''), code('''import pandas as pd
from IPython.display import display,Image,Markdown
display(pd.read_csv(OUTPUT/'pretrained_eligibility.csv')[['model','status','reason']])
display(pd.read_csv(OUTPUT/'comparison.csv').round(6))
display(pd.read_csv(OUTPUT/'native_frontal_before_after_summary.csv').round(6))
display(pd.read_csv(OUTPUT/'native_rest_participant_summary.csv').round(6))
display(Markdown((OUTPUT/'RESULTS.md').read_text()))
for path in FIGURES:display(Image(filename=path))
print((OUTPUT/'completion.json').read_text())
save_json(OUTPUT/'execution_state.json',{'state':'complete','git_sha':actual,
    'new_models_run':[],'training_performed':False,'production_qualified':False})
''')]
    return nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec':{
        'name':'python3','display_name':'Python 3','language':'python'}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'submit', 'status'])
    parser.add_argument('--run-id', default='frontal-pretrained-closeout-001')
    args = parser.parse_args()
    if args.action == 'status':
        print(cli('kernels', 'status', KERNEL))
        return
    spec = {'run_id':args.run_id, 'campaign_id':'frontal-pretrained-20261017',
            'kernel':KERNEL, 'git_sha':git('rev-parse','HEAD'),
            'git_url':git('remote','get-url','origin'),
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'output':'/kaggle/working/results/frontal-pretrained-20261017/'+args.run_id}
    if args.action == 'build':
        spec['git_sha'] = 'RESOLVED_TO_PUBLISHED_SHA_AT_SUBMISSION'
        nbformat.write(notebook(spec), ROOT/'notebooks/04_Frontal_Pretrained_Closeout.ipynb')
        return
    if git('ls-remote','origin','refs/heads/'+BRANCH).split()[0] != spec['git_sha'] or git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('Commit and push before exact-SHA submission')
    folder = ROOT/'.kaggle-work'/args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    nbformat.write(launcher(spec,notebook(spec)), folder/'experiment.ipynb')
    metadata = {'id':KERNEL,'title':'Frontal Pretrained Closeout','code_file':'experiment.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_internet':True,
        'enable_gpu':False,'dataset_sources':[], 'competition_sources':[],
        'kernel_sources':['satyapaladugu/klados-frontal-six-methods/5',
                          'satyapaladugu/frontal-mode-clean-eeg-audit/2']}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (folder/'run_spec.json').write_text(json.dumps(spec,indent=2))
    message = cli('kernels','push','-p',str(folder))
    (folder/'submission.txt').write_text(message)
    print(message)


if __name__ == '__main__':
    main()
