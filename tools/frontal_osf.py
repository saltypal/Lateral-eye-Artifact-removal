"""Submit the frozen frontal correlation audit on Kaggle, without local EEG work."""
import argparse
from datetime import datetime,timezone
import json
import nbformat
from run import ROOT,BRANCH,git,cli
from frontal import launcher

KERNEL='satyapaladugu/frontal-klados-and-osf-correlations'


def notebook(spec):
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells=[md('''# Frontal only: paired clean EEG and before/after EOG correlation

Klados: independently recompute correlation with paired clean EEG from saved
six-method predictions. Write names and original row indices into every labelled
prediction/mode NPZ. Labels follow the requested paper-list assumption.

Native OSF: compare unchanged-reference HEOG/VEOG correlations before and after,
separately for blink, lateral, vertical and rest. Use only exact common frontal
names F3/F4/F7/F8/Fz; FPz is not an alias for missing FP1/FP2. No posterior or
central processing. No native reconstruction SNR or fictitious clean target.
'''),md('## 1. Exact source, frozen parent and environment'),
    code('RUN_SPEC = '+repr(spec)+'\n'+'''import os,sys,pathlib,tempfile,subprocess,json,platform
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
assert pathlib.Path('/kaggle/input').exists()
print('BOOTSTRAP_START',RUN_SPEC['run_id'],RUN_SPEC['git_sha'],flush=True)
CHECKOUT=pathlib.Path(tempfile.mkdtemp(prefix='frontal-osf-source-'))
subprocess.run(['git','clone','--quiet',RUN_SPEC['git_url'],str(CHECKOUT)],check=True)
subprocess.run(['git','-C',str(CHECKOUT),'checkout','--quiet',RUN_SPEC['git_sha']],check=True)
actual=subprocess.check_output(['git','-C',str(CHECKOUT),'rev-parse','HEAD'],text=True).strip()
assert actual==RUN_SPEC['git_sha']
subprocess.run([sys.executable,'-m','pip','install','-q','pymatreader==1.0.0',
    'pytest==8.3.5','tabulate==0.9.0','torchinfo==1.8.0','PySDKit==0.5.0','condat-tv==0.0.5'],
    check=True,timeout=300)
sys.path.insert(0,str(CHECKOUT/'src'))
import torch,importlib.metadata
torch.set_num_threads(2)
from eog_clean.data import save_json
OUTPUT=pathlib.Path(RUN_SPEC['output']);OUTPUT.mkdir(parents=True,exist_ok=True)
CONFIG=json.loads((CHECKOUT/'configs/frontal_osf.json').read_text())
CONFIG.update(RUN_SPEC.get('config_overrides',{}))
save_json(OUTPUT/'run_spec.json',RUN_SPEC);save_json(OUTPUT/'configuration.json',CONFIG)
save_json(OUTPUT/'environment.json',{'git_sha':actual,'python':sys.version,
    'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'versions':{name:importlib.metadata.version(name) for name in
      ['numpy','scipy','pandas','torch','matplotlib','PySDKit','condat-tv','pymatreader']}})
print('SOURCE_VERIFIED',actual,flush=True)
'''),md('''## 2. Numerical contracts

The new checks verify exact named-channel packing and reject FPz→FP1 aliases.
Existing tests cover signal boundaries, covariance, metrics, VMD and GMETV.
'''),code('''environment=os.environ.copy();environment['PYTHONPATH']=str(CHECKOUT/'src')
result=subprocess.run([sys.executable,'-m','pytest','-q',str(CHECKOUT/'tests'),
    '--junitxml='+str(OUTPUT/'contract_tests.xml')],text=True,capture_output=True,env=environment)
(OUTPUT/'contract_tests.txt').write_text(result.stdout+'\\n'+result.stderr)
print(result.stdout,flush=True)
if result.returncode:raise RuntimeError('Contract failure; stop native evaluation')
'''),md('''## 3. Frozen-model frontal evaluation

The parent is exact Kaggle version5, run frontal-six-full-001, sourcec196374.
Verify parent checksums before loading models and predictions. Recompute ordinary
Pearson CC independently with NumPy and compare it against original saved metrics.

OSF source and restoration checksums must verify. Confirmation participants stay
closed. Use block1 calibration/block2 scoring, or the declared whole-trial prefix
fallback. Filter each continuous trial separately. Four development sessions and
the first two scoring trials per condition are the declared bounded evaluation.

Historical fixed VMD uses K3/alpha2000 and original model-feature conventions.
SVMD–FCM/VME–GMETV retain the previous declared adaptations. Neural weights are
frozen from Klados; no native training or test-driven tuning. EEGOAR's two missing
custom-montage channels are zero/masked; this transfer limitation remains visible.
GEVD–MWF fits native ocular/rest covariance on unscored calibration only.

All methods operate on the same five named frontal signals. Reconstruct complete
trials with overlap-add. Calculate before/after correlations with the same original
HEOG_lpf/VEOG_lpf references; preserve signed values and average absolute values
before participant aggregation. Save every mode's H/V evidence and all failures.
'''),code('''from eog_clean.frontal_correlations import run
run('/kaggle/input',OUTPUT,CHECKOUT,CONFIG)
from eog_clean.correlation_figures import write_review
write_review(OUTPUT)
'''),md('## 4. Explicit before/after tables'),code('''import pandas as pd
from IPython.display import display
display(pd.read_csv(OUTPUT/'klados_correlation_summary.csv').round(6))
display(pd.read_csv(OUTPUT/'osf_frontal_correlation_summary.csv').round(6))
display(pd.read_csv(OUTPUT/'osf_frontal_participant_correlations.csv').round(6))
display(pd.read_csv(OUTPUT/'osf_frontal_rest_preservation.csv').round(6))
print((OUTPUT/'native_completion.json').read_text())
print((OUTPUT/'osf_method_failures.json').read_text())
'''),md('''## 5. Frontal figures and interpretation

Clean-reference CC is available only for paired Klados. Native resting-input CC
is a preservation proxy. Lower EOG correlation alone can also result from losing
neural EEG; inspect rest modification, spectral and covariance results together.
'''),code('''from eog_clean.correlation_figures import make_figures
from IPython.display import Image,display
FIGURES=make_figures(OUTPUT)
for path in FIGURES:display(Image(filename=path))
save_json(OUTPUT/'execution_state.json',{'state':'complete','git_sha':actual,
    'run_id':RUN_SPEC['run_id'],'regions_processed':['frontal'],'figures':len(FIGURES),
    'native_clean_target_available':False,'production_qualified':False})
''')]
    return nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{
        'name':'python3','display_name':'Python 3','language':'python'}})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','submit','status'])
    parser.add_argument('--run-id',default='frontal-osf-correlations-001')
    args=parser.parse_args()
    if args.action=='status':print(cli('kernels','status',KERNEL));return
    config=json.loads((ROOT/'configs/frontal_osf.json').read_text())
    spec={'run_id':args.run_id,'campaign_id':config['campaign_id'],'kernel':KERNEL,
        'git_sha':git('rev-parse','HEAD'),'git_url':git('remote','get-url','origin'),
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'output':f"/kaggle/working/results/{config['campaign_id']}/{args.run_id}"}
    if args.action=='build':
        spec['git_sha']='RESOLVED_TO_PUBLISHED_SHA_AT_SUBMISSION'
        nbformat.write(notebook(spec),ROOT/'notebooks/Frontal_OSF_Correlations.ipynb');return
    published=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
    if spec['git_sha']!=published or git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('Commit and push before exact-SHA submission')
    folder=ROOT/'.kaggle-work'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    nbformat.write(launcher(spec,notebook(spec)),folder/'experiment.ipynb')
    metadata={'id':KERNEL,'title':'Frontal Klados and OSF Correlations',
        'code_file':'experiment.ipynb','language':'python','kernel_type':'notebook',
        'is_private':True,'enable_internet':True,'enable_gpu':True,
        'kernel_sources':config['kernel_sources'],'dataset_sources':config['dataset_sources'],
        'competition_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (folder/'run_spec.json').write_text(json.dumps(spec,indent=2))
    message=cli('kernels','push','-p',str(folder));(folder/'submission.txt').write_text(message);print(message)


if __name__=='__main__':main()
