"""Small Kaggle CPU audit; reuse saved modes, never repeat decomposition/training."""
import argparse
from datetime import datetime,timezone
import json
import nbformat
from run import ROOT,BRANCH,git,cli
from frontal import launcher

KERNEL='satyapaladugu/frontal-mode-clean-eeg-audit'


def notebook(spec):
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells=[md('''# Every frontal mode: clean EEG, HEOG and VEOG

Use the completed Klados six-method experiment's actual saved decompositions.
Check each historical VMD, SVMD and VME mode against its paired clean EEG channel
and both ocular references. Audit retained residuals separately. No new training,
decomposition, K/alpha search, posterior processing or clean-target mode selection.
'''),code('RUN_SPEC = '+repr(spec)+'\n'+'''import os,sys,pathlib,tempfile,subprocess,json
assert pathlib.Path('/kaggle/input').exists()
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
print('BOOTSTRAP_START',RUN_SPEC['run_id'],RUN_SPEC['git_sha'],flush=True)
CHECKOUT=pathlib.Path(tempfile.mkdtemp(prefix='frontal-mode-audit-source-'))
subprocess.run(['git','clone','--quiet',RUN_SPEC['git_url'],str(CHECKOUT)],check=True)
subprocess.run(['git','-C',str(CHECKOUT),'checkout','--quiet',RUN_SPEC['git_sha']],check=True)
actual=subprocess.check_output(['git','-C',str(CHECKOUT),'rev-parse','HEAD'],text=True).strip()
assert actual==RUN_SPEC['git_sha']
subprocess.run([sys.executable,'-m','pip','install','-q','pytest==8.3.5','pymatreader==1.0.0',
    'tabulate==0.9.0','torchinfo==1.8.0','PySDKit==0.5.0','condat-tv==0.0.5'],check=True,timeout=300)
sys.path.insert(0,str(CHECKOUT/'src'))
from eog_clean.data import save_json
import torch,importlib.metadata
torch.set_num_threads(2)
OUTPUT=pathlib.Path(RUN_SPEC['output']);OUTPUT.mkdir(parents=True,exist_ok=True)
save_json(OUTPUT/'run_spec.json',RUN_SPEC)
save_json(OUTPUT/'environment.json',{'git_sha':actual,'python':sys.version,'cuda':torch.cuda.is_available(),
    'versions':{name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','torch','matplotlib']}})
'''),md('''## Contracts and execution

A known orthogonal EEG/HEOG/VEOG fixture checks the three correlation columns.
The full parent artifact checksums and exact source identity must verify.
'''),code('''environment=os.environ.copy();environment['PYTHONPATH']=str(CHECKOUT/'src')
result=subprocess.run([sys.executable,'-m','pytest','-q',str(CHECKOUT/'tests'),
    '--junitxml='+str(OUTPUT/'contract_tests.xml')],text=True,capture_output=True,env=environment)
(OUTPUT/'contract_tests.txt').write_text(result.stdout+'\\n'+result.stderr)
print(result.stdout,flush=True)
if result.returncode:raise RuntimeError('Numerical contracts failed')
from eog_clean.mode_audit import run
FIGURES=run('/kaggle/input',CHECKOUT,OUTPUT)
'''),md('''## Complete numerical evidence

Mode indices are not fixed EEG frequency bins. Inspect measured center frequency
and actual K. High ocular and high clean association can coexist. This is a
diagnostic of paired Klados, not an oracle selector available in native OSF.
'''),code('''import pandas as pd
from IPython.display import display,Image
display(pd.read_csv(OUTPUT/'frontal_mode_correlation_summary.csv').round(6))
display(pd.read_csv(OUTPUT/'native_frontal_before_after_summary.csv').round(6))
display(pd.read_csv(OUTPUT/'native_rest_participant_summary.csv').round(6))
print((OUTPUT/'audit_completion.json').read_text())
for path in FIGURES:display(Image(filename=path))
save_json(OUTPUT/'execution_state.json',{'state':'complete','git_sha':actual,
    'regions_processed':['frontal'],'clean_targets_used_for_runtime_selection':False})
''')]
    return nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{
        'name':'python3','display_name':'Python 3','language':'python'}})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','submit','status'])
    parser.add_argument('--run-id',default='frontal-mode-clean-audit-001');args=parser.parse_args()
    if args.action=='status':print(cli('kernels','status',KERNEL));return
    spec={'run_id':args.run_id,'campaign_id':'frontal-mode-audit-20261017','kernel':KERNEL,
        'git_sha':git('rev-parse','HEAD'),'git_url':git('remote','get-url','origin'),
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'output':'/kaggle/working/results/frontal-mode-audit-20261017/'+args.run_id}
    if args.action=='build':
        spec['git_sha']='RESOLVED_TO_PUBLISHED_SHA_AT_SUBMISSION'
        nbformat.write(notebook(spec),ROOT/'notebooks/Frontal_Mode_Clean_Correlations.ipynb');return
    if git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]!=spec['git_sha'] or git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('Commit and push before exact-SHA submission')
    folder=ROOT/'.kaggle-work'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    nbformat.write(launcher(spec,notebook(spec)),folder/'experiment.ipynb')
    metadata={'id':KERNEL,'title':'Frontal Mode Clean EEG Audit','code_file':'experiment.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_internet':True,
        'enable_gpu':False,'dataset_sources':[],'kernel_sources':['satyapaladugu/klados-frontal-six-methods/5',
            'satyapaladugu/frontal-klados-and-osf-correlations/1'],
        'competition_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (folder/'run_spec.json').write_text(json.dumps(spec,indent=2))
    message=cli('kernels','push','-p',str(folder));(folder/'submission.txt').write_text(message);print(message)


if __name__=='__main__':main()
