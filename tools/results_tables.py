"""Compile the consolidated Markdown result tables on Kaggle, without EEG processing."""
import argparse
from datetime import datetime,timezone
import json
import nbformat
from run import ROOT,BRANCH,git,cli
from frontal import launcher

KERNEL='satyapaladugu/eog-all-results-tables'


def notebook(spec):
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells=[md('''# 08 — All results: one set of Markdown tables

Verify accepted frontal, central/temporal, combined and historical parent artifacts.
Aggregate saved correlations with channel/window/record balancing. No EEG array
is loaded, no denoising/training is rerun and no predictions change. Native
EEG has no reconstruction SNR. This notebook exports ALL_RESULTS_TABLES.md.
'''),code('RUN_SPEC = '+repr(spec)+'\n'+'''import os,sys,pathlib,tempfile,subprocess,json,platform
assert pathlib.Path('/kaggle/input').exists()
print('BOOTSTRAP_START',RUN_SPEC['run_id'],RUN_SPEC['git_sha'],flush=True)
CHECKOUT=pathlib.Path(tempfile.mkdtemp(prefix='results-tables-source-'))
subprocess.run(['git','clone','--quiet',RUN_SPEC['git_url'],str(CHECKOUT)],check=True)
subprocess.run(['git','-C',str(CHECKOUT),'checkout','--quiet',RUN_SPEC['git_sha']],check=True)
actual=subprocess.check_output(['git','-C',str(CHECKOUT),'rev-parse','HEAD'],text=True).strip()
assert actual==RUN_SPEC['git_sha']
subprocess.run([sys.executable,'-m','pip','install','-q','pytest==8.3.5','tabulate==0.9.0'],check=True,timeout=120)
sys.path.insert(0,str(CHECKOUT/'src'))
from eog_clean.data import save_json
import pandas,numpy,scipy
OUTPUT=pathlib.Path(RUN_SPEC['output']);OUTPUT.mkdir(parents=True,exist_ok=True)
save_json(OUTPUT/'run_spec.json',RUN_SPEC)
save_json(OUTPUT/'environment.json',{'source_sha':actual,'python':sys.version,'platform':platform.platform(),
    'pandas':pandas.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'hardware':'Kaggle CPU'})
'''),md('## Verify aggregation, then create every table'),code('''environment=os.environ.copy();environment['PYTHONPATH']=str(CHECKOUT/'src')
test=subprocess.run([sys.executable,'-m','pytest','-q',str(CHECKOUT/'tests/test_results_tables.py'),
    '--junitxml='+str(OUTPUT/'table_tests.xml')],capture_output=True,text=True,env=environment)
(OUTPUT/'table_tests.txt').write_text(test.stdout+'\\n'+test.stderr)
print(test.stdout,flush=True)
if test.returncode:raise RuntimeError('Saved-correlation aggregation fixtures failed')
from eog_clean.results_tables import build
build('/kaggle/input',OUTPUT)
'''),md('## Consolidated result document'),code('''from IPython.display import display,Markdown
display(Markdown((OUTPUT/'ALL_RESULTS_TABLES.md').read_text()))
print((OUTPUT/'completion.json').read_text())
save_json(OUTPUT/'execution_state.json',{'state':'complete','source_sha':actual,
    'numerical_scope':'aggregation of saved metrics only','predictions_changed':False})
''')]
    return nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','submit','status'])
    parser.add_argument('--run-id',default='all-results-tables-001');args=parser.parse_args()
    if args.action=='status':print(cli('kernels','status',KERNEL));return
    spec={'run_id':args.run_id,'campaign_id':'results-tables-20261017','kernel':KERNEL,
        'git_sha':git('rev-parse','HEAD'),'git_url':git('remote','get-url','origin'),
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'output':'/kaggle/working/results/results-tables-20261017/'+args.run_id}
    if args.action=='build':
        spec['git_sha']='RESOLVED_TO_PUBLISHED_SHA_AT_SUBMISSION'
        nbformat.write(notebook(spec),ROOT/'notebooks/08_All_Results_Tables.ipynb');return
    if git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]!=spec['git_sha'] or git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('Commit/push exact source before reporting')
    folder=ROOT/'.kaggle-work'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    nbformat.write(launcher(spec,notebook(spec)),folder/'experiment.ipynb')
    metadata={'id':KERNEL,'title':'EOG All Results Tables','code_file':'experiment.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_internet':True,'enable_gpu':False,
        'dataset_sources':[],'competition_sources':[],
        'kernel_sources':['satyapaladugu/frontal-sgeyesub-regica/3','satyapaladugu/central-temporal-eog-comparison/1',
            'satyapaladugu/frontal-posterior-combined-comparison/1','satyapaladugu/klados-osf-clean-comparison/2']}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (folder/'run_spec.json').write_text(json.dumps(spec,indent=2))
    message=cli('kernels','push','-p',str(folder));(folder/'submission.txt').write_text(message);print(message)


if __name__=='__main__':main()
