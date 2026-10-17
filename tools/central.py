"""Author and launch the separate central/temporal Kaggle CPU comparison."""
import argparse
from datetime import datetime, timezone
import json
import nbformat
from run import ROOT, BRANCH, git, cli
from frontal import launcher
from frontal_pretrained import notebook as existing_template

KERNEL='satyapaladugu/central-temporal-eog-comparison'


def notebook(spec):
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    # Reuse authoring boilerplate; the emitted notebook contains the full setup
    # cell and has no dependency on another notebook's execution or variables.
    bootstrap=existing_template(spec).cells[2]
    if 'BOOTSTRAP_START' not in bootstrap.source:
        raise ValueError('Shared authoring bootstrap changed')
    bootstrap.source=bootstrap.source.replace('frontal-pretrained-source-','central-source-')
    cells=[md('''# 05 — Central and temporal EEG: separate calibration and evaluation

Central means verified FC/C/CP labels. Temporal FT/T/TP labels are reported
separately with independent fits. Klados uses paper-list row order as an
explicit assumption; paired core-central rows are C3/C4/Cz. Native OSF uses
actual named channels and separate whole-trial calibration/scoring blocks.

Compare identity, joint HEOG/VEOG ridge (penalty0.001) and GEVD–MWF (rank1),
with fixed correction strengths0.25/1. No parameter search, VMD, neural training,
frontal communication, LEMON or confirmation-data access. No posterior correction.
'''),md('## 1. Exact published source and fresh environment'),bootstrap,
    md('''## 2. Contracts and numerical experiment

All numerical work occurs here on Kaggle. Check disjoint anatomical groups,
fixed calibration, single subtraction and the existing signal/metric contracts.
Original source archives/checksums and the exact archived Klados record split
must verify. Each completed record/group saves recoverable predictions.
'''),code('''environment=os.environ.copy();environment['PYTHONPATH']=str(CHECKOUT/'src')
tests=subprocess.run([sys.executable,'-m','pytest','-q',str(CHECKOUT/'tests'),
    '--junitxml='+str(OUTPUT/'contract_tests.xml')],text=True,capture_output=True,env=environment)
(OUTPUT/'contract_tests.txt').write_text(tests.stdout+'\\n'+tests.stderr)
print(tests.stdout,flush=True)
if tests.returncode:raise RuntimeError('Contracts failed; stop before processing')
from eog_clean.central import run,report
run('/kaggle/input',CHECKOUT,OUTPUT)
FIGURES=report(OUTPUT)
'''),md('''## 3. Results, preservation and limitations

Paired SNR/RMSE/CC are Klados only. Native before/after HEOG/VEOG correlations
use unchanged references; resting preservation is a proxy without clean truth.
Absolute coefficients are averaged before participant aggregation. Failed fits
remain identity outputs and are disclosed. A preservation pass alone does not
establish suppression. Klados is historically exposed and row binding unverified.
Calibration/filter boundaries differ from the archived frontal comparison;
regional SNRs cannot be added or treated as matched superiority estimates.
'''),code('''import pandas as pd
from IPython.display import display,Image,Markdown
for name in ['klados_summary.csv','osf_summary.csv','osf_rest_summary.csv','preservation_gates.csv']:
    print(name);display(pd.read_csv(OUTPUT/name).round(6))
print((OUTPUT/'completion.json').read_text())
print((OUTPUT/'calibration_failures.json').read_text())
for path in FIGURES:display(Image(filename=path))
save_json(OUTPUT/'execution_state.json',{'state':'complete','git_sha':actual,
    'groups':['central','temporal'],'frontal_context_used':False,'production_qualified':False})
''')]
    return nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{
        'name':'python3','display_name':'Python 3','language':'python'}})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','submit','status'])
    parser.add_argument('--run-id',default='central-temporal-001');args=parser.parse_args()
    if args.action=='status':print(cli('kernels','status',KERNEL));return
    spec={'run_id':args.run_id,'campaign_id':'central-temporal-20261017','kernel':KERNEL,
        'git_sha':git('rev-parse','HEAD'),'git_url':git('remote','get-url','origin'),
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'output':'/kaggle/working/results/central-temporal-20261017/'+args.run_id}
    if args.action=='build':
        spec['git_sha']='RESOLVED_TO_PUBLISHED_SHA_AT_SUBMISSION'
        nbformat.write(notebook(spec),ROOT/'notebooks/05_Central_Temporal_Comparison.ipynb');return
    if git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]!=spec['git_sha'] or git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('Commit/push exact source before submission')
    folder=ROOT/'.kaggle-work'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    nbformat.write(launcher(spec,notebook(spec)),folder/'experiment.ipynb')
    metadata={'id':KERNEL,'title':'Central Temporal EOG Comparison','code_file':'experiment.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_internet':True,
        'enable_gpu':False,'dataset_sources':['satyapaladugu/lateral-eye-complete-dataset'],
        'kernel_sources':['satyapaladugu/klados-frontal-six-methods/5','satyapaladugu/region-aware-eog-restore/1'],
        'competition_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (folder/'run_spec.json').write_text(json.dumps(spec,indent=2))
    message=cli('kernels','push','-p',str(folder));(folder/'submission.txt').write_text(message);print(message)


if __name__=='__main__':main()
