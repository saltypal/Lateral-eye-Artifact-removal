"""Launch the separately versioned matched frontal/posterior comparison."""
import argparse
from datetime import datetime,timezone
import json
import nbformat
from run import ROOT,BRANCH,git,cli
from frontal import launcher
from frontal_pretrained import notebook as template

KERNEL='satyapaladugu/frontal-posterior-combined-comparison'


def notebook(spec):
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells=[md('''# 07 — Combined frontal/posterior comparison: whole-cap SNR

Compare identity, historical frontal VMD+EOG, and calibrated frontal ridge,
crossed with posterior identity, calibrated ridge and rank1 GEVD-MWF. These
are nine fixed structural ablations, not AutoVMD or a parameter search.

Klados frontal: FP1/FP2/F3/F4/F7/F8/Fz; posterior: P3/P4/O1/O2/Pz.
Paper-list row binding remains an explicit unverified assumption. Central
C3/C4/Cz and temporal T3/T4/T5/T6 remain exactly unchanged. Full19-channel
raw/clean scoring arrays and historical frontal outputs are verified archives.

Each branch estimates correction relative to original EEG. Disjoint masks and
one subtraction form the combined output. Whole-cap SNR is measured from
that output and the clean19-channel target; regional dB are never added.
This stage is composition, not frontal-to-back message passing. No new VMD
decomposition or neural training. No LEMON or confirmation sources.

SGEYESUB lacks verified six-class Klados calibration, so no guessed paired
SGEYESUB score is produced. Native OSF has no clean target/reconstruction SNR.
'''),md('## 1. Fresh exact-SHA source and environment'),template(spec).cells[2],
    md('''## 2. Contracts, calibration and combined experiment

Run mask/identity/pooled-SNR fixtures before processing. Fit ridge0.001 and
posterior rank1 MWF at strength1 using the first2000 samples, independently
preprocessed. MWF uses bottom/top30% calibration EOG activity; these are
low/high ocular proxies, not known clean labels. No target is used to fit.
Archived historical frontal predictions and clean-input controls remain fixed.
'''),code('''environment=os.environ.copy();environment['PYTHONPATH']=str(CHECKOUT/'src')
tests=subprocess.run([sys.executable,'-m','pytest','-q',str(CHECKOUT/'tests'),
    '--junitxml='+str(OUTPUT/'contract_tests.xml')],text=True,capture_output=True,env=environment)
(OUTPUT/'contract_tests.txt').write_text(tests.stdout+'\\n'+tests.stderr)
print(tests.stdout,flush=True)
if tests.returncode:raise RuntimeError('Numerical contracts failed')
from eog_clean.regional_comparison import run,report
run('/kaggle/input',CHECKOUT,OUTPUT)
FIGURES=report(OUTPUT)
'''),md('''## 3. Results and qualification limits

Report frontal/posterior/whole-cap SNR and clean CC separately. Include RMSE,
temporal/spectral RRMSE, channel H/V associations, clean-input controls,
record-balanced summaries and paired record-cluster uncertainty. The eight
reused record identities are not eight verified independent participants.
Central and temporal pass-through may limit whole-cap performance; they are
not silently denoised. No final winner or deployment claim follows from SNR.
'''),code('''import pandas as pd
from IPython.display import display,Image
table=pd.read_csv(OUTPUT/'comparison.csv')
display(table[table.group.isin(['frontal','posterior','whole_cap'])].round(6))
display(pd.read_csv(OUTPUT/'preservation_gates.csv').round(6))
display(pd.read_csv(OUTPUT/'combined_snr_uncertainty.csv').round(6))
print((OUTPUT/'method_failures.json').read_text())
print((OUTPUT/'completion.json').read_text())
for path in FIGURES:display(Image(filename=path))
save_json(OUTPUT/'execution_state.json',{'state':'complete','git_sha':actual,
    'regions_corrected':['frontal','posterior'],'central_temporal_unchanged':True,
    'regional_dB_added':False,'production_qualified':False})
''')]
    return nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','submit','status'])
    parser.add_argument('--run-id',default='frontal-posterior-combined-001');args=parser.parse_args()
    if args.action=='status':print(cli('kernels','status',KERNEL));return
    spec={'run_id':args.run_id,'campaign_id':'frontal-posterior-20261017','kernel':KERNEL,
        'git_sha':git('rev-parse','HEAD'),'git_url':git('remote','get-url','origin'),
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'output':'/kaggle/working/results/frontal-posterior-20261017/'+args.run_id}
    if args.action=='build':
        spec['git_sha']='RESOLVED_TO_PUBLISHED_SHA_AT_SUBMISSION'
        nbformat.write(notebook(spec),ROOT/'notebooks/07_Frontal_Posterior_Combined.ipynb');return
    if git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]!=spec['git_sha'] or git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('Commit/push exact source before execution')
    folder=ROOT/'.kaggle-work'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    nbformat.write(launcher(spec,notebook(spec)),folder/'experiment.ipynb')
    metadata={'id':KERNEL,'title':'Frontal Posterior Combined Comparison','code_file':'experiment.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_internet':True,'enable_gpu':False,
        'dataset_sources':['satyapaladugu/lateral-eye-complete-dataset'],
        'kernel_sources':['satyapaladugu/region-aware-eog-restore/1','satyapaladugu/klados-frontal-six-methods/5',
                          'satyapaladugu/region-aware-eog-snr-context/1'],'competition_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (folder/'run_spec.json').write_text(json.dumps(spec,indent=2))
    message=cli('kernels','push','-p',str(folder));(folder/'submission.txt').write_text(message);print(message)


if __name__=='__main__':main()
