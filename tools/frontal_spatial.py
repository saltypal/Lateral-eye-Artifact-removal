"""Kaggle-only author runtime check, followed by frontal spatial comparison."""
import argparse
from datetime import datetime,timezone
import json
import nbformat
from run import ROOT,BRANCH,git,cli
from frontal import launcher
from frontal_pretrained import notebook as template

KERNEL='satyapaladugu/frontal-sgeyesub-regica'


def notebook(spec):
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    bootstrap=template(spec).cells[2]
    bootstrap.source+='''
import shutil
if shutil.which('octave') is None:
    print('AUTHOR_BACKEND_INSTALL_START',flush=True)
    with (OUTPUT/'octave_install.log').open('w') as stream:
        subprocess.run(['apt-get','update','-qq'],check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        subprocess.run(['apt-get','install','-y','-qq','--no-install-recommends','octave','octave-signal'],
            check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
print(subprocess.check_output(['octave','--version'],text=True).splitlines()[0],flush=True)
'''
    cells=[md('''# 06 — Frontal SGEYESUB and REG-ICA: full-montage spatial fit

Use full verified EEG montage for the author spatial methods, then score only
the five exact frontal electrodes F3/F4/F7/F8/Fz on native OSF. Frontal channels
can reveal ocular spatial patterns more strongly; other electrodes supply
spatial information. This rationale is not evidence of improved preservation.
No central/posterior results or explicit message passing in this experiment.
VMD stays frontal-only. Calibration/scoring trials are disjoint.

SGEYESUB uses original six-class sample annotations in calibration only.
REG-ICA uses original extended Infomax, fractal component selector and SRLS
regression through Octave. Freezing ICA/selection on calibration and resetting
SRLS per trial is an explicit evaluation adaptation. The two boundary samples
are retained rather than deleted by the original filter's zero initialization.
No neural training, AutoVMD search, LEMON or confirmation access.
The full stage also evaluates REG-ICA on the exact paired Klados windows and
seven frontal rows of the completed benchmark. All19 EEG rows fit its ICA;
SGEYESUB is ineligible on Klados without its six-class calibration labels.
Recovery verifies failed version2 payload hashes and calibration inputs before
reuse. One-float32-ULP filter-rounding differences are logged; actual scored
frontal rows use exact archived bytes. Larger discrepancies fail. Scientific
acceptance limits are unchanged.
'''),md('## 1. Exact source, numerical dependencies and original-code runtime'),bootstrap,
    md('''## 2. Numerical contracts, then declared execution stage

Readiness runs full-montage SGEYESUB on one real OSF calibration plus a small
REG-ICA numerical fixture. The full stage starts only after that succeeds.
Full evaluation matches native scoring input/reference arrays exactly to the
completed frontal parent. No guessed Klados labels for SGEYESUB.
'''),code('''environment=os.environ.copy();environment['PYTHONPATH']=str(CHECKOUT/'src')
tests=subprocess.run([sys.executable,'-m','pytest','-q',str(CHECKOUT/'tests'),
    '--junitxml='+str(OUTPUT/'contract_tests.xml')],text=True,capture_output=True,env=environment)
(OUTPUT/'contract_tests.txt').write_text(tests.stdout+'\\n'+tests.stderr)
print(tests.stdout,flush=True)
if tests.returncode:raise RuntimeError('Contracts failed')
from eog_clean.frontal_spatial import readiness,run_native
if RUN_SPEC['stage']=='readiness':readiness('/kaggle/input',CHECKOUT,OUTPUT)
else:run_native('/kaggle/input',CHECKOUT,OUTPUT)
'''),md('''## 3. Saved evidence and interpretation

Readiness is execution evidence, not model selection. Native OSF has no paired
clean EEG, so it has no reconstruction SNR. All failures are retained. A low
ocular correlation cannot qualify a method without preservation checks.
Existing neural and VMD comparators retain their previously disclosed access
and training conditions; matching inputs does not erase those differences.
'''),code('''import numpy as np,pandas as pd,matplotlib.pyplot as plt
from IPython.display import display,Image
FIGURES=[]
if RUN_SPEC['stage']=='readiness':
    print((OUTPUT/'readiness.json').read_text())
    print((OUTPUT/'native_calibration_eligibility.json').read_text())
    with np.load(OUTPUT/'native_readiness_prediction.npz',allow_pickle=False) as data:
        figure,axis=plt.subplots(figsize=(12,4));t=np.arange(min(1000,data['raw'].shape[-1]))/200
        axis.plot(t,data['raw'][0,:len(t)],label='Original F3')
        axis.plot(t,data['sgeyesub'][0,:len(t)],label='Author SGEYESUB')
        axis.set(xlabel='Time (s)',ylabel='Source amplitude units',title='Runtime check only: one frontal trial, not aggregate accuracy')
        axis.legend()
    display(pd.DataFrame([{'method':'SGEYESUB','runtime_check':'real full-montage calibration'},
                          {'method':'REG-ICA','runtime_check':'author ICA/FD/SRLS fixture'}]))
else:
    from eog_clean.spatial_report import report
    FIGURES=report(OUTPUT,'/kaggle/input')
    if (OUTPUT/'klados_comparison.csv').is_file():
        display(pd.read_csv(OUTPUT/'klados_comparison.csv').round(6))
    table=pd.read_csv(OUTPUT/'osf_comparison.csv');rest=pd.read_csv(OUTPUT/'osf_rest_summary.csv')
    display(table.round(6));display(rest.round(6))
    display(pd.read_csv(OUTPUT/'osf_rest_worst_participant.csv').round(6))
    print((OUTPUT/'method_failures.json').read_text())
    figure,axes=plt.subplots(1,3,figsize=(18,6),sharey=True)
    methods=sorted(table.method.unique())
    blink=table[table.condition=='blink'].set_index('method').loc[methods]
    lateral=table[table.condition=='lateral'].set_index('method').loc[methods]
    resting=rest.set_index('method').loc[methods]
    for axis,values,title in zip(axes,[blink.veog_abs_cc_after,lateral.heog_abs_cc_after,100*resting.relative_change],
        ['Blink abs VEOG CC','Lateral abs HEOG CC','Rest relative RMS modification (%)']):
        axis.barh(methods,values);axis.set_title(title);axis.grid(axis='x',alpha=.2)
    figure.suptitle('Frontal outputs: full-montage SGEYESUB/REG-ICA and existing frontal comparators')
figure.tight_layout();path=OUTPUT/'frontal_spatial_review.png'
figure.savefig(path,dpi=150,bbox_inches='tight');figure.savefig(OUTPUT/'frontal_spatial_review.pdf',bbox_inches='tight')
plt.close(figure);display(Image(filename=str(path)))
for figure_path in FIGURES:display(Image(filename=figure_path))
if RUN_SPEC['stage']!='readiness':
    display(pd.read_csv(OUTPUT/'preservation_review.csv'))
    display(pd.read_csv(OUTPUT/'paired_cluster_uncertainty.csv').round(6))
save_json(OUTPUT/'execution_state.json',{'state':'complete','git_sha':actual,
    'stage':RUN_SPEC['stage'],'scored_regions':['frontal'],'production_qualified':False})
''')]
    return nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{
        'name':'python3','display_name':'Python 3','language':'python'}})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','submit','status'])
    parser.add_argument('--run-id',default='frontal-spatial-readiness-001')
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--resume-failed-parent',action='store_true');args=parser.parse_args()
    if args.resume_failed_parent and not args.full:raise ValueError('Recovery applies only to the full stage')
    if args.action=='status':print(cli('kernels','status',KERNEL));return
    spec={'run_id':args.run_id,'campaign_id':'frontal-spatial-20261017','kernel':KERNEL,
        'git_sha':git('rev-parse','HEAD'),'git_url':git('remote','get-url','origin'),
        'created_utc':datetime.now(timezone.utc).isoformat(),'stage':'full' if args.full else 'readiness',
        'recovery_parent':'frontal-spatial-full-001/e8d0021' if args.resume_failed_parent else None,
        'output':'/kaggle/working/results/frontal-spatial-20261017/'+args.run_id}
    if args.action=='build':
        spec['git_sha']='RESOLVED_TO_PUBLISHED_SHA_AT_SUBMISSION'
        nbformat.write(notebook(spec),ROOT/'notebooks/06_Frontal_Spatial_Comparison.ipynb');return
    if git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]!=spec['git_sha'] or git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('Commit/push before exact-SHA execution')
    folder=ROOT/'.kaggle-work'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    nbformat.write(launcher(spec,notebook(spec)),folder/'experiment.ipynb')
    metadata={'id':KERNEL,'title':'Frontal SGEYESUB REGICA','code_file':'experiment.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_internet':True,'enable_gpu':False,
        'dataset_sources':['satyapaladugu/lateral-eye-complete-dataset'],
        'kernel_sources':['satyapaladugu/region-aware-eog-restore/1',
            'satyapaladugu/frontal-klados-and-osf-correlations/1',
            'satyapaladugu/klados-frontal-six-methods/5',
            'satyapaladugu/region-aware-eog-snr-context/1'],'competition_sources':[]}
    if args.resume_failed_parent:metadata['kernel_sources'].append(KERNEL+'/2')
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (folder/'run_spec.json').write_text(json.dumps(spec,indent=2))
    message=cli('kernels','push','-p',str(folder));(folder/'submission.txt').write_text(message);print(message)


if __name__=='__main__':main()
