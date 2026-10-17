"""Author, submit and monitor the six-method Kaggle comparison.

The outer notebook executes the readable experiment with nbclient and saves the
real executed notebook after each cell. No EEG computation happens locally.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import nbformat
from run import ROOT, BRANCH, git, cli

KERNEL='satyapaladugu/klados-frontal-six-methods'


def experiment(spec):
    markdown,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells=[markdown('''# Klados frontal comparison: six methods

Seven frontal channels use the paper-listed order requested by the user:
FP1, FP2, F3, F4, F7, F8 and Fz. The list-to-MATLAB-row correspondence remains
an explicit assumption. Participant identities are unavailable.

The same archived Klados windows, targets and 37/8/8 record split are used for
every method. Test records are historically development-exposed: this is a
matched development comparison, not a fresh confirmation or SOTA claim.

**Six methods:** historical EOG-assisted VMD model, SVMD–FCM adaptation,
VME–GMETV adaptation, Klados-fine-tuned EEGOAR-Net, GEVD–MWF, and author-code
BandRouteNet trained on the Klados training split. Identity, direct EOG regression
and resampling-only are additional controls. No LEMON or AutoVMD search is used.

SVMD–FCM ocular-paper datasets: synthetic, BCI Competition IV and SP300 BCI.
Its unavailable full settings prevent an exact reproduction. The VME–GMETV
ocular-paper dataset and complete matrix settings are not publicly verified.
Our explicit settings and differences are documented in `docs/PAPERS.md`.
'''),markdown('## 1. Published source and environment'),
    code('RUN_SPEC = '+repr(spec)+'\n'+'''import os,sys,pathlib,tempfile,subprocess,json,platform
from datetime import datetime,timezone
assert pathlib.Path('/kaggle/input').exists(), 'Numerical execution is Kaggle only'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
print('BOOTSTRAP_START',RUN_SPEC['run_id'],RUN_SPEC['git_sha'],flush=True)
CHECKOUT=pathlib.Path(tempfile.mkdtemp(prefix='frontal-six-source-'))
subprocess.run(['git','clone','--quiet',RUN_SPEC['git_url'],str(CHECKOUT)],check=True)
subprocess.run(['git','-C',str(CHECKOUT),'checkout','--quiet',RUN_SPEC['git_sha']],check=True)
actual=subprocess.check_output(['git','-C',str(CHECKOUT),'rev-parse','HEAD'],text=True).strip()
assert actual==RUN_SPEC['git_sha']
subprocess.run([sys.executable,'-m','pip','install','-q',
    'pymatreader==1.0.0','pytest==8.3.5','tabulate==0.9.0','torchinfo==1.8.0',
    'PySDKit==0.5.0','condat-tv==0.0.5'],check=True,timeout=300)
sys.path.insert(0,str(CHECKOUT/'src'))
import importlib.metadata,torch
torch.set_num_threads(2)
from eog_clean.data import save_json
OUTPUT=pathlib.Path(RUN_SPEC['output'])
OUTPUT.mkdir(parents=True,exist_ok=True)
save_json(OUTPUT/'run_spec.json',RUN_SPEC)
CONFIG=json.loads((CHECKOUT/'configs/frontal_six.json').read_text())
CONFIG.update(RUN_SPEC.get('config_overrides',{}))
save_json(OUTPUT/'configuration.json',CONFIG)
save_json(OUTPUT/'environment.json',{'actual_git_sha':actual,'python':sys.version,
    'platform':platform.platform(),'cuda':torch.cuda.is_available(),
    'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'packages':{name:importlib.metadata.version(name) for name in
      ['numpy','scipy','scikit-learn','pandas','matplotlib','torch','PySDKit','condat-tv','torchinfo']},
    'utc':datetime.now(timezone.utc).isoformat()})
print('SOURCE_VERIFIED',actual,'cuda',torch.cuda.is_available(),flush=True)
'''),markdown('''## 2. Contracts before training

Check the existing signal contracts plus the GMETV convexity restriction,
nonperiodic difference transpose, FCM memberships, Higuchi feature and published
solver availability. Every numerical test executes here on Kaggle.
'''),code('''environment=os.environ.copy()
environment['PYTHONPATH']=str(CHECKOUT/'src')
tests=subprocess.run([sys.executable,'-m','pytest','-q',str(CHECKOUT/'tests'),
    '--junitxml='+str(OUTPUT/'contract_tests.xml')],text=True,capture_output=True,env=environment)
(OUTPUT/'contract_tests.txt').write_text(tests.stdout+'\\n'+tests.stderr)
print(tests.stdout,flush=True)
if tests.returncode:raise RuntimeError('Contract tests failed; stop before model training')
'''),markdown('''## 3. Matched execution and recoverable artifacts

Historical VMD uses frozen K=3, alpha=2000 modes and its original metadata feature
convention; context is restricted to the seven frontal rows. Its original
all-19-row 17.927 dB score is not relabelled as a frontal score.

SVMD uses the original solver EEG-demo recipe (maxAlpha=1000, tau=0, tol=1e-6,
noise-power stopping). The declared FCM adaptation uses moving 200-sample windows,
100-sample hop, kurtosis/mean absolute amplitude/SD/Higuchi features, m=2,
two clusters, membership>=0.8 and measured mode center<=8 Hz. Its semantic cluster
rule is explicit. HEOG/VEOG do not select these modes.

VME uses compactness 20000 and initial center 0 Hz. GMETV follows the original
2020 Algorithm 3 with C=cI, gamma=0.8 and lambda=0.5 after input RMS scaling.
The ocular paper's unspecified optimized matrix is not claimed reproduced.

EEGOAR-Net uses the author architecture and pretrained initialization, 128 Hz
1-second segments, and a custom fixed seven-frontal-channel plus one absent
column layout. The released 64-label montage omits FP1/FP2 and the network mixes
channels spatially, so frozen anonymous-slot inference is invalid. Fine-tune on
Klados training records only, paired MSE, AdamW, lr0.001, weight decay0.0001,
batch8 and the same 15-epoch budget. Select on validation MSE. Input-window RMS
scales both input and target; never normalize targets separately. This is a
custom-montage fine-tuning adaptation, not unchanged pretrained inference.
Resampling-only remains a control. Original physical-unit ambiguity remains.

GEVD–MWF covariance masks use training EOG activity only (upper/lower 30%), with
rank two and full correction strength. Test targets and test calibration never
enter fitting. This is an instantaneous regularized adaptation of the MWF method.

BandRouteNet uses pinned author BRAANetV2 code, six Fourier bands, a GRU and
cross-band attention. Sampling changes to 200 Hz; explicit bands end at 100 Hz.
The main signal is already 0.5–40 Hz, so high bands carry little signal. Each
1024-sample window is divided into two 512-sample blocks. Input RMS scales both
input and target. Train with paired MSE, AdamW, lr=0.001, weight decay=0.0001,
batch eight, 15 epochs, validation-MSE checkpoint selection. Smoke runs use one
window per split and one epoch and must never be reported as final accuracy.

All methods also receive paired clean inputs for preservation testing. Failures
remain in scoring as identity fallbacks and are listed separately.
'''),code('''from eog_clean.frontal_benchmark import run
run('/kaggle/input',OUTPUT,CHECKOUT,CONFIG)
'''),markdown('''## 4. Complete result tables

Primary paper metrics: MSE, RMSE, MAE, temporal/spectral RRMSE and ordinary Pearson CC.
Spectral RRMSE follows the released BandRouteNet periodogram implementation:
RMS difference of |rFFT|²/T divided by RMS target spectrum.
Output SNR = 10 log10(sum(clean^2)/sum((clean-prediction)^2)). Improvement is
separate. Pooled window SNR is averaged within record, then across records;
mean per-channel SNR is an additional different aggregation. Report all records.

Clean modification, alpha/beta error and covariance are engineering safeguards.
Residual EEG–HEOG/VEOG correlation alone cannot establish neural preservation.
Klados windows have no verified blink/lateral labels, so no invented condition
stratification is presented.
'''),code('''import pandas as pd
from IPython.display import display
display(pd.read_csv(OUTPUT/'comparison.csv').round(6))
display(pd.read_csv(OUTPUT/'per_record.csv').round(6))
display(pd.read_csv(OUTPUT/'clean_preservation.csv').groupby(['method','record']).mean(numeric_only=True).round(6))
print('FAILURES',json.dumps(json.loads((OUTPUT/'failures.json').read_text()),indent=2))
print('BANDROUTE_SELECTION', (OUTPUT/'bandroute_selection.json').read_text())
print('DATA_CONTRACT', (OUTPUT/'data_contract.json').read_text())
'''),markdown('''## 5. Diagnostic plots

Individual-record distributions, per-channel outcomes, clean modification,
EOG associations, matched waveforms, PSD, training history, every historical
mode's H/V associations, and failures. Plots save PNG/PDF and appear below.
'''),code('''from eog_clean.frontal_figures import make_figures
from IPython.display import Image,display
FIGURES=make_figures(OUTPUT)
for path in FIGURES:display(Image(filename=path))
'''),markdown('''## 6. Interpretation and next regional experiment

Completion of execution is not production qualification. Any negative result,
failed decomposition or preservation failure remains visible. Both test reuse
and the paper-order mapping assumption limit claims. No regional dB scores add.

After selecting a frontal method from evidence, compare posterior pass-through,
GEVD–MWF and ICA component regression. Add exactly one frontal transfer ablation:
common blink context plus signed right-minus-left ocular context. Fit transfer
on calibration only. Central starts with pass-through, then conservative EOG
regression/MWF if preservation holds. All regional artifacts are estimated from
the original signal and subtracted once with disjoint masks.
'''),code('''save_json(OUTPUT/'execution_state.json',{'state':'complete','run_id':RUN_SPEC['run_id'],
    'git_sha':actual,'smoke':CONFIG.get('smoke',False),'production_qualified':False,
    'figures':len(FIGURES)})
print('NUMERICAL_NOTEBOOK_COMPLETE',RUN_SPEC['run_id'],flush=True)
''')]
    return nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{
        'name':'python3','display_name':'Python 3','language':'python'}})


def launcher(spec):
    inner=nbformat.writes(experiment(spec))
    source='RUN_SPEC = '+repr(spec)+'\nEXPERIMENT_JSON = '+repr(inner)+'\n'+'''import pathlib,json,hashlib,sys,traceback
import nbformat
from nbclient import NotebookClient
from IPython.display import display,Markdown
OUTPUT=pathlib.Path(RUN_SPEC['output'])
OUTPUT.mkdir(parents=True,exist_ok=True)
notebook=nbformat.reads(EXPERIMENT_JSON,as_version=4)
destination=OUTPUT/'Frontal_Comparison.executed.ipynb'
nbformat.write(notebook,OUTPUT/'Frontal_Comparison.source.ipynb')
def persist(**kwargs):
    nbformat.write(notebook,destination)
class StreamingClient(NotebookClient):
    def process_message(self,msg,cell,cell_index):
        if msg['msg_type']=='stream':
            print(msg['content']['text'],end='',flush=True)
        return super().process_message(msg,cell,cell_index)
complete=False
try:
    client=StreamingClient(notebook,timeout=None,kernel_name='python3',
        on_cell_executed=persist,resources={'metadata':{'path':'/kaggle/working'}})
    client.execute()
    complete=True
finally:
    nbformat.write(notebook,destination)
    payloads=[]
    for path in sorted(OUTPUT.rglob('*')):
        if path.is_file() and path.name!='artifact_manifest.json':
            payloads.append({'path':str(path.relative_to(OUTPUT)),'bytes':path.stat().st_size,
                             'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest={'run_id':RUN_SPEC['run_id'],'git_sha':RUN_SPEC['git_sha'],
              'files':payloads,'payload_count':len(payloads),'complete':complete,
              'executed_notebook_included':True,'production_qualified':False}
    (OUTPUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
for cell in notebook.cells:
    if cell.cell_type=='markdown':display(Markdown(cell.source))
    else:
        for output in cell.get('outputs',[]):
            if output.output_type in ('display_data','execute_result'):
                display(output.data,raw=True)
print('RUN_COMPLETE',RUN_SPEC['run_id'],'payloads',len(payloads),flush=True)
'''
    return nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        '# Six-method frontal comparison\nThe experiment below executes in a fresh kernel. '
        'Its real executed notebook is saved after each cell and downloaded with the artifacts.'),
        nbformat.v4.new_code_cell(source)],metadata={'kernelspec':{
        'name':'python3','display_name':'Python 3','language':'python'}})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['build','submit','status'])
    parser.add_argument('--run-id',default='frontal-six-smoke-001')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.action=='status':
        print(cli('kernels','status',KERNEL));return
    config=json.loads((ROOT/'configs/frontal_six.json').read_text())
    spec={'run_id':args.run_id,'campaign_id':config['campaign_id'],
          'git_sha':git('rev-parse','HEAD'),'git_url':git('remote','get-url','origin'),
          'kernel':KERNEL,'created_utc':datetime.now(timezone.utc).isoformat(),
          'output':f"/kaggle/working/results/{config['campaign_id']}/{args.run_id}",
          'config_overrides':{'smoke':True,'bandroute_epochs':1} if args.smoke else {}}
    if args.action=='build':
        spec['git_sha']='RESOLVED_TO_PUBLISHED_SHA_AT_SUBMISSION'
        nbformat.write(experiment(spec),ROOT/'notebooks/Frontal_Comparison.ipynb')
        print('Readable comparison notebook generated');return
    published=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
    if spec['git_sha']!=published or git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('Commit and push before exact-SHA submission')
    folder=ROOT/'.kaggle-work'/args.run_id
    folder.mkdir(parents=True,exist_ok=False)
    nbformat.write(launcher(spec),folder/'experiment.ipynb')
    metadata={'id':KERNEL,'title':'Klados Frontal Six Methods','code_file':'experiment.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_internet':True,
        'enable_gpu':True,'dataset_sources':config['dataset_sources'],
        'kernel_sources':config['kernel_sources'],'competition_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (folder/'run_spec.json').write_text(json.dumps(spec,indent=2))
    message=cli('kernels','push','-p',str(folder))
    (folder/'submission.txt').write_text(message)
    print(message)


if __name__=='__main__':main()
