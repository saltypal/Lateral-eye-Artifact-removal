"""Author notebook JSON without executing numerical code locally."""
import json
from pathlib import Path

STAGES = [
    ("00_Campaign_Control", "contracts", "Validate the reproducible execution environment and numerical contracts."),
    ("01_Dataset_Audit", "audit", "Verify source bytes, EEG/EOG/label semantics, identities and exposure."),
    ("02_Contracts_and_Splits", "contracts", "Validate lag signs, finite boundaries, VMD parity, masks and metrics."),
    ("03_Controlled_Corpus", "corpus", "Freeze participant/donor/recipient roles and generate coherent mixtures."),
    ("04_Frontal_VMD_Search", "vmd", "Compare K=3..10 and five alpha values; VMD must add value beyond regression."),
    ("04a_VMD_Correction_Diagnosis", "vmd-diagnosis", "Inspect solver reconstruction, projection closure, mode gates and calibration baselines."),
    ("04b_VMD_SOBI_Comparison", "vmd-sobi", "Evaluate a paper-inspired mode-domain SOBI candidate with explicit EOG/entropy selection."),
    ("05_Posterior_Experts", "posterior", "Validate GEVD-MWF and ICA with raw frontal support on unscored calibration."),
    ("06_Regional_Comparison", "regional", "Compare disjoint regional correction, shared processing and context ablations."),
    ("07_Approach_Review", "review", "Review all candidates and preservation; authorize models only on passing gates."),
    ("07a_Paper_Metric_Contracts", "paper-fixture", "Validate source-mapped equations before any formal paper evaluation."),
    ("07b_Native_Paper_Protocol", "native-protocol", "Evaluate complete native conditions and an explicitly adapted eight-second chance analysis."),
    ("08_Teacher_Crossfit", "teacher-oof", "After passing gates, generate source-isolated teacher targets."),
    ("09_EEG_Only_Student", "student-paired", "After passing gates, train one EEG-only baseline."),
    ("10_Distillation_Ablations", "student-distill", "Evaluate teacher supervision and declared loss/architecture additions."),
    ("11_Model_Freeze", "freeze", "Freeze development-selected weights and analysis before confirmation access."),
    ("12_Reserved_Evaluation", "final-eval", "Evaluate untouched sources with the frozen cleaner; native data have no paired SNR."),
    ("13_Export_and_Report", "export", "Verify fresh-process deployment and assemble reproducible research evidence.")
]

DETAILS = {
    "vmd-sobi": "Required parents: completed corpus and VMD search. Reuse archived K/alpha vectors; whiten their mode matrix and jointly diagonalize symmetric delayed covariances. Two declared delay banks, positive convergence/rank checks and failed-fit pass-through are explicit adaptations. Compare selected-source subtraction, source reference projection and approximate-entropy-plus-EOG selection. Retain the original residual and source means; all artifacts are relative to raw EEG. A 12-example pilot must pass before the 128-example mechanism comparison. The paper's SVM/GA pipeline is not claimed replicated; this experiment cannot select a teacher or authorize models. See docs/VMD_SOBI_PLAN.md.",
    "vmd-diagnosis": "Required parents: completed corpus and VMD search. Reuse immutable selected-mode vectors; do not refit or overwrite the search. Compare exact mode-plus-residual reconstruction and linear projection closure against matched direct regression. Isolate discarded mode projections, residual projection and convergence pass-through; show declared-threshold ablations without choosing a new teacher. Partition recovery MSE into squared mean bias and centered variance; centered-only SNR is never an acceptance metric. Test an explicit unscored EOG calibration baseline as an experimental alternative to zero-mean scoring projection. Clean targets and true artifact are diagnostic-only; no target enters the correction. This Fz-focused mechanism study cannot authorize a final teacher or model.",
    "native-protocol": "Required parents: completed corpus, VMD and posterior kernels plus the original source bundle. Reuse frozen development-only calibration/scoring trial IDs and held-out-fold recipes. Correct complete trials with aligned reference windows and one overlap-add subtraction. Match author-code condition concatenation for per-channel rest RMSE and ordinary Pearson, retaining signed and absolute results. Prefer verified publisher HEOG_lpf/VEOG_lpf; disclose raw fallback. Welch segments never cross trial joins. Bootstrap five distinct verified participants and eight-second rest/ocular banks for 5,000 draws; whole-montage averaging and non-overlapping bank sampling are declared adaptations, not exact paper-region replication or equivalence. See docs/NATIVE_PROTOCOL_PLAN.md. A two-record pilot must pass before the full job.",
    "paper-fixture": "Validate IVMD-SOBI RRMSE/MSE and the disclosed printed CC discrepancy, ordinary condition-specific EEG-EOG Pearson from EEGOAR-Net/Kobler, rest RMSE, Welch 2-second/1-second overlap with the paper's bands, explicit PSNR scaling and participant-level paired permutation. Read docs/EVALUATION_PROTOCOL.md for exact source locations and adaptations. SNR output, SNR improvement, PSNR and linear input RMS ratio are separate quantities. No synthetic fixture or old project-only gate can authorize models.",
    "classical-fixture":"Bounded synthetic software integration on Kaggle: generate five synthetic source buckets, execute reduced VMD/posterior searches, regional matched ablations, OSF-style proxies, legacy-style paired scoring and the review gate. The reduced fixture grid is explicitly saved and cannot replace the forty-setting production search. Synthetic pilot results must never authorize neural development. All numerical contracts run first.",
    "source-fixture":"Focused Kaggle reproduction of raw LEMON BrainVision archive loading. Verify original archive/header hashes and actual EEG/marker inventory. Resolve only a unique same-directory renamed companion, retain original header bytes and disclose the temporary loading-header changes. Never invent event annotations or modify EEG payloads. Validate one development participant before resuming the full corpus.",
    "contracts": "Validate exact lag sign/edges, signed per-channel reference association, unit-invariant VMD parity, residual retention, overlap-add, analytic GEVD operators and source reservation. Failures stop the campaign; old draft tests are not evidence for this run.",
    "audit": "Required inputs: checksummed private original bundle plus the study04 FDT supplement. Inspect original EEG/EOG/label types and global participant IDs, keep unknown Klados anatomy unknown, freeze 20% confirmation identities before windows. Development caches contain separate calibration and scoring trials. Magdeburg waveform access stays closed.",
    "corpus": "Required parent: completed audit kernel. Download raw development LEMON recordings, filter calibration/scoring independently at 200 Hz and 0.5–40 Hz. Retain low-ocular recipient references using a fixed recipient-calibration criterion. Build donor-calibrated lagged HEOG/VEOG spatial fields on exact-name montage intersections; save clean/blink/lateral/mixed conditions and -5/0/+5 dB input mixtures. Donors and recipients share a frozen crossfit bucket so either identity remains excluded together. Native OSF and legacy Klados are separate cohorts.",
    "vmd": "Required parent: completed corpus kernel. RMS-normalized VMD searches K=3..10 and alpha=250/500/1000/2000/4000 with uniform initialization, tau=0, no forced DC, tolerance 1e-6 and cap 2000. Save u_k vectors [K,1024], residual, learned center frequencies, bandwidths, overlap, convergence and runtime. Compare whole-mode removal, selective joint-EOG projected correction and direct ridge under matched lag/penalty/strength/association grids. Five modes are not assumed to be five EEG bands. Parameter selection cannot inspect confirmation sources.",
    "posterior": "Required parent: completed corpus kernel. Fit regularized GEVD-MWF ranks 1..4, instantaneous/short-lag covariance and extended Picard ICA on unscored calibration only. Compare posterior-only, raw frontal support, full montage with posterior-only correction, and signed frontal summaries. Component reference selection uses calibration EOG. Estimate artifacts from the original scoring EEG and retain PCA residuals. Insufficient calibration/convergence failures score as passthrough. SGEYESUB is reported only if its original calibration implementation is qualified.",
    "regional": "Required parents: completed corpus, VMD search and posterior search kernels. For each held-out source bucket, select recipes using the other buckets. Frontal VMD, posterior ICA/MWF and shared-channel baseline all see the same original input; subtract disjoint corrections exactly once. Compare identity, direct regression, shared VMD and no-context variants. Report source-balanced blink/lateral/mixed SNR, clean modification, alpha/beta and covariance preservation, paired correlation and HEOG/VEOG proxies. Preserve all eligible failures.",
    "review": "Required parents: completed contracts, corpus, VMD, posterior and regional kernels. Inspect full search tables, center-frequency/vector figures, source-level uncertainty, failures and computational cost. The classical gate requires >=15 dB controlled development mean SNR, <=1% worst-source mean clean modification, <=0.5 dB alpha/beta error, <=0.02 covariance error and <=0.005 paired-correlation deterioration versus direct regression. A regional VMD hybrid must show benefit over matched regression. Only a passing archived gate permits neural implementation/training under the user's conditional authorization.",
    "teacher-oof": "Requires a passing classical_gate.json. Teacher configuration selection and global fits must exclude both recipient and donor identities for each target. HEOG/VEOG remain teacher-only information; save source identities and exclusion checks for every target.",
    "student-paired": "Requires a passing classical gate and verified source partitions. Start with paired-only EEG supervision before distillation. The approved candidate is a shared width-32 temporal encoder with five dilated blocks, BiGRU 32 units per direction and zero-initialized regional residual heads. EEG masks and verified metadata permit variable channel counts. HEOG/VEOG never enter the exported model. No training begins while the classical gate is closed.",
    "student-distill": "Requires approval evidence and a validated paired baseline. Evaluate train-only cross-fitted teacher supervision, spectral/covariance preservation, optional 20 dB shortfall loss, regional/shared and bounded architecture ablations. Development selection uses grouped sources and seeds 42/3407/2026. Do not select on reserved data.",
    "freeze": "Freeze one source SHA, checkpoint hash, inference gate and analysis configuration from grouped development evidence. Record the evidence snapshot before any confirmation waveform evaluation.",
    "final-eval": "Requires the frozen model/analysis artifacts. Open reserved controlled sources and independent real EEG exactly once. Controlled SNR is reconstruction against retained recipient references; real EEG without targets receives suppression/preservation proxies. Report failures and participant-level uncertainty without retuning.",
    "export": "Verify fresh-process CPU loading and whole-record overlap-add, channel masks, missing metadata/context and measured 10/20/30/40/50-channel scaling. Package reproducible inference, source/checkpoint hashes, achieved SNR and limitations. Offline BiGRU preprocessing is not a claim of causal real-time deployment."
}


NEW_STAGES = [
    ("21_Neural_Preservation_Diagnosis", "neural-diagnosis", "Audit exact clean targets and saved neural distortion before expanding training."),
    ("14_Research_Readiness", "research-ready", "Verify restored OSF sources and all frozen development pairs."),
    ("15_Neural_Contracts", "neural-fixture", "Validate neural identity, masks, routing and optimization on Kaggle."),
    ("16_AutoVMD_Cache", "autovmd-cache", "Cache balanced all-frontal decompositions for all forty settings."),
    ("17_AutoVMD_Selection", "autovmd-search", "Nested development-only successive screening."),
    ("18_Regional_Router", "router-train", "Matched EEG-only regional and fixed-band neural experiments."),
    ("19_Paired_Deployment_Student", "student-paired", "Paired-only TCN-BiGRU deployment baseline.")
]
DETAILS.update({
    "neural-diagnosis":"Attach readiness, corpus and the completed MSE pilot outputs. Verify actual clean arrays equal their paired targets with zero added artifact, mixture closure, saved prediction identities and complete validation coverage. Compare against the exact identity control and save projection gains and correction errors. This diagnoses data and predictions; it cannot establish that a proposed loss fixes preservation.",
    "research-ready":"Attach the pinned corpus and original/restored source bundles. Verify all development source paths before trial processing, reproduce the restored study04 session and validate every paired example and source bucket. A passing readiness artifact permits paired neural research; it does not assert scientific accuracy or teacher qualification.",
    "neural-fixture":"Synthetic software contracts only: components sum to input, zero heads reproduce identity, variable channel/mode masks and permutation, no-context behavior, finite optimization and model reload. This fixture cannot establish denoising accuracy.",
    "autovmd-cache":"Attach readiness and corpus. RMS-normalized K=3..10, alpha=250/500/1000/2000/4000; retain residual and physical descriptors. Shard by explicit settings and save every numerical failure. Source-balanced search includes clean/blink/lateral/mixed and -5/0/+5dB inputs, with all verified frontal channels.",
    "autovmd-search":"Require complete source-excluded inner-fold evidence. Rank forty five-epoch candidates, eight fifteen-epoch candidates and two full-training candidates. Record rejected candidates; do not tune on outer evaluation or confirmation sources.",
    "router-train":"Width-32 shared frequency-conditioned routing with raw full-band context. Compare fixed-band, frontal-VMD/posterior-Fourier, no-context, all-VMD and raw-only arms. EEG/EOG references never enter model inference; subtract one artifact estimate from the original input. Configurations freeze across epochs.",
})


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
(OUT/'experiment.json').write_text(json.dumps(RUN_SPEC.get('experiment',{}),indent=2))
print('Exact Git SHA:', ACTUAL_SHA, 'Stage:', RUN_SPEC['stage'], 'Output:',OUT,flush=True)
'''
    execute = '''command = [str(ENV_PY),'-m','vmd_eog.runner','--stage',RUN_SPEC['stage'],
           '--input','/kaggle/input','--output',str(OUT),'--profile',RUN_SPEC['profile'],
           '--config',RUN_SPEC.get('config_path','configs/campaign.json'),
           '--experiment',str(OUT/'experiment.json')]
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
    inspect_modules = '''CONFIGURATION = json.loads((CHECKOUT/RUN_SPEC.get('config_path','configs/campaign.json')).read_text())
print('Declared signal contract:', {key: CONFIGURATION[key] for key in ('fs','window','hop','seed')})
print('Preservation limits:', CONFIGURATION['preservation'])
print('Algorithm explanation: docs/SCIENTIFIC_EXPLANATION.md in the pinned checkout')
print('Modules: signal.py -> reference.py/frontal.py/posterior.py -> experiments.py -> review evidence')
print('Git source:', ACTUAL_SHA)
'''
    tables = '''import csv
from itertools import islice
for name in ('vmd_search.csv','posterior_search.csv','review_methods.csv','condition_source_uncertainty.csv','paper_metric_fixture.csv','paper_paired_channels.csv','paper_native_channels.csv','paper_rest_bandpower_channels.csv','paper_paired_source_means.csv','paper_native_source_means.csv','paper_paired_permutation_tests.csv','paper_native_permutation_tests.csv','native_condition_channels.csv','native_condition_participants.csv','native_condition_permutations.csv','vmd_diagnosis_method_means.csv','vmd_diagnosis_clean_preservation.csv','vmd_mode_projection_diagnostics.csv','paper_plot_participant_conditions.csv','vmd_sobi_method_means.csv','vmd_sobi_clean_preservation.csv'):
    path = OUT/name
    if path.exists():
        print(name, '(first 8 saved rows; complete table remains in outputs)')
        with path.open() as handle:
            display(list(islice(csv.DictReader(handle),8)))
for name in ('selected_frontal.json','selected_posterior.json','classical_gate.json','paper_evaluation_protocol.json','paper_metric_fixture_summary.json'):
    if (OUT/name).exists():
        print(name, (OUT/name).read_text()[:8000])
review = OUT/'APPROACH_REVIEW.md'
if review.exists():
    from IPython.display import Markdown
    display(Markdown(review.read_text()))
'''
    descriptions = {item[1]: item[2] for item in STAGES+NEW_STAGES}
    detail = DETAILS[stage]
    if spec.get('campaign_id','').startswith('vmd-bandroute') and stage == 'student-paired':
        detail = 'Requires verified research readiness. Paired-only EEG TCN-BiGRU, input-derived scaling, signed metadata context, zero correction heads and no runtime EOG/VMD. Classical accuracy gates do not prevent paired learning.'
    notebook = {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
        "cells":[cell("markdown", "# Regional VMD EOG removal — "+stage+"\n\n"+descriptions.get(stage,stage)),
                 cell("markdown", "## Scientific question, inputs and contracts\n\n"+detail),
                 cell("markdown", "## Fresh source and environment\nResolve latest published code once and pin it. New temporary clones avoid stale Python imports; original inputs remain immutable."),
                 cell("code",setup),
                 cell("markdown", "## Inspect the pinned algorithm contract\nThe source modules remain inspectable in this checkout. Mode vectors, references and paired targets have distinct roles; the accompanying explanation defines their shapes and inference boundaries."),
                 cell("code",inspect_modules),
                 cell("markdown", "## Execute reusable modules\nHEOG/VEOG are teacher/evaluation information. Clean targets cannot enter inference. Failed numerical or scientific gates stop dependent work."),
                 cell("code",execute),
                 cell("markdown", "## Saved configurations and result tables\nThese are immutable outputs from this run. Readiness, teacher eligibility and final scientific qualification are distinct decisions."),
                 cell("code",tables),
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
