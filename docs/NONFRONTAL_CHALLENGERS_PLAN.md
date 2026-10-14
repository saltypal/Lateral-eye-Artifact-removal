# Posterior and central correction challengers

14 October 2026 IST. User-authorized expansion of exploratory regional experiments.
This permits experiments despite the current frontal/native qualification failure;
it does not promote a teacher, open confirmation sources or authorize neural training.

## Question and frozen context

Can selectively correcting ocular spatial components improve posterior and central
reconstruction over the existing regional hybrid? Preserve frontal K3/alpha2000
full-field predictions (17.112814 dB frontal development mean). No posterior or
central VMD, no AutoVMD and no repetition of the 737-candidate posterior grid.
Full archived central mean was10.903434 dB and posterior13.541703 dB; whole
VMD-context13.162704 dB. Regional dB scores cannot be added.

## Evidence influencing design

- Issa and Juhasz2019: wavelet correction within ocular ICA components and
  contaminated intervals rather than deleting entire components. Primary paper:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6956025/ . Our fixed sym4/MAD thresholds,
  calibration-only Picard fit and EOG-assisted component selection are adaptations;
  no exact author replication or EEG-only inference claim.
- Mannan, Jeong and Kamran2016: process ocular ICs using regression, retaining
  their neural contribution. Primary paper:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4853904/ . Our lagged ridge with frozen
  calibration coefficients replaces their entropy/MAD/extended-RLS pipeline;
  label it component-regression adaptation, not REG-ICA reproduction.
- Kobler et al2020 SGEYESUB: calibration-trained ocular subspace subtraction
  addresses dipole and eyelid activity and protects resting/ERP signals.
  https://pubmed.ncbi.nlm.nih.gov/32497788/ ; author implementation
  https://github.com/rkobler/eyeartifactcorrection . Remains a later challenger
  requiring its actual calibration paradigm/contracts and author-code parity.
  Author source inspection confirms six sample classes: left/right movement1/2,
  vertical movement3/4, blink5, rest6. The existing four trial-type codes are a
  different taxonomy; do not substitute them or fabricate directional labels
  from HEOG signs. Audit the original artifactclasses channel/calibration labels
  before porting elastic-net classifiers and the staged eye/blink matrices.
- GED identifies contrasts rather than biological truth. Reference-informed
  low-rank transfer is our project extension, not SGEYESUB or the original MWF:
  https://doi.org/10.1016/j.neuroimage.2021.118809 . Measure preservation separately.

## First experiment: six fixed mechanisms

Fit one full-available-montage ICA on each unscored calibration; output only
verified posterior or central/shared rows. Full support includes frontal channels
but never decomposes them with new VMD. Match support and association gate across
ICA correction arms. Use original EEG for every artifact estimate.

1. ICA whole selected-component subtraction: matched rejection control.
2. ICA component ridge: joint HEOG/VEOG lagged calibration regression of selected
   sources, back-projected to a complete artifact field; infer one bounded shared
   gain per output region from scoring input. Targets are absent from inference.
3. Wavelet ICA: subtract large-coefficient contribution from selected sources.
4. Selective wavelet ICA: same contribution restricted to source-amplitude events
   using calibration-rest MAD, dilation and taper. Sustained lateral events are
   evaluated explicitly; a blink-shaped detector is not assumed adequate.
5. Reference-field rank1 transfer: rest-whitened rank reduction of the calibration
   EOG-predicted field; scoring EEG determines bounded region gain.
6. Reference-field rank2 transfer: same parameters, second-rank ablation.

Each mechanism saves three compositions: posterior-only change, central-only
change and both changed. Frozen frontal output and the untouched other branch
remain exact. Baselines: identity, archived VMD-context hybrid and archived
raw-context hybrid (same frontal VMD output). No scoring-target recipe selection.

Information-access distinction: scoring ICA/wavelet arms use original full-montage
EEG, including raw frontal support. Reference-rank arms use full-montage calibration
to learn the field but scoring HEOG/VEOG plus each output region's EEG determine
correction; they do not consume the corrected frontal waveform or its estimated
ocular field. Combining these outputs with frozen frontal VMD is a regional
hybrid, not evidence of explicit VMD-to-posterior communication. After the fixed
comparison, NF09 separately tests no-context versus signed frontal summaries and
the estimated frontal ocular field, using calibration-only transfer fitting and
source-quality flags. Match raw and VMD-derived context; do not assume VMD helps.

Fixed settings: ICA association0.4; calibration ridge lags[-10,0,10], penalty0.01;
gain0..8; wavelet sym4, maximum level6 subject to valid length; rest-MAD multiplier3;
source event dilation50samples and taper10samples at200Hz. These are declared
project settings, not published optimal parameters. New recipes are experimental.

## Tasks and acceptance

- NF01 Implement target-free estimation interfaces, disjoint composition and
  focused tests for shape/finite errors, boundaries, zero fields, exact pass-through,
  ICA source preservation, wavelet decomposition reconstruction and rank limits.
- NF02 Pin corpus002 and context-match-full001 manifests; verify every consumed
  array, split, calibration reference and metadata; reject reserved examples.
- NF03 Kaggle five-recipient/50-example pilot with all nine contaminated strata,
  clean controls and all methods. Save predictions, fits, progress and failures
  per example. Pilot metrics cannot establish a winner.
- NF04 After contracts/recovery pass, run same fixed recipes on320 examples/
  32recipients; save per-channel/record/region/condition/input paper metrics and
  source-aware uncertainty. No broad parameter search or pilot-driven recipe edits.
- NF05 Choose a shortlist only on grouped development evidence. Refit/selection
  evaluation excludes recipient and donor identities. Fixed-recipe exploratory
  comparisons using own unscored calibration are distinct from held-out selection.
- NF06 Native OSF ocular/rest validation of frozen shortlisted methods; unchanged
  references, no reconstruction SNR. Previous native failures remain recorded.
- NF07 Eligible author SGEYESUB, selective wavelet replication and ERP/BCI tests
  follow their source/calibration contracts; no guessed anatomy or head model.
- NF10 Frozen-output review: compare each shortlisted nonfrontal correction with
  archived VMD versus raw-regression frontal outputs, without new fitting. Report
  both declared channel-first mean SNR and per-window pooled-energy SNR. Split
  verified middle FC/C/CP channels from temporal shared channels for additional
  diagnostics while preserving the original primary regional bucket. No neural
  or teacher promotion. Original-input artifact composition, baseline metric
  parity, exact parent identities and failure counts remain required.

Full-development result motivates the next hypothesis: reference-rank2 raises
whole channel-first mean from13.162704 to15.155692dB, but blink+5 output falls
from19.821757 to17.562110 and mixed+5 from18.501700 to17.328050. Severe lateral
-5 remains8.050232. Diagnose weak-input overcorrection and severe-lateral residuals
before a calibration-fitted confidence/shrinkage blend with the old branch.
Such a blend is a proposed correction experiment, not a selected recipe or an
excuse to use true input-SNR labels at inference. No broad K/alpha search.

Parallel GEVD-MWF control: numbered notebook59 uses the same corpus/context
parents and reporting engine with a separate campaign. Exactly two recipes,
rank1/rank2, instantaneous full-montage support, strength1, same0.4association
gate. Calibration rest/ocular covariances use the already validated posterior
solver. No repetition of737candidate search. Saved-fit replay and baseline
metric parity are mandatory, as in notebook58; all corrections remain relative
to original input, with frontal outputs frozen. First run is50examples, not a
new teacher selection. This control is authorized alongside the320example
six-mechanism run after its pilot contracts/recovery passed.

## Reporting and stop rules

## Next mechanisms after the fixed pilot

| Priority | Mechanism and region | What changes / eligibility |
| --- | --- | --- |
| 1 | Existing GEVD-MWF on central/shared outputs | Reuse validated posterior.py; fixed rank1 and rank2 controls, same full/raw-frontal support, calibration rest/ocular labels and association gate. This tests central use without repeating the posterior search. |
| 2 | SGEYESUB on posterior/central outputs | Audit author demo and native OSF calibration labels/reference/channel layout; port only with author parity. Matrix subtraction after fitting is attractive for scalable devices. Do not call our reference-rank method SGEYESUB. |
| 3 | Reference CCA/iCanClean adaptation | Separate correlated artifact subspace before back-projection; HEOG/VEOG and signed frontal field as separately matched reference options. Phantom evidence is not independent human ocular validation. |
| 4 | SOBI plus selective source regression | Temporal source statistics provide a different separation from Picard. Existing SOBI code was written for VMD vectors; multichannel calibration needs numerical-rank and trial-boundary repairs before reuse. No posterior VMD. |
| 5 | ASR/rASR benchmark control | Requires eligible low-ocular calibration, common preprocessing and source-aware cutoff selection. Test reconstruction rather than rejecting samples; score every original interval. Variance selection may remove task-related neural activity and is not ocular-specific. |
| 6 | Wavelet-enhanced CCA | A published left/right common-component idea is a challenger only. Common bilateral activity can be neural and signed lateral eye activity may not be common; require reference-supported identification instead of assuming first component is ocular. |

Primary references for this extension, checked14October2026:

- iCanClean2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC10574843/ . CCA identifies
  reference-correlated components, followed by regression/back-projection; the
  cited experiment is phantomEEG and must be reported in that scope.
- SOBINMF2015: https://doi.org/10.1541/ieejeiss.135.954 . SOBI/NMF evaluated with
  motor-imagery EEG; adding nonnegative factorization to signed EEG requires its
  actual transform/objective, not simply clipping negative electrode values.
- ASR/rASR2019: https://pmc.ncbi.nlm.nih.gov/articles/PMC6499032/ . Ocular amplitude
  reduction and preservation trade-offs; not evidence that ASR will beat our SNR.
- wCCA2011: https://pubmed.ncbi.nlm.nih.gov/22255263/ . Inspect complete settings
  before any reproduction claim; assumptions need lateral-specific validation.
- CiSSA-DWT2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC9921497/ . Originally
  discusses few/single frontal channels; postpone a central/posterior adaptation
  until it adds a testable hypothesis beyond the running wavelet component arms.

The central/shared label must be unpacked into verified channels in reporting;
it is a montage-defined collection rather than a biological claim that all its
electrodes have the same ocular coupling. Preserve mu/alpha and beta activity at
central electrodes and posterior alpha; ERP/BCI task preservation needs separate
eligible labelled recordings. Unknown anatomy still passes through.

Archived metadata inspection confirms the current central/shared bucket is:
C1-C6, Cz, CP1-CP6, CPz, FC1-FC6, FT7/FT8, T7/T8 and TP7/TP8. Thus its26rows
include20frontocentral/central/centroparietal rows and6temporal rows. Retain this
frozen primary bucket for matched comparisons, then report the two verified
subgroups separately on Kaggle before assigning different subgroup policies.
Names identify the audited montage; no electrode-coordinate inference is used.

EMA-ASR (28September2026preprint) is a later exploratory ASR variant:
https://arxiv.org/abs/2609.35223 . Its10subject/24channel study reports improved
blink attenuation with increased reconstruction/spectral deviation. This is a
preservation tradeoff, not evidence of15/20dB paired reconstruction or lateral
superiority. Inspect author code and calibration drift safeguards before a run.

Primary paired MSE, temporal RRMSE, ordinary Pearson CC; output SNR and improvement
separate. Also report spectral and covariance preservation, runtime, fit failures
and pass-through. Gates unchanged: clean modification<=.01, alpha/beta<=.5dB,
covariance<=.02, CC deterioration<=.005. Evaluate comparator-specific deterioration
against both frozen matched context baselines and label comparator identity.
Controlled artifact fields are regression-generated and may favor regression;
EOG amplitude coupling is still imperfect. These are development experiments,
not independent validation or production qualification. Native tests required.

Retain all failures. No method wins because it includes VMD. Best whole-system
SNR is assessed from composed waveforms, never summed regional dB. If the fixed
challengers fail, report their shortfall and use the existing frozen baseline.


## Executable follow-up order after full320 comparison

The completed comparison rejects the tested MWF, whole-IC rejection and wavelet
recipes as next production candidates. Keep them as controls; do not repeat a
large blind grid. Frontal K3/2000 outputs remain fixed during these nonfrontal tests.

1. **NF10 frozen frontend and aggregation audit — launched.** Exact full001/1 and
   context-match-full001/1 outputs only. Compare VMD/ridge frontal with identical
   rank2 or ICA-component-ridge middle/posterior outputs. Report pooled-energy and
   channel-first SNR separately; do not compare the historical17.9272dB as if its
   cohort, EOG access and aggregation were identical. Report20verified middle and
   6temporal-channel subgroups. Success means complete parity/provenance, not a
   desired winner. No new fit or closed-source access.
2. **NF11 weak-artifact conservative gain — proposed.** Audit saved association,
   gain and artifact/EEG-energy ratios per condition first. The current method
   estimates its map on calibration, but region gain from observed scoring EEG;
   it is not a calibration-only gain. Neural contribution to this estimate is
   a plausible overcorrection mechanism, not an established root cause. Compare
   the existing gain with one calibration-constrained gain/shrinkage variant.
   Use the existing declared strength bank0.25/0.5/0.75/1 rather than inventing a
   broad search. If blending old and rank2 fields, use a=(1-w)*a_old+w*a_rank2 on
   disjoint nonfrontal rows, then subtract once from original. The weight must be
   inferred from observable EEG/EOG and training/calibration, never true input
   SNR, clean target, condition labels or scoring artifact. Grouped selection
   excludes both recipients and donors. Reject if weak blink/mixed loss persists
   or clean/CC gates fail; retain severe lateral and every original interval.
3. **NF12 reference CCA challenger — proposed.** Reuse verified calibration,
   references and rest covariance. CCA separates EEG/reference-correlated latent
   directions; estimate their sensor projection on calibration, then correct only
   reference-supported portions. Compare HEOG/VEOG alone against the same EOG
   plus separate signed frontal artifact context. Raw-frontal and no-context
   controls must match information/budget. Use rank1/2 controls first. Require
   reconstruction, numerical-rank, permutation and trial-boundary fixtures before
   the50example pilot. Original iCanClean validation here is phantom data; ours
   would be a labelled human-ocular adaptation, not a reproduced published score.
4. **NF07 author SGEYESUB parity — eligibility before implementation.** Original
   sample labels1/2left/right,3/4up/down,5blink,6rest are distinct from our four
   trial codes. Read existing ARTIFACTCLASSES channel on Kaggle; do not create
   left/right labels from HEOG signs. Verify native channel/reference/filter and
   actual calibration duration per recording. Match author MATLAB/Octave outputs
   on deterministic fixtures before a Python method gets the published name.
   Insufficient labels/calibration or unavailable parity blocks this arm, not the
   whole campaign. It may consume raw full-montage calibration but outputs only
   middle/posterior correction here; no nonfrontal VMD.
5. **NF13 temporal-separation control — conditional.** Multichannel SOBI plus
   selective reference regression provides a separation assumption different from
   ICA. Repair rank/trial-boundary contracts before porting the VMD-domain helper;
   do not decompose posterior signals with VMD. One fixed delay bank and matched
   calibration first, not another unconstrained grid. ASR/rASR remain later
   preservation controls requiring eligible low-ocular calibration.
6. **NF06 native qualification and fresh source validation.** Frozen shortlisted
   recipe on unchanged native HEOG/VEOG references, blink/lateral/rest/ERP or BCI
   outcomes where eligible. No native reconstruction SNR. Fix source/reference
   coupling in a separately versioned controlled corpus; preserve old scores.
   Only a frozen system evaluated on reserved/independent data can satisfy the
   fresh15dB target. No student training is authorized by current development gain.

Middle/temporal subgroup results determine whether separate policies are useful;
montage labels alone do not establish physiological artifact dominance. Protect
central mu/alpha/beta and posterior alpha with the existing paired/rest tests;
actual task-response preservation additionally requires eligible labelled data.

Each new numerical arm gets a Git module, pinned Kaggle notebook,50example pilot,
recoverable fits/predictions/progress, matched320comparison only after contracts
pass, source-aware uncertainty and native qualification separately. Runtime and
memory at10–50channels remain measured deliverables, not assumed scalability.
