# Processing regions, posterior evidence, visualization contract and VMD variants

16 October 2026. Explanation plus bounded experiment proposal. No SVMD/MVMD
experiment is claimed here; AutoVMD remains stopped. Original posterior results
owned by the other chat are read-only.

## Region meaning: the implementation is the authority

data.py maps Fp/AF/F names to frontal (0), P/PO/O plus Iz to posterior (1),
FC/FT/C/CP/TP/T to central/shared (2), and unknown names to unknown (3).
Example Fp1,F3,Fz are frontal; P3,Pz,O1 are posterior; FCz,C3,Cz,CPz,T7 are
central/shared. These are project processing groups, not exact cortical lobes.
CP means centroparietal but is in the shared group in this implementation;
FC means frontocentral but is not in the VMD frontal mask. Do not infer regions
from channel indices. Unknown rows pass through in the current classical runs.

The generated illustration docs/assets/head_processing_regions_20261016.png
uses curved orange frontal, blue posterior and teal central/shared zones.
It is an educational AI-generated top-view illustration, not measured sensor
coordinates or a quantitative topography. Built-in image generation was used.
Prompt specification: nose-up head; example electrodes in their implemented
groups; curved boundaries; legend of prefixes; explicit conceptual-boundary
disclaimer. It does not count as an empirical scalp result.

Shared processing and shared frontal context are different. The first names an
output group. The second means information passed between branches, e.g.
separate left/right/midline frontal summaries and signed right-minus-left.
Central/shared electrodes can use a non-VMD algorithm like posterior electrodes
while being fitted/scored on a distinct mask. There is no unique, established
algorithm called the shared method.

## Work completed

Frontal: fixed K3/alpha2000 decomposition, reference-associated mode selection,
calibrated EOG field correction, raw/shared and separate-axis regression controls,
anchored gain and solver-only raw fallback. VMD-SOBI and selective correction
adaptations were also explored; exact author replication is not universally
established. The latest controlled hybrid has17.804727dB whole/18.261209dB frontal
channel-first output SNR. This is not VMD alone. Native OSF two-record testing
shows ocular suppression but fails all four resting preservation limits.

Posterior early broad experiment:737 candidates,320 examples,32 recipients,
1440 calibration fits,92 tests. ICA and GEVD-MWF each compared posterior-only,
common frontal, signed summaries, raw frontal and full montage support. Correct
posterior output only; unscored calibration fits; single original-input correction.
The search-table leaders were direct regression13.542dB and full-montageICA
13.523dB posterior output; signed-summaryICA12.918dB, MWF12.335dB. These are
development candidate maxima after a partial preservation screen, not held-out
performance or native qualification. Source: posterior results.md.

Later nonfrontal experiments: ICA rejection, reference regression within ICs,
waveletICA, selective waveletICA, rest-whitened reference fields rank1/2 and
reference-side/EEG-side regularized CCA. Frontal outputs held fixed; posterior,
central and combined corrections compared. In the32-recipient component study,
joint ICA component regression gave14.735326dB whole, reference rank2 gave
15.155692dB, waveletICA12.396655dB and selectivewaveletICA12.333200dB.
These are whole-head scores, not posterior-only scores. Rank2 posterior output
was14.095698dB and central14.905802dB. Negative outcomes are retained.

The later complete-reference/CCA comparison recorded posterior15.350217dB
for direct joint lagged ridge,14.432034dB referenceCCA rank2 and14.095698dB
previous rank2. Middle/temporal were separately reported. This used legacy
regression-generated mixtures with known reference-scaling limitations; do not
compare its leaderboard directly with the repaired homogeneous corpus.
In the latest repaired broader comparison, the common nonfrontal direct branch
has posterior15.473721dB and central/shared19.203094dB. It was held identical
across corrected frontal arms; that experiment is not a fresh ICA/MWF comparison.
No posterior winner or native posterior qualification follows from these values.

SGEYESUB reproduction was audited/planned; required calibration label recovery
and reference parity were not established in the evidence inspected here. Do
not list it as a successfully reproduced experiment. See NONFRONTAL_CCA_PLAN.md,
NONFRONTAL_CHALLENGERS_RESULTS.md and POSTERIOR_CHALLENGERS_20261013.md.

## Posterior improvement priorities

1. Use the repaired corpus for a small matched shortlist: identity, direct joint
   EOG regression, full-montage ICA with posterior-only correction, IC reference
   regression and reference-rank2. No repeated737-recipe search.
2. Establish posterior and central native resting preservation independently.
   The latest native frontal-only test says nothing about posterior correction.
3. Diagnose reference-field scaling, signed lateral transfer, calibration rank
   and condition-specific failures. More suppression cannot excuse preservation
   failures. Preserve low-artifact posterior samples with observed-data support.
4. Estimate only the reference-associated contribution within selected ICs;
   compare against whole-component rejection under identical masks/calibration.
5. Test communication under matched recipes: none, common frontal, signed
   left/right/midline and qualified frontal estimated ocular field. Frontal
   estimates can leak neural content; fit transfer on calibration and apply
   branch quality flags. Strong frontal activity alone must not trigger posterior
   correction. Current unqualified frontal output is not clean ground truth.

## SVMD and bounded variants

Nazari/Sakhaei2020 SVMD extracts modes successively rather than jointly and
does not require a preset mode count. It still needs stopping and bandwidth/
regularization choices. The original simulations report lower computation and
initialization sensitivity; measure these on our EEG before making that claim.
https://www.sciencedirect.com/science/article/pii/S0165168420301535
Author corrected MATLAB implementation:
https://www.mathworks.com/matlabcentral/fileexchange/98649-successive-variational-mode-decomposition-svmd-m

Sravanthi/Sharma, BSPC115(2026),109434, describes SVMD plus fuzzy C-means for
eye-blink removal. The DOI contains2025 but the issue is15April2026. The
accessible abstract does not establish lateral-eye efficacy or provide enough
settings for exact replication. Do not claim the full method has been audited.
https://doi.org/10.1016/j.bspc.2025.109434

SVMD is not simply running VMD K1 repeatedly on residuals; its optimization
includes separation criteria. It is also not the stopped AutoVMD neural/grid
queue. A standalone frontal SVMD challenger is consistent with the new request.
No posterior or central SVMD is proposed.

Bounded sequence: audit author algorithm/license/settings and verify a port
against author fixtures; compare frontal fixedVMD K3/2000 versus SVMD with the
same correction mechanism and raw regression control. Then compare the same
decompositions with selective time-local reference correction. This factorial
separation distinguishes solver benefit from correction benefit. Freeze settings
on development calibration; keep confirmation closed. Save variable mode counts,
centers, residual, zero/peak mode correlations, stopping reason, runtime and every
failure. SVMD is a decomposition challenger, not a promised cure for overcorrection.

Second-priority MVMD aligns common modal frequencies across frontal channels:
https://arxiv.org/abs/1907.04509 . Common frequency is not proof of ocular origin.
VME extracts a desired mode using approximate center information; its original
ECG respiratory application does not establish EEG EOG performance:
https://pubmed.ncbi.nlm.nih.gov/28783649/ . Neither is currently tested here.

## Immediate visualization deliverable

Notebook78 runs native-visual-audit on pinned native-frozen-frontal001/1. No
new VMD, fit, prediction changes or threshold tuning. Nine figures saved as
PNG/PDF/SVG: lateral,blink,vertical association; resting modification,alpha,beta,
covariance; resting PSD; mode-reference evidence. Full mode audit saves all
12852 vectors' zero-lag and signed peak-lag HEOG/VEOG correlations, peak lags,
frequency/selection/solver flags and replays every saved mode selection.
This repairs the missing audit table without changing original results.
The first-example heatmap is a deterministic illustration, not all-mode evidence
by itself; the complete CSV is the audit. All numerical work is on Kaggle.

## Paper figure register:32 distinct analysis questions

Main/supplement allocation depends on journal limits. Multi-panel figures are
appropriate, but do not manufacture plots to reach a count. Each figure needs
exact run/SHA/input identity, metric definition, units, sample/participant counts,
failure handling, selection policy and reproducible numerical source tables.
The generated head is conceptual; measured figures require actual evidence.

| ID | Figure/question | Evidence status |
|---|---|---|
|01|Processing-region head schematic|Generated conceptual illustration|
|02|Regional correction/data-flow diagram|Design, not numerical evidence|
|03|Source/recipient/donor partitions|Verified metadata; render pending|
|04|Blink/lateral/mixed input examples|Saved arrays; render pending|
|05|Frontal mode vectors and original EEG|Saved modes; render pending|
|06|Mode-plus-residual reconstruction error|Saved numerical diagnostics|
|07|Mode center frequencies and bandwidths|Saved numerical diagnostics|
|08|Zero-lag mode–HEOG/VEOG heatmap|Notebook78 audit|
|09|Peak-lag correlations and peak lag distribution|Notebook78 audit table|
|10|Selected modes/solver flags across windows|Notebook78 audit table|
|11|FCM features/memberships|Requires executed eligible FCM variant|
|12|Ocular support and correction intervals|Saved decisions; diagnosis pending|
|13|Frontal SNR by condition/input level|Saved controlled results|
|14|Posterior SNR by method and condition|Saved, retain corpus separation|
|15|Central and temporal outcomes|Saved regional results|
|16|Whole-head channel-first versus pooled SNR|Saved, label distinct measures|
|17|MSE and temporal/spectral RRMSE|Saved where implemented; no invented metrics|
|18|Paired Pearson CC and deterioration|Saved controlled results|
|19|Electrode-level error/topography|Requires verified montage coordinates|
|20|Recipient-level outcomes|Saved development tables|
|21|Recipient/donor-aware uncertainty|Saved supplemental procedure, label clearly|
|22|Resting frontal PSD|Notebook78 figure08|
|23|Resting alpha preservation|Notebook78 figure05|
|24|Resting beta preservation|Notebook78 figure06|
|25|Resting covariance preservation|Notebook78 figure07|
|26|Resting modification/RMSE|Notebook78 figure04 plus saved RMSE|
|27|Native blink correlation|Notebook78 figure02|
|28|Native lateral correlation|Notebook78 figure01|
|29|Native vertical correlation|Notebook78 figure03|
|30|Convergence/fallback/failure distribution|Saved modes and failure ledgers|
|31|VMD/SVMD/MVMD runtime and quality|Requires new bounded comparisons|
|32|10–50-channel time/memory and export parity|Not measured; no fabricated curve|

For the final report, target at least28 evidence-backed plots/panels across main
paper and supplement. The present deliverable is nine native figures plus one
conceptual illustration and the complete mode audit, not all32 completed figures.
