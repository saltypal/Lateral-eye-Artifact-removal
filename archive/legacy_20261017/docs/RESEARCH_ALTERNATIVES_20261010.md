# Research alternatives: ocular specificity, selective correction and deployment

Reviewed 2026-10-10. This is an evidence review and proposed experiment backlog;
it does not change the active forty-setting AutoVMD campaign. Numerical work
remains on Kaggle. No candidate below is qualified as superior on our data.
The reviewed implementation snapshot is Git SHA
`3c4cf5ae05e86b1113bf6d0968825ace90e68b20`.

## Decision

Keep the approved frontal-VMD/posterior-band-routing experiment running. Prioritize
improving **which portion of a component is corrected, when it is corrected,
and which ocular source drives the correction**. Merely adding decompositions
or increasing network size is not a scientific explanation for improved recovery.

## Primary-source evidence register

Access labels distinguish full methods available from publisher abstracts or
indexed excerpts. Missing code, participant splits and lateral-specific evidence
must be resolved before claiming a replication.

### 1. Selective wavelet-enhanced ICA: strongest new classical challenger

[Issa and Juhasz, Brain Sciences 2019](https://www.mdpi.com/2076-3425/9/12/355),
DOI 10.3390/brainsci9120355. Methods and algorithm were available in indexed
publisher text. It identifies ocular ICA components from frontal association,
locates ocular peaks, and corrects component intervals with wavelets rather than
automatically rejecting entire components. Its methods explicitly discuss both
vertical/blink and horizontal/eye-movement sections. Reported settings include
one-second correction windows, sym4 and five wavelet levels, with whole-component
rejection when interval coverage exceeds 60%. These are paper settings, not
parameters validated at our 200 Hz preprocessing.

Our proposed adaptation: fit a spatial separator using the existing calibration
contract, identify blink and lateral support separately using unchanged VEOG and
HEOG, and correct only reference-supported component intervals. Select wavelet
levels by physical frequency support at 200 Hz. Reconstruct the artifact and
subtract once from the original EEG. Compare to our existing component rejection
and GEVD-MWF controls. Generic low-frequency removal inside a mixed component can
still remove neural activity; interval specificity alone does not guarantee preservation.

### 2. VME-DWT: localization instead of decomposition of every segment

[Shahbakhti et al., IEEE TNSRE 2021](https://pubmed.ncbi.nlm.nih.gov/33497337/),
DOI 10.1109/TNSRE.2021.3054733. Abstract verified; complete implementation not
inspected. Variational mode extraction locates blink intervals, then an automatic
wavelet method corrects those intervals. The abstract reports CC 0.92 and RRMSE
0.42 under its own evaluation. Evidence is blink-specific.

Our use: a frontal event-local correction comparator and a motivation for
training-only event supervision. A lateral branch must be independently validated;
this paper does not resolve it. VME extraction and VMD decomposition are distinct
algorithms. A localization model could reduce computation only if measured false
negatives and boundary errors remain acceptable.

### 3. VME-GMETV: recover EEG from the artifact-dominated component

[Biomedical Signal Processing and Control 2024](https://www.sciencedirect.com/science/article/abs/pii/S1746809423010558),
DOI 10.1016/j.bspc.2023.105622. Publisher abstract/excerpts verified; full equations
and code unavailable in this pass. It combines variational mode extraction with
generalized Moreau-envelope total variation and adds retained EEG residue back
to the non-ocular portion. It reports mean RRMSE 0.1557 and CC 0.9695.

Our use: study component repair instead of whole-mode attenuation. Do not interpret
its average RRMSE as our mean output SNR, or assume its reconstruction reference
and lateral coverage match ours. Exact objective, optimization, parameters, data
split and preservation evidence need verification before an implementation claim.

### 4. MVMD: aligned frontal representations

[Rehman and Aftab, IEEE TSP 2019](https://arxiv.org/abs/1907.04509),
DOI 10.1109/TSP.2019.2951223. The multivariate formulation learns joint modes with
shared center frequencies across input channels. It is a decomposition paper,
including EEG alpha separation, rather than proof of ocular cleaning superiority.
[Gavas et al., PerCom Workshops 2020](https://staff.itee.uq.edu.au/jaga/proceedings/percomworkshops2020/papers/p232-gavas.pdf)
applies MVMD to blink removal. Indexed paper text reports empirical K=10 and
interval identification to avoid rejecting every low-frequency mode. Direct PDF
access failed in this pass; lateral evaluation remains unverified.

Our proposed challenger: decompose a small verified frontal group jointly, retain
signed mode vectors and residuals, and feed frequency-aligned descriptors into
the existing router. Compare against independent frontal VMD with the same sources
and correction capacity. MVMD can improve cross-channel alignment but shared
center frequencies do not identify ocular sources. Joint frontal decomposition
also introduces a montage dependency that must be tested under channel dropping.
Avoid a full-cap mode expansion as the first comparison because it increases cache
and fitting costs and changes more than one experimental factor.

### 5. SVMD-FCM: alternative to choosing K in advance

[Nazari and Sakhaei, Signal Processing 2020](https://www.sciencedirect.com/science/article/pii/S0165168420301535),
DOI 10.1016/j.sigpro.2020.107610. Publisher abstract verified. SVMD extracts modes
successively without requiring the mode count beforehand and reports advantages
in complexity and initialization sensitivity in its demonstrations.
[SVMD-FCM ocular application, BSPC 2026](https://www.sciencedirect.com/science/article/pii/S1746809425019457),
DOI 10.1016/j.bspc.2025.109434. Publisher abstract/excerpts verified. This method
uses successive decomposition and fuzzy clustering for eye-blink artifact removal.
Lateral-specific results and exact split independence were not verified.

Our use: a later decomposition challenger if fixed-recipe AutoVMD shows unstable
mode counts or excessive runtime. Automatic stopping still has assumptions and
parameters to validate. SVMD is not the implemented AutoVMD grid, and introducing
variable modes per window would expand the approved first-version scope. FCM
membership should drive selective correction rather than an assumed permission
to erase a whole mode.

### 6. Frontal-supported MWF and calibrated ocular subspaces

[Frontal-supported blink MWF, BSPC 2018](https://www.sciencedirect.com/science/article/pii/S1746809418301149)
estimates blink artifacts using a subset of frontal electrodes. Publisher abstract
verified; blink evidence does not establish lateral correction.
[Somers, Francart and Bertrand, JNE 2018](https://pubmed.ncbi.nlm.nih.gov/29393057/)
and [author toolbox](https://github.com/exporl/mwf-artifact-removal) use annotated
artifact/clean segments and a low-rank GEVD artifact-covariance estimate.
[Kobler et al., NeuroImage 2020](https://www.sciencedirect.com/science/article/pii/S1053811920304869)
and [author SGEYESUB code](https://github.com/rkobler/eyeartifactcorrection) address
eye movement and eyelid artifacts with dedicated calibration. The study compared
69 participants, and its abstract reports a favorable correction/preservation
trade-off after approximately five minutes of calibration.

Our use: these strengthen the posterior spatial-estimation branch already in the
plan; they are rechecked references, not newly implemented algorithms. Add separate
blink/lateral calibration strata and quality flags to investigate whether one
covariance model averages away distinct ocular patterns. Verify eligibility for
SGEYESUB's calibration paradigm before using that name for a result. Posterior
weak contamination should not trigger strong correction merely because frontal
amplitude is high. Fit posterior transfer using its own calibration evidence.

### 7. VMD-SOBI and frontal VMD regression: rechecked controls

[Xiong et al., Sensors 2024](https://pubmed.ncbi.nlm.nih.gov/38475177/),
DOI 10.3390/s24051642, combines SVM detection, genetic-algorithm VMD parameter
optimization, SOBI and entropy-based source selection. Abstract and indexed
methods/results reviewed; a direct full-text page was blocked in this pass.
[Frontal VMD ocular algorithm, BBE 2020](https://www.sciencedirect.com/science/article/pii/S0208521618303528),
DOI 10.1016/j.bbe.2019.03.002, explicitly targets blinks, lateral movements and
flutter, using segment detection, VMD-derived artifact estimation and regression.

Our use: retain these as existing-control motivations. Our source-separated
grid and EOG-guided adaptations are not replications of their SVM/GA pipelines.
Pseudochannels derived from one electrode add representations rather than new
independent physical observations, so an extra BSS step is not guaranteed to
improve source identifiability. Whole-component rejection remains a preservation risk.

### 8. BandRouteNet and Band2CleanFormer: frequency-aware neural controls

[BandRouteNet v2, May 2026](https://arxiv.org/html/2604.24428v2) provides full-band
conditioning, temporal band routing and cross-band fusion. Its EOG benchmark table
reports 13.9819 dB SNR improvement, not 13.9819 dB output SNR. Its printed RMS-based
input-SNR equation requires reconciliation with the energy-ratio contract. The
text describes an 8:1:1 split after augmentation without enough detail to establish
recipient/artifact source disjointness. These are reproduction questions, not
a finding that the authors leaked data.
[Band2CleanFormer, BSPC June 2026](https://www.sciencedirect.com/science/article/abs/pii/S1746809426004143),
DOI 10.1016/j.bspc.2026.109860, describes FFT band processing and CNN/Transformer
inter-band modeling on EEGdenoiseNet. Abstract/excerpts verified, exact code and
complete split procedure not inspected.

Our use: the current fixed-band arm is the necessary control for the VMD claim.
Inter-band attention alone cannot establish the value of VMD, frontal anatomy,
or lateral-specific correction. Benchmark results must be reproduced separately
at their sampling rate and target definition.

### 9. Diff-ADN: deployment-compatible residual refinement

[Jamil et al., JNE, 22 September 2026](https://pubmed.ncbi.nlm.nih.gov/42772331/),
DOI 10.1088/1741-2552/aeab36. Indexed primary abstract verified; full methods and
author implementation not retrieved. It uses severity-conditioned reconstruction
followed by diffusion-supervised residual refinement during training. Inference
uses one deterministic residual correction rather than iterative reverse sampling.
The abstract reports EOG Pearson correlation 0.936 and CPU timing on the tested
machine, with independent artifact sources and downstream motor-imagery evaluation.

Our proposed use: a later student refinement experiment if paired residual errors
remain structured. Preserve the actual noisy EEG as the reconstruction anchor.
Match parameter/training budgets, test fresh CPU latency and evaluate scientific
features as well as reconstruction. Neither generative training nor the reported
CC establishes our 15-20 dB target. Do not fabricate the missing diffusion schedule
or claim exact replication from an abstract.

### 10. Successive jump-and-mode decomposition: exploratory lateral morphology

[Nazari et al., 2025 preprint](https://arxiv.org/html/2504.08453v1) separates
oscillatory modes from jump components, including a multivariate extension.
Full text available. No ocular EEG benchmark was verified.

Our hypothesis: sustained gaze shifts may benefit from an explicit non-oscillatory
representation, rather than forcing every artifact into narrowband oscillations.
This is a research inference. Neural slow potentials also have non-oscillatory
structure; jump detection cannot serve as an artifact label. Keep it exploratory,
with HEOG-supported paired-error evaluation and clean slow-potential preservation.

## Proposed bounded research backlog

1. **Existing results first:** inspect condition-specific residuals of fixed-band
   and regional VMD runs. Determine whether error is dominated by artifact peaks,
   sustained lateral shifts, mistaken clean correction, or reconstruction boundaries.
2. **Selective correction control:** compare interval-corrected wavelet ICA to
   component rejection and GEVD-MWF, with identical calibration/scoring intervals.
3. **Ocular-specific training auxiliary:** use training HEOG/VEOG and calibration
   to generate separate blink/lateral support labels with an unknown/ambiguous state.
   Test an event head and support-weighted paired-error loss against MSE alone.
   EOG remains absent from deployment inputs. Do not label weak reference correlation
   as clean, or force native EEG/EOG correlation to zero.
4. **Posterior two-source estimation:** compare separate calibrated blink/lateral
   transfer models to the current combined model; estimate correction relative to
   original posterior input and subtract once. Preserve signed lateral information.
5. **Frontal MVMD challenger:** one small verified frontal group with identical
   input and loss; document montage dependence and missing-channel fallback.
6. **Student residual refinement:** investigate Diff-ADN only after full algorithm
   access or label an independently designed adaptation explicitly.
7. **SVMD/SJMD alternatives:** evaluate after the fixed-K search evidence identifies
   a concrete decomposition limitation. They require a scoped plan amendment.

These are hypotheses and priorities. They are not queued jobs, measured gains,
approved architecture replacements, or completion of the active search.

## Evidence required to promote an alternative

Use the same development recipient/donor identities, preprocessing, target scale,
input levels and held-out intervals as its comparator. Report blink, lateral and
mixed errors separately. Include convergence/fallback counts, clean modification,
alpha/beta error, covariance error, paired correlation, runtime and memory.
Reserve confirmation remains closed during this work. Native OSF receives unchanged
HEOG/VEOG association and preservation proxies, never paired reconstruction SNR.

For one paired example under our energy definition,
`SNR_out = -20 log10(RRMSE_temporal)`. Thus 15 dB requires relative error about
0.178 and 20 dB requires 0.1. Converting an arithmetic mean RRMSE into a mean SNR
is invalid because logarithms do not commute with averaging. Metric definitions,
input contamination and source splits must match before comparing published scores.

## Completion checklist for this research pass

- [x] Primary-source alternatives searched and concrete mechanisms compared.
- [x] Existing VMD-SOBI/MWF/SGEYESUB distinguished from new challengers.
- [x] Blink-only, lateral-unverified and general decomposition evidence labelled.
- [x] Access/reproduction gaps and proposed implementation adaptations recorded.
- [x] Active Kaggle queue preserved while research proceeds.
- [ ] Full algorithm/code access for VME-GMETV, SVMD-FCM and Diff-ADN.
- [ ] Candidate-specific, matched Kaggle evidence before any superiority claim.
