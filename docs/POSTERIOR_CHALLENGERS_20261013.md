# Posterior challengers — 13 October 2026

Literature-supported proposals, not measured winners or newly implemented methods.
VMD remains fixed and frontal-only; AutoVMD remains stopped. Existing posterior
source-excluded direct regression/full ICA results remain the matched controls.

## Prioritized candidates

1. **SGEYESUB reference-method reproduction.** Calibration-dependent ocular
   subspace subtraction addresses blink and eye movement; the author supplies
   code and the OSF dataset already used here. Strong fit to native preservation
   evaluation, but eligibility and implementation parity must be checked.
   Posterior-only output is our regional adaptation, not the original protocol.
   No claim of EOG-free or calibration-free deployment.
   [Paper](https://doi.org/10.1016/j.neuroimage.2020.117000),
   [author implementation](https://github.com/rkobler/eyeartifactcorrection).

2. **Selective wavelet-enhanced ICA.** Correct ocular intervals inside selected
   components instead of discarding the entire component. Published scope covers
   blink and horizontal movement and includes Klados evaluation. Wavelet scales
   and boundaries must be adapted/verified at200Hz; low-frequency neural content
   can still be lost. Native preservation is required.
   [Issa/Juhasz2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6956025/).

3. **Dual-reference adaptive FIR correction.** He/Wilson/Russell2004 uses HEOG
   and VEOG with RLS adaptation. A bounded, event-gated adaptation is a proposed
   extension, not the paper's exact algorithm. Online coefficient updates can
   absorb neural/reference correlation; compare frozen calibration versus
   conservative adaptation, with rest identity and stability tests.
   [Original paper](https://pubmed.ncbi.nlm.nih.gov/15191087/).

4. **Estimated frontal ocular-field transfer.** Our next hybrid hypothesis:
   use fixed frontal VMD's estimated blink/lateral contributions as posterior
   predictors, preserving signed left/right context and quality flags. Fit
   posterior transfer on unscored calibration and subtract once from original.
   This differs from the already-tested raw frontal ICA/MWF support. Borowicz2018
   supplies precedent for frontal-estimated blink propagation, but not validation
   of this VMD/lateral extension. Frontal artifact estimates are not clean ocular
   truth and can carry neural leakage.
   [Blink-only precedent](https://doi.org/10.1016/j.bspc.2018.05.012).

## Lower-priority option and interpretation limits

ASR is a general transient/high-amplitude artifact method. It can improve later
ICA, but residual ocular components may remain; it is not an assured blink/lateral
solution or our first preservation challenger.
[Evaluation](https://pubmed.ncbi.nlm.nih.gov/31329105/).

None of these papers establishes a posterior-specific winner for our corpus.
The whole-cap9.09dB result also includes uncorrected central/shared rows. Before
an expensive method comparison, run a Kaggle-only regional/channel error
breakdown; better posterior cleaning alone may not close that whole-cap gap.
Compare fixed VMD+posterior pass-through, direct, fullICA and new candidates
under identical data, calibration, metrics and native rest gates. No new broad
search or neural campaign is launched by this literature note.
