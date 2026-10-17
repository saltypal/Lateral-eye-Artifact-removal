# Central, temporal and posterior candidates after frontal closeout

Reviewed17October2026. This is a prospective research shortlist, not new
waveform results. The active numerical run scores frontal only. No LEMON,
nonfrontal VMD or AutoVMD search is introduced.

## What decides a regional method

Central means FC/C/CP; temporal FT/T/TP is separate; posterior uses verified
parietal/occipital names under the existing explicit mapping. These are electrode
groups, not proof that a signal originates exclusively in the corresponding lobe.
A method is not inherently central or posterior: a spatial fit can use the full
montage, while the correction output and preservation tests are restricted to
the specified group. Full montage is not a VMD-to-posterior message pathway.

The previous central/temporal ridge and instantaneous rank-one MWF experiments
did not qualify any corrective method. Higher paired SNR alone cannot override
their documented clean-input and native resting distortion. The old posterior
results.md uses the retired LEMON controlled construction; retain its historical
evidence but do not use it to rank Klados/OSF candidates in the active project.

## Small candidate set and why

| Candidate | Mechanism | Proposed role and required check |
|---|---|---|
| Original SGEYESUB | Learn horizontal, vertical and blink spatial directions from six-class calibration; subtract the estimated ocular subspace | Reuse the validated full-montage fit for later central/posterior outputs. Native OSF is eligible; exact Klados reproduction is not eligible without calibration classes. Frontal results do not guarantee posterior preservation. |
| Original REG-ICA routines | Extended Infomax, fractal identification, adaptive EOG regression inside selected components | Central and separately temporal challenger. Neural activity mixed into selected components remains a risk. Report trial-local adaptation, original selection and solver caps; do not call fixed ridge REG-ICA. |
| Targeted wavelet ICA in RELAX | Correct artifact periods in ocular components instead of subtracting whole components continuously | Highest-priority new preservation challenger for central/posterior; temporal separately. Audit original code/dependencies and prove unchanged intervals, reconstruction and waveform boundaries before large evaluation. No claim of superiority in our datasets yet. |
| Author GEVD-MWF with annotated calibration | Estimate covariance from artifact and low-artifact intervals, retain a low-rank artifact model, optionally include temporal delays | Posterior challenger to the existing instantaneous adaptation. Use separate blink/lateral masks and independent resting calibration, compare the author routine to our existing implementation first. Preserve posterior alpha/covariance; temporal delays cannot cross trial boundaries. |

Keep identity and calibrated two-reference ridge as controls. These four
candidates are a bounded comparison set, not four mandatory cleaning stages.
Do not chain them merely because each exists.

## Literature that changes the design

1. **SGEYESUB, Kobler et al.2020.** The author repository explicitly provides
   calibration fitting and subsequent online/offline correction. Six original
   sample classes are required by its code. This supports the present original-
   code OSF comparison, not guessed class generation for Klados.
   [Paper](https://doi.org/10.1016/j.neuroimage.2020.117000),
   [author implementation](https://github.com/rkobler/eyeartifactcorrection).

2. **REG-ICA, Klados et al.2011.** Its rationale is to regress the ocular
   contribution inside artifact-containing independent components, where it is
   expected to be more concentrated than at the electrodes. That is a hypothesis
   to test against direct regression, not a guarantee that neural content is
   absent. Our calibration-frozen decomposition and per-trial SRLS reset remain
   explicitly documented evaluation adaptations.
   [Publication record and abstract](https://pure.mpg.de/view/item_2095219),
   [official original plugin](https://sccn.ucsd.edu/eeglab/plugins/REG-ICAv1.00.zip).

3. **Mohamed and Juhasz2019 selective wavelet ICA.** The described algorithm
   uses frontal-component correlation to identify ocular components, detects
   peaks and corrects one-second neighborhoods using wavelet decomposition.
   It discusses both horizontal and vertical/blink contamination without
   requiring separate EOG reference channels. Its1–47Hz/average-reference
   preprocessing differs from ours; a matched0.5–40Hz version must be labelled
   an adaptation. Do not present an unavailable author implementation as run.
   [Paper](https://www.mdpi.com/2076-3425/9/12/355).

4. **Bailey et al.2025 targeted cleaning.** Published online10October2025,
   Clinical Neurophysiology180,2111378. This is the published successor to the
   older2024 preprint still cited in the repository README. The study targets
   eye-component periods and muscle-component frequencies and evaluates ERP,
   connectivity and source-localisation distortion. Our inference is that this
   provides a particularly relevant preservation challenger; it does not show
   a measured central/posterior win on our Klados/OSF protocol. The released
   RELAX repository contains RELAX_targeted_wICA.m. Verify its dependencies and
   distinguish targeted wICA from the full RELAX pipeline, which also changes
   reference and can reject channels/intervals.
   [Publication record](https://pubmed.ncbi.nlm.nih.gov/41092567/),
   [author code](https://github.com/NeilwBailey/RELAX).

5. **Somers et al.2018 GEVD-MWF.** The method uses annotated artifact segments
   and a generalized-eigenvalue low-rank artifact covariance approximation;
   author code also supports delay embedding. This motivates testing an actual
   author implementation, rather than assuming our instantaneous rank-one
   adaptation exhausted MWF. Artifact/rest labels do not create a known clean
   waveform. The repository's delay-spacing examples are informal guidance,
   not universal200Hz settings or a licence to tune on evaluation targets.
   [Publication](https://pubmed.ncbi.nlm.nih.gov/29393057/),
   [author implementation](https://github.com/exporl/mwf-artifact-removal).

6. **RELAX2023.** Its MWF/wICA comparison supports treating these as meaningful
   alternatives with preservation checks. Its full preprocessing/rejection
   pipeline is not directly comparable to our fixed-window outputs unless
   changes, exclusions and reference differences are controlled.
   [Original paper](https://www.sciencedirect.com/science/article/pii/S1388245723000275).

ASR stays a secondary broad-artifact control. Its original calibration code
requires representative low-artifact EEG; its processing reconstructs high-
variance subspaces relative to that calibration. It is not a directional HEOG/
VEOG separator, so we should not prioritize it over targeted ocular methods
without evidence. Any future trial must account for lookahead, calibration
quality and all altered samples.
[Original calibration](https://github.com/sccn/clean_rawdata/blob/master/asr_calibrate.m),
[original processing](https://github.com/sccn/clean_rawdata/blob/master/asr_process.m).

## Execution order after frontal results

- [ ] Close the current frontal comparison with executed notebook and failures.
- [ ] Select no corrective frontal winner if the preservation evidence fails.
- [ ] Audit targeted-wICA and author-MWF dependencies; pin original source versions.
- [ ] Verify unchanged-interval behavior, reconstruction and original-code runtime on Kaggle.
- [ ] Reuse full-montage SG/REG fits for separately scored central and posterior outputs; temporal remains separate.
- [ ] Compare targeted wICA and author MWF against those fits and identity/ridge with identical arrays.
- [ ] Report paired SNR/RMSE/CC and native blink/lateral suppression with preservation first.
- [ ] Freeze at most two defensible candidates per region, or report no qualified correction.

No invented overall score. Preserve existing thresholds: worst-record/participant
mean modification1%, alpha/beta0.5dB, covariance0.02 and paired CC deterioration
0.005 against the declared matched comparator. Native rest remains a proxy;
there is no native reconstruction SNR. Keep all failures and do not add dB
across regions. Frontal-to-back communication is a later explicit experiment.
