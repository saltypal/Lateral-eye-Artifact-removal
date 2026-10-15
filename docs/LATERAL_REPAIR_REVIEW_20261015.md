# Literature-led lateral-eye repair: fixed frontal VMD

## Problem and bounded implementation plan

The completed repaired-corpus pilot has a severe lateral frontal example at
8.812491 dB versus 13.486290 dB for ridge, with CC deterioration 0.0562836.
The current VMD gain uses selected-mode activity with a full-field denominator.
This may attenuate the estimated gain when ocular activity is distributed beyond
the selected span or remains in the reconstruction residual. The saved gain
0.863411 versus raw 1.030755 supports investigating this mechanism; it does not
prove a complete causal diagnosis or identify an oracle gain.

Implement a new notebook/stage using the exact verified corpus and completed
frontal review as parents. Reuse their saved frontal K3/alpha2000 modes after
checking input hashes, reconstruction and channel identity. Freeze nonfrontal
outputs. No new decomposition, K/alpha search, labels as inference inputs,
scoring targets in fitting, neural training or reserved-source access.

Six predeclared arms: archived identity, archived raw-shared ridge, archived
VMD-shared gain, VMD matched-subspace scalar gain, raw separate-axis gains, and
VMD matched-subspace separate-axis gains. Apply every arm to every condition;
do not route using blink/lateral scoring labels. Temporal gating is a subsequent
mechanism if evidence warrants it, not a silent change in this experiment.

## Primary-source review and implementation consequences

### Plöchl, Ossandón and König (2012)

[Combining EEG and eye tracking](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2012.00278/full),
DOI 10.3389/fnhum.2012.00278. Their measured ocular generators include eyelid,
corneo-retinal and saccadic contributions with different topographies. Two EOG
axes do not necessarily span all sources; adding a third reference alone does
not guarantee improvement. Consequence: preserve signed reference fields and
test separate horizontal/vertical gains, while retaining native validation.
This motivates an adaptation, not replication of their eye-tracker-guided ICA.

### Frontal VMD–regression (2020)

[An improved algorithm for efficient ocular artifact suppression from frontal EEG electrodes using VMD](https://www.sciencedirect.com/science/article/pii/S0208521618303528),
DOI 10.1016/j.bbe.2019.03.002. Publisher text explicitly includes lateral eye
movements and describes segment detection with modified multiscale sample
entropy, ocular BLIMF estimation, and regression of the estimated artifact.
The accessible text describes a zero-center option and low-frequency separation;
complete numeric settings were not recovered in this review. Our fixed K3/2000
reference-assisted gain is not its reproduced algorithm. Consequence: VMD can
provide an artifact representation rather than justify whole-mode deletion.
Changing DC constraints is deferred; this experiment reuses the existing modes.

### Issa and Juhasz (2019)

[Improved EOG Artifact Removal Using Wavelet Enhanced Independent Component Analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC6956025/),
DOI 10.3390/brainsci9120355. They correct selected component intervals rather
than discard an entire component; the described pipeline uses one-second
correction windows and five wavelet levels, and covers vertical and horizontal
ocular sections. Consequence: reserve a temporal-support challenger if the
matched gain repair still modifies clean intervals. The wavelet/ICA paper does
not validate our VMD adaptation. Its original filtering and wavelet scales
cannot be copied blindly to our 200 Hz preprocessing.

### Schlögl et al. (2007)

[A fully automated correction method of EOG artifacts in EEG recordings](https://pub.ista.ac.at/~schloegl/publications/schloegl2007eog.pdf),
DOI 10.1016/j.clinph.2006.09.003. A reference-regression correction was evaluated
with blinded expert scoring. Consequence: retain the joint reference regression
control; regression is a scientifically established competitor, not an invalid
method merely because a VMD hybrid is preferred. Our lagged ridge and scoring
gain are project extensions, not author-equivalent settings.

### Berg and Scherg (1994)

[A multiple source approach to the correction of eye artifacts](https://www.sciencedirect.com/science/article/pii/0013469494900949),
DOI 10.1016/0013-4694(94)90094-9. The accessible publisher description separates
vertical/horizontal movements and blinks through source spatial vectors.
Consequence: maintain source/axis distinctions and signed anatomy; a complete
multiple-source reproduction needs additional model and calibration evidence.
Do not pretend two observed axes represent independently known ocular sources.

### Liu and Zhang (2022)

[Remove Artifacts from a Single-Channel EEG Based on VMD and SOBI](https://pmc.ncbi.nlm.nih.gov/articles/PMC9460664/),
DOI 10.3390/s22176698. The paper combines mode decomposition, second-order
separation and entropy-based selection. Its semi-simulated EOG/EMG results do
not establish lateral-specific superiority under our protocol. Consequence:
do not restart the previously tested SOBI queue without evidence that gain
repair leaves a source-separation problem. K3/2000 remains a project recipe.

## Retrieval quality and limits

alphaXiv discovery was consulted. Its PDF lookup correctly retrieved the 2024
cepstral prefrontal paper (arXiv:2404.08199), but title lookups for the 2020 VMD,
2019 wavelet and 2022 VMD-SOBI papers returned unrelated papers that cited them.
Those mismatches were rejected as evidence. Publisher/PMC/author sources above
control the review. No claim is based on a title-match result alone.

The cepstral paper reports classification/detection and feature-space correction;
its recognition accuracy is not waveform reconstruction SNR. It is not the
immediate replacement for a waveform-preserving teacher.

## Low-level change and tests

Add `gain_mode="projected_axis"` to the existing frontal calibration transfer.
Calibration still fits joint lagged HEOG/VEOG coefficients. Split the two signed
axis fields using the existing lag-major row ordering. Project each field into
the selected VMD-mode span for that channel. Estimate two regularized gains from
these projected features and selected-mode observations. Anchor regularization
at the matched-subspace common gain: a narrow projected span can otherwise make
the two fields collinear and let a zero-centered minimum-norm solution assign
large, physically inconsistent gains when restored to the full fields. Save the
prior and projected Gram eigenvalues. Apply the bounded gains
to the full calibrated axis fields once. No absolute-value HEOG substitution,
eye-direction guessing, independent per-channel scoring regression or global
correlation-to-zero objective. Existing modes and their residuals remain intact.

Tests on Kaggle: distinct-axis recovery; reference-polarity invariance; channel
and mode permutation; unit scaling; all-failed and no-mode support pass-through;
constant/missing axis; preservation of nonfrontal/masked rows; archived control
and nonfrontal parity; cached input hashes and rounding-aware reconstruction.

## Acceptance and report

Save per-record/channel and condition metrics, MSE/RRMSE/Pearson, both output-SNR
aggregations, residual/reference diagnostics, gains, projected field coverage,
predictions, failures, environment and source identity. Score lateral first but
report blink/mixed/clean as well. Keep the existing CC<=0.005 deterioration,
clean modification<=1%, alpha/beta<=0.5dB and covariance<=0.02 requirements.
Two pairs cannot qualify generalization even if these checks pass. Native OSF
and broader grouped development remain required; no student authorization.

## Mathematical implementation of the repair

For frontal channel c, let v_c be its selected-mode sum, f_c its calibrated
joint ocular field, and P_c the temporal projection onto its selected-mode
span. All vectors in these gain calculations are centered on valid scoring
samples. The old gain uses sum_c <v_c,f_c> / sum_c ||f_c||². The matched scalar
gain uses sum_c <v_c,P_c f_c> / sum_c ||P_c f_c||², followed by the existing
[0,8] clipping. Zero projected energy yields zero gain.

For two signed axis fields f_c,H and f_c,V, define features P_c f_c,a and their
RMS values d_a, stacked over usable frontal rows and samples. Write normalized
features as Z_a = feature_a/d_a and coefficients theta_a = d_a*g_a. The repaired
axis fit minimizes ||Z^T theta-v||²/N + lambda*||theta-d*g0||², with lambda=0.01
and g0 the matched scalar gain. Solve the regularized normal equations, convert
theta back to physical gains, then apply the declared [0,8] clipping. This last
clipping is a safety operation, not an exact box-constrained optimizer. Inactive
axes receive zero gain. Gram eigenvalues and the common-gain prior are saved.

The prior matters when projection makes the axes collinear: otherwise equal
normalized coefficient penalties can correspond to radically different physical
gains for weak versus strong axes. Neither these equations nor their tests
establish real-data superiority. The full calibrated fields are restored only
after gain estimation, with original EEG minus the estimated artifact once.

## Measured rejection of the projected-field proposal

`lateral-gain-review-002/2` at 9a8a91e completed 322 Kaggle tests and all 20
records. All 37 payloads were recovered and verified; manifest
`c82ce0cddaedb46f0ed57c2e555ff71dda833f9e9260f936eebac0e83268d5c2`.
Tests passed in 23.37 s; numerical experiment took 35.712861 s. No new VMD.
The condition plot was visually inspected: labels and legend are readable.

The projected scalar and projected-axis arms are rejected as proposed fixes:
frontal mean pooled-window SNR was 9.748292 and 6.974390 dB respectively,
versus 21.768334 for the old VMD gain and 20.488443 for raw shared regression.
At severe lateral input, their two-recipient pooled-window means were 10.884512
and 7.876364 dB versus old VMD 10.142813 and raw shared 13.756186 dB. Raw-axis
separation alone did not beat shared regression on every lateral level either.
Do not report the failed proposals as improvements because their tests passed.

Projection onto EEG-derived modes can produce an inflated gain when mode
content is increasingly dominated by neural EEG. A compatible denominator does
not make the data-dependent subspace a correct ocular-source estimator.

## Version 3: residual completion and conservative anchoring

The observed severe-lateral residual of the first frontal channel has absolute
reference association above the existing 0.2 component threshold in both pairs
(VEOG 0.233927 for sub-032305 and HEOG 0.209442 for sub-032306). This motivates
testing residual eligibility, not assuming all residual content is ocular.

Add two fixed arms to the same matched comparison, preserving all six earlier
arms and negative results:

- `vmd_residual`: include the reconstruction residual in the gain observation
  only when it satisfies the same 0.2 reference-association rule as modes. Keep
  the full-field denominator; never delete the residual from EEG wholesale.
- `vmd_anchored`: g = 0.75*g_raw + 0.25*g_VMD with both original bounded gains.
  This limits the effect of incomplete mode selection while retaining a VMD
  adjustment. The weight0.25 is an explicitly declared engineering choice, not
  a published optimum or a learned condition-label selector.

New fixtures isolate a missing ocular contribution in the residual and verify
its recovery, plus reduction of deviation from the raw-field anchor. Existing
reference, units and region fixtures cover both arms. These are development
repairs and remain unqualified until the real pilot, grouped and native checks.
