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
these projected features and selected-mode observations. Apply the bounded gains
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
