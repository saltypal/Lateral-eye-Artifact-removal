# Frozen regional system: central/shared coverage experiment

This is an exploratory development experiment authorized on 13 October 2026.
The full diagnostic found 87.9066% mean per-example reconstruction-error share
in untouched central/shared channels for the existing K3 frontal + direct
posterior arm. That share is participant/condition/input-level balanced; it is
not a ratio of globally pooled error sums. Frontal output SNR is 17.1128 dB,
posterior 13.5417 dB, and whole-cap channel-first mean 9.0942 dB.

## Locked inputs and four primary arms

Use the same frozen 320 development examples, recipient/donor partition,
preprocessing and predictions from regional-full-001/1. Pilot retains the first
recipient in each fold, all three input strengths and clean controls (50
examples). Full retains all 320. Reserved sources remain closed.

1. Frozen K3/alpha2000 frontal correction + frozen direct posterior correction;
   central/shared pass-through.
2. Same frozen branches + calibration-only joint HEOG/VEOG lagged ridge field
   on verified region-2 channels.
3. Same frozen branches + separately event-gated HEOG and VEOG contributions
   from the same joint ridge fit. Thresholds use resting calibration only.
4. Same frozen branches + calibration-fitted transfer from estimated frontal
   VMD ocular fields. Fit those frontal fields out of fold across at most four
   deterministic, whole 1024-sample calibration chunks. Use signed left/right/
   midline/common/lateral summaries; freeze central transfer before scoring.

Identity and the saved raw-frontal + direct-posterior output are reporting
controls. No parameter search or posterior refit occurs. Lags are -10/0/10,
ridge penalty .01. Frontal association threshold .4, mode association .2,
shared nonnegative field gain, iteration cap 2000 and tolerance 1e-6 remain
unchanged. The bounded calibration-context construction is a project
adaptation, not a reproduction of a published spatial method.

## Interfaces and correctness

- `nonfrontal_ridge.fit_frozen_ridge(calibration_eeg, predictors, boundaries)`
  returns fixed coefficients, means/scales and lag policy. Scoring EEG and
  paired targets are not accepted by its inference interface.
- `calibration_frontal_context` creates cached calibration-only modes and OOF
  artifact summaries. It never accesses a paired scoring target. Fewer than
  two eligible chunks or invalid context produces explicit pass-through.
- `regional_coverage` verifies parent hashes, identities and waveform parity;
  applies corrections only on masked region-2 channels; leaves frontal and
  posterior outputs bitwise equal to the parent and unknown rows unchanged.

Fit and gate each calibration independently. Lag exclusions respect every
trial/chunk join. All components estimate artifacts relative to original EEG;
the assembled waveform uses one subtraction, with disjoint masks. Save fitted
coefficients, selected chunk offsets, VMD vectors/residuals, quality flags,
predictions, per-example metrics and progress before moving to another record.

## Known risks and stop rules

Controlled mixtures scale the ocular field in EEG while retaining unscaled
donor HEOG/VEOG, including in clean controls. Frozen calibration coefficients
therefore face a deliberate amplitude mismatch. Do not fit scoring targets,
use stored mixture amplitude or silently rescale references to repair it.
Clean-input overcorrection is a possible measured failure. The fourth arm
tests whether estimated frontal activity supplies more useful strength context.

K3 remains a fixed starting recipe, not an optimum. Calibration-mode caches
are passive caches, not AutoVMD. Failed/iteration-limited decompositions remain
counted. The capped four-chunk calibration adapter has limited event coverage
and does not establish transfer over arbitrary recordings or participants.

All numerical fixtures, fitting, paper metrics, engineering gates and figures
run on Kaggle. Local checks inspect syntax, notebook schema and provenance.
Run the pilot first; retrieve and verify its tests, output manifests and frozen
branch parity before full execution. A numerical contract failure stops the
dependent run; scientific gate failures remain reported evidence rather than
triggering hidden threshold changes. No neural training or final qualification
follows automatically. Native preservation and independent evaluation remain
required after a promising controlled result.
