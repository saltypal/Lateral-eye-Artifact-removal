# Explain the frontal implementation

## The signal we estimate

For each frontal channel, contaminated EEG is a retained EEG reference plus an
ocular contribution. Our estimator sees EEG, observed HEOG/VEOG and unscored
calibration. It does not see a paired clean target or true simulated artifact.
Targets appear only when scoring a completed prediction.

The corrected signal is always the original minus one estimated artifact. We
do not sum selected retained modes as the final EEG: that would discard the
decomposition residual and can discard neural signal outside those modes.

## What a VMD mode is

A mode is a real-valued vector with one value per sample, not a label such as
delta or alpha. For a 1024-sample channel, K3 yields an array `[3, 1024]`. VMD
iteratively estimates narrow-band vectors and their center frequencies. Alpha
controls its bandwidth penalty. We retain `residual = original - sum(modes)`.
Modes plus residual therefore reconstruct the original within the numerical
contract even when finite-penalty VMD does not fit every sample exactly.

There are two separate fixed arms, K3 and K4 with alpha2000. They are project
starting/comparison settings, not a measured global optimum. Five conventional
EEG frequency bands do not justify choosing five adaptive modes. Centers,
bandwidths, overlap and convergence are inspected; nearby centers alone do not
establish failure. No K/alpha optimization is active.

## Detect, select, estimate, subtract

1. `frontal_indices` verifies electrode names, declared anatomy and masks.
   A guessed frontal position is rejected; nonfrontal rows are passed through.
2. The outer detector measures association with signed HEOG/VEOG or uses the
   separately fitted EEG-only SVM adaptation. Its fitting excludes both heldout
   recipients and donors. The SVM amplitude feature uses input/calibration SD,
   preserving its interpretation across volts and microvolts.
3. Component selection measures each mode/source's HEOG and VEOG evidence.
   A high-correlation mode may still contain neural activity; correlation is
   evidence of ocular association, not proof the entire mode is artifact.
4. Projected correction estimates only the reference-predictable contribution
   inside selected components. Whole-mode correction is a matched challenger,
   not an assumed safe default. SOBI additionally separates the mode matrix
   using delayed second-order covariance structure before correction.
5. Interval correction multiplies the estimate by separate calibration-derived
   HEOG/VEOG support. The current 3-SD support threshold is an explicit project
   adaptation; insufficient temporal coverage can leave artifact. Inspection
   measures this against known simulated artifacts without fitting that truth.
6. The artifact is subtracted from original EEG once. Decomposition/SOBI failure
   returns scored pass-through and a visible failure record.

The declared short-lag banks address delayed ocular leakage. A lag projection
cannot read across trial joins. Calibration-anchored offset correction is
currently a diagnostic, not a secretly added inference recipe: zero-mean
projection can leave ocular mean offsets and inflate residual error.

## Important files and flow

### Calibrated signed-field VMD hybrid

`frontal_calibration_transfer.py` fits a six-row lagged HEOG/VEOG ridge model
on unscored calibration EEG. The lag-major rows represent HEOG and VEOG at
-10, 0 and +10 samples. Each frontal channel receives signed spatial transfer
coefficients, with calibration trial joins excluded from fitting.

On a scoring window, those frozen coefficients predict an ocular field. The
shared-gain arm estimates one bounded gain from eye-associated K3 mode vectors.
The original EEG still defines intervention support, and the final correction
subtracts the estimated field once from the original. It does not replace EEG
with the sum of retained modes. Its matched raw arm estimates the same gain
without VMD, with otherwise identical references, calibration and gates.

The signed-axis challenger splits the calibrated field into horizontal and
vertical contributions and estimates two montage-wide gains. RMS normalization
and ridge regularization stabilize the two-variable fit. It changes correction
gain estimation, not K/alpha or source partitions. All variants are project
extensions, not exact author reproductions.

The shared VMD hybrid measured 17.112814 dB controlled development SNR; its
matched raw arm measured 18.193247 dB. Thus VMD has not yet demonstrated an
advantage. The recipient/donor-aware approximate interval for the VMD mean is
14.45-19.75 dB, and severe lateral movement at -5 dB input remains 8.95 dB.
The regression-generated mixture construction favors calibrated transfer.
Real OSF suppression and resting-EEG preservation must be checked separately.

| File | Responsibility |
|---|---|
| `frontal_campaign.py` | Frontal-first authorization, fixed settings and parent checkpoints |
| `frontal_methods.py` | Region guards, VMD wrapper, component evidence, entropy and whole-cap estimator |
| `frontal_replication.py` | Passive fixed-cache import, declared correction bank and source-excluded selection |
| `frontal_evaluation.py` | Frozen recipes on all eligible development sources; no retuning |
| `frontal_evidence.py` | First-example diagnostics and parity between reusable inference and saved scoring |
| `frontal_review.py` | Complete-row/finite-metric engineering gates and readable controlled review |
| `frontal_native.py` | Original OSF trials, unchanged evaluation EOG, overlap-add and record recovery |
| `tools/frontal_queue.py` | Local submission/retrieval/status orchestration; no numerical EEG execution |

Notebook 24 validates contracts; 25 is the smaller comparison; 28 screens the
correction bank; 29 scores frozen recipes; 27 inspects mechanisms; 26 validates
native OSF. Every notebook obtains fresh Git code, pins its resolved SHA and
consumes saved versioned parents. Notebook variables never serve as parent artifacts.

## Cost and regional boundary

Single-channel VMD requires Fourier transforms and repeated K-mode spectral
updates: approximately `O(T log T + I K T)` per window, with iteration count I.
The bounded-memory solver retains current/previous arrays rather than the
entire iteration history. C frontal channels scale approximately linearly in C;
cache-assisted experiment timing excludes cache generation and is not runtime
inference timing. Component regression and SOBI add their own measured cost.
End-to-end CPU memory/runtime at 10-50 channels remains a later measurement.

The present public research wrapper accepts full-cap EEG but corrects frontal
rows only. Posterior correction is pending separate validated integration.
Eventually its artifact will be estimated from original posterior EEG and
calibration with optional common/signed frontal and frontal-artifact context.
The region masks must be disjoint, and the combined artifact is subtracted once.
We have not established that additional posterior correction is beneficial.

The eventual EEG-only student will learn paired denoising and may use a
qualified cross-fitted teacher. It will not require runtime VMD or EOG. No
qualified teacher or exported neural model exists for this new campaign yet.
