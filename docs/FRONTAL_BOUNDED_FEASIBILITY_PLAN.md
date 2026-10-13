# Fixed frontal correction feasibility screen

13 October 2026. Mode: Super Code; primary implements, Sol extra-high reviews.

The cached native diagnostic reproduced 255 outputs and passed 208 tests. Axis
separation reduces some resting interventions but does not pass preservation.
The next question is narrow: can any of the already declared twelve correction
settings satisfy native resting preservation on the two exposed pilot records?

## Scope and inputs

Freeze K=3, alpha=2000, joint calibration coefficients, shared gains, mode
selection, detector thresholds, trial preprocessing and overlap-add. Attach
native separate-axis pilot/1, readiness/1 and frontal review contracts/1.
Verify the exact pilot manifest and source SHA. No raw sources, fitting, VMD,
neural work, reserved data or posterior changes enter this screen.

The bank is association {0.4,0.6,0.8} × multiplier {0.25,0.5,0.75,1.0}.
Each setting has matched raw/VMD representations and OR/separate support:
48 active arms plus identity. This is twelve settings, not a new K/alpha grid.
Cached axis fields already include admission at0.4 and shared gain. Raising
admission is exactly reproducible by zeroing rejected windows; lowering it is
unsupported and must raise an error. Failed channels remain zero fields.

For each window, assemble signed fields with the existing support, apply the
new admission and multiplier, then reuse the original trial overlap-add and
single subtraction. Baseline0.4/1 arms and identity must reproduce the five
saved outputs:255 checks. Verified frontal masks stay fixed; all other rows
must remain bitwise unchanged. Save every full-trial prediction and decision.

## Reports and acceptance

Use the parent's explicit resting-trial identities for post-inference reporting
only. Save trial, record and worst-record modification, alpha/beta and covariance
with the unchanged .01/.5dB/.02 thresholds; save rest RMSE in source array units.
Save ordinary signed/absolute HEOG and VEOG correlations per trial/channel,
including both before and after. Authoritative labels are retained in the
verified native_frontal_sources.jsonl ledger. Use them after prediction to
report complete-condition concatenated ordinary correlations:HEOG for lateral,
VEOG for blink/vertical, original EEG for resting RMSE. Require40 baseline
record/condition metric parity rows. Labels never enter correction or selection.
Preserve the unchanged parent reports. No significance/winner claim on two records.
No native reconstruction SNR is reported. Add a preservation trade-off figure.

This is an exhaustive feasibility screen on development-exposed records, not
grouped recipe selection or independent validation. Report all candidates;
do not select a global recipe from scored native rest. A native pass is only a
necessary pilot constraint. Controlled grouped accuracy/preservation and full
native cohort validation remain required before qualification.

If no active candidate passes, report the bounded repair as infeasible under
these gates and stop its promotion. Identity passing does not establish cleaning.
If a candidate passes, prepare source-separated controlled testing of the fixed
bank. Existing controlled caches support OR replay; separate-axis replay needs
one-time field extraction from archived modes/calibration with unchanged-output
parity, never guessed decomposition of a joint artifact.

## Modular implementation and validation

| Component | Input | Output | Owner | Acceptance |
|---|---|---|---|---|
| Cache replay | signed fields, supports, shared decisions | 49 window artifacts | Primary | frozen bank, finite/axis/decision checks |
| Trial runner | exact parent trial/windows | predictions and decisions | Primary |255 baseline checks, nonfrontal identity |
| Reports | scored predictions + reporting-only rest keys | full candidate tables/gates | Primary | no omissions, unchanged limits, no SNR claim |
| Notebook53 | pinned Git SHA/versioned parents | recoverable Kaggle run | Primary | tests pass before replay, hash retrieval |
| Review | code and artifacts | scope/interpretation audit | Sol | no selection leakage or qualification claim |

Numerical fixtures cover threshold boundaries, unsupported lower thresholds,
scalar attenuation, signed cancellation, one-axis support, invalid decision
metadata, failed fields, nonfrontal pass-through, original-subtraction and OLA.
All numerical execution is on Kaggle; local work is static authoring only.
