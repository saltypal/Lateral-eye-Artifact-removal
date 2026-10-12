# Next bounded frontal repair — independent Sol planning recommendation

Planning only. Do not insert these changes into the frozen central-coverage
pilot. This follows its evidence review and requires a separate run specification.
Keep K3/alpha2000 fixed, AutoVMD stopped and all numerical execution on Kaggle.

The immediate prerequisite is now the matched-context attribution experiment
in notebook 50. The full central-coverage result is verified at 13.162704 dB
whole-cap output SNR; it does not qualify the frozen frontal branch.

## Implementation readiness audit

The Luna audit identifies a minimal opt-in implementation: expose two signed
joint-ridge fields from `calibration_transfer`, using lag-major HEOG rows
`0::2` and VEOG rows `1::2`, and multiply both by the existing shared gain.
This is different from the already-tested independent-axis-gain method.
Preserve the default three-result API and current numerical outputs.
Add `AxisAmplitudeSupport.support_axes` while keeping `support` unchanged.
Per-axis threshold, dilation and taper retain the current 256/128 detector
and 41-sample extension. Require field-sum, zero-support, one-axis isolation,
unit/permutation, missing-context and original-input composition fixtures.

The audit also identifies a separate boundary question: native support is
currently computed inside each padded 1024-sample overlap-add window. A
whole-trial support track with a position-aware window callback may reduce
seam sensitivity. Do not introduce that change only into the separate-axis
arm, because it would confound this first support ablation. Keep the current
window policy matched for both arms first; declare whole-trial support as a
separate subsequent ablation if justified. Reflection and support must remain
inside each scoring trial; do not cross trial or calibration joins.

Native precedent: `frontal-native-monotonic-001`, source
`437198ca4d4ea4aff3003bb15775c8b29fb49ac9`, parent
`frontal-monotonic-events-001` at
`939fadf435440b3cac7436f03c3ce6d80dfa52aa`. The native precedent is a
development-exposed p02/p03 pilot, not a full native or independent cohort.
Pin corpus/2, readiness/1, review-contracts/1, controlled monotonic/1 and
the verified restore/1/source bundle. Check all required native waveforms
before execution. Old native admission compares against an earlier weaker
controlled comparator; a new exploratory stage must retain the present
calibrated-raw CC failure and must not silently call it qualified.

## Measured reason for the experiment

The native monotonic-support experiment reduced worst resting modification
from .456386 to .185130, but modification and covariance remain outside the
.01/.02 gates. Covariance is .127834; alpha .3600 dB and beta .1530 dB pass.
Rest is a preservation proxy, not proven artifact-free EEG.

Saved p02 rest trial 26/window 3 has association .43002056, shared gain
1.30478453, support .51344085 and axis detections [3,0]. The existing combined
OR support admits both reference-field contributions when only HEOG triggers.
This establishes the code mechanism; its contribution to the measured loss
still requires an ablation.

Evidence: `docs/FRONTAL_RESULTS.md` and the immutable
`frontal-native-monotonic-001` native gates/decisions. The user-requested
GPT-6.1 Sol extra-high planner inspected these saved artifacts read-only.

## Fixed mechanistic ablation before tuning

Compare five outputs: identity, current raw/shared-gain OR support,
raw/shared-gain separate-axis support, current VMD/shared-gain OR support,
and VMD/shared-gain separate-axis support. Retain lags -10/0/10, ridge .01,
association .4, mode association .2, gain bound 0–8 and the existing detector
window/hop/dilation/taper.

`artifact = shared_gain * (HEOG_support * HEOG_field + VEOG_support * VEOG_field)`

Expose both axis fields from the existing joint calibration fit; do not fit
independent axis gains. That already-tested alternative scored 15.396155 dB
versus 17.112814 dB for shared gain. Avoid repeating the saved negative
logistic, gain-scaled, whole-window and subspace-gain challengers.

Inference accepts original EEG, unchanged named references, frozen calibration
fits, and fixed frontal modes/residuals for the VMD arm. It accepts no targets,
true artifact, mixture strength or scoring-condition labels. Return fields,
supports, gain, association, triggers, artifact and failure flags. Keep
nonfrontal channels untouched and subtract the assembled artifact once.

## Conditional bounded selection

Only if the fixed ablation is informative, declare at most twelve recipes:
association {.4,.6,.8} × correction multiplier {.25,.5,.75,1}. Use the same
policy for matched raw/VMD pairs. Rest threshold fitting remains calibration
only; scored rest cannot determine event thresholds or change the candidate bank.

Reuse frozen recipient/donor buckets. Select global recipes on other buckets
with grouped inner validation; evaluate each outer bucket once. Predeclared
unscored calibration from a held-out source may fit its individual transfer,
but scored trials cannot select the global policy. Previously inspected p02/p03
remain development-exposed even with source-excluded predictions.

Preservation feasibility precedes SNR optimization. Keep the primary CC
comparator `calibration_raw_gain`: its controlled output SNR is 18.1932 dB,
CC .981142. Do not restore the weaker direct-regression comparator merely
because the old frontal report called a candidate passed. Report matched-policy
raw controls in addition to the fixed primary comparator.

## Tests and stop rules

- Both supports one reproduce current correction; both zero give exact identity.
- HEOG-only support admits no VEOG contribution; test the reverse and axis permutation.
- Verify units, boundaries, failed modes, anatomy/masks, overlap-add and single subtraction.
- Save fields, supports, gain, decisions and metrics per trial before advancing.
- Stop dependent work on numerical, parity, interface or provenance failure.
- No feasible joint accuracy/preservation recipe means no feasible repair; retain
  that measured conclusion without weakening gates or declaring the least-bad arm qualified.

Controlled mixtures have unchanged-reference amplitude mismatch. Native EEG
has no paired reconstruction SNR. No models or reserved evaluation follow
automatically from this exploratory ablation.


## Implementation snapshot: 13 October 2026 IST

The fixed native mechanism test is authored as notebook51 and stage
`frontal-native-separate-axis`. Numerical validation is pending Kaggle execution.
No correction/detection parameter search is included.

`calibration_transfer(..., return_axis_fields=True)` exposes two contributions
from the same joint ridge coefficients, multiplied by the same shared scalar
gain and restricted to the same triggered usable frontal channels. Its default
three-value return and numerical path remain unchanged. The opt-in mode is
restricted to shared gain; this does not repeat independent axis-gain fitting.
`AxisAmplitudeSupport.support_axes()` thresholds, dilates and tapers each axis
independently. Existing OR-support code remains unchanged.

The exploratory stage pins the previous controlled monotonic and native pilot
manifests and their source SHAs. It accepts unqualified evidence for a repair
experiment and records qualification false. It does not weaken the existing
qualified-teacher checkpoint or treat the stronger current CC comparator as
passed. Pilot-only execution requires the same two development recordings.

Five scored arms: identity, raw OR support, raw separate support, VMD OR support
and VMD separate support. Joint coefficients, shared gain, calibration,
thresholds, K3/2000 and per1024-window support/overlap-add policy remain fixed.
Identity and both OR arms must reproduce all predecessor trials within
rtol1e-7/atol1e-12; channel metadata must match and nonfrontal channels remain
bitwise unchanged. Trial preprocessing remains independent. Save every
window's original EEG, references, two raw/VMD axis contributions, separate
supports and union support, plus full trial predictions and decisions.

Fixtures cover default-path parity, sum reconstruction, reference-axis
permutation, original units, masks, failed modes, all-one/all-zero supports,
one-axis exclusion, malformed inputs and source/checksum/qualification scope.
Cancelling fields are tested explicitly: equal/opposite contributions can
cancel with both supports present, while one-axis support destroys that
cancellation. Separate gating is not assumed to lower modification.

Native reports retain unchanged-reference EEG–HEOG/VEOG correlation, resting
RMSE, spectrum, modification and covariance; native reconstruction SNR is not
reported. A native benefit still requires a matched controlled CC check before
qualification or downstream model development.


Pre-launch GPT-6.1 Sol extra-high static review found no remaining launch
blocker after fixing the notebook host-kernel inspection import. Axis fields
already include the intervention trigger and shared gain; they are signed
correction contributions, not independently identified pure ocular sources.
Reduced resting correction alone is insufficient: compare matched suppression
and preservation per recording. Two development-exposed recordings do not
establish population safety, and the controlled CC failure remains unresolved.
