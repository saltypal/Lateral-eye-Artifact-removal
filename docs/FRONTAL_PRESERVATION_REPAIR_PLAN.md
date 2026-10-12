# Next bounded frontal repair — independent Sol planning recommendation

Planning only. Do not insert these changes into the frozen central-coverage
pilot. This follows its evidence review and requires a separate run specification.
Keep K3/alpha2000 fixed, AutoVMD stopped and all numerical execution on Kaggle.

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
