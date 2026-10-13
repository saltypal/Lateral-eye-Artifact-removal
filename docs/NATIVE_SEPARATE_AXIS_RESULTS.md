# Native frontal separate-axis support: measured development results

Verified13 October2026 IST. Run `frontal-native-separate-axis-001`, Kaggle
`satyapaladugu/vmd-frontal-native-separate-axis/1`, source
`ce6f31d672f5061cfaae44a22c2318e4e4afc351`, manifest
`63055a6fa7ed3c17b767e6ba64e0f7ad845b5d40af0afe0b78d59da7c83ca049`.

## Numerical completion and scope

202 tests passed,14 dependency warnings,23.03 seconds. Two development-exposed
OSF recordings (p02/p03),51 scoring trials, five arms,153 predecessor
prediction checks. Original EEG/references and identity/raw-OR/VMD-OR trial
outputs reproduce the predecessor. Nonfrontal channels remain exactly
unchanged. Eight inherited failure entries remain; failed decompositions use
pass-through. Runtime264.612103 seconds, runner594.289838 seconds, peak
resident memory396,708 KiB. No AutoVMD, posterior VMD, neural training, new
parameter search, scoring-label access at inference or confirmation opening.

Source SHA, run identity, completion and scope checks passed. Critical72
payloads are checksum-verified. Full recovery now contains51 trial predictions,306 axis-window bundles and
51 mode-example bundles; all495 manifest payloads, including protocol and spectrum artifacts, verified.
The receipt is retained under the run directory.
Luna monitoring was interrupted by an account usage-limit error. The primary
rechecked the live terminal status and recovered outputs; do not claim
continuous monitoring during that interruption. This was not a Kaggle failure.

## What changed

Same joint HEOG/VEOG calibration fit, montage-wide shared gain, threshold.4,
lags[-10,0,10], ridge.01, fixed frontal K3/2000 and1024/512 overlap-add.
Only time support changed: the old policy admits both signed field
contributions whenever either axis triggers; the new policy gates each axis
independently. Coefficients are not separate independently identified ocular
sources. Existing OR support and default calibration-transfer output remain
unchanged. Neither targets nor native scored condition labels enter inference.

## Resting preservation

Worst-record mean values, exactly as reported by Kaggle. Relative modification
is a ratio;0.158564 corresponds to15.8564%. Gates are engineering constraints
separate from paper-native measures. Native rest is not proven artifact-free.

| Arm | Relative modification | Alpha error dB | Beta error dB | Relative covariance error | All preservation gates |
|---|---:|---:|---:|---:|---|
| VMD OR support | 0.185130 | 0.360004 | 0.152970 | 0.127834 | fail |
| VMD separate support | 0.158564 | 0.220841 | 0.109023 | 0.108051 | fail |
| Raw OR support | 0.220164 | 0.403878 | 0.165964 | 0.137110 | fail |
| Raw separate support | 0.186919 | 0.249230 | 0.116666 | 0.115809 | fail |
| Identity | 0 | 0 | 0 | 0 | pass preservation only |
| Declared limit | 0.01 | 0.5 | 0.5 | 0.02 | all required |

Every active arm fails modification and covariance. Alpha/beta errors pass.
Identity passing preservation does not qualify artifact suppression.

| Recording | VMD OR modification | VMD separate modification | VMD OR covariance | VMD separate covariance |
|---|---:|---:|---:|---:|
| p02 | 0.091911 | 0.082373 | 0.033841 | 0.026056 |
| p03 | 0.185130 | 0.158564 | 0.127834 | 0.108051 |

## Paper-native unchanged-reference correlation

Ordinary absolute EEG–VEOG correlation for blink and vertical, EEG–HEOG for
lateral. These are suppression proxies, not clean reconstruction accuracy or
SNR. Lower after-correlation alone does not prove neural preservation.

| Recording/condition | Before | VMD OR after | VMD separate after | Raw OR after | Raw separate after |
|---|---:|---:|---:|---:|---:|
| p02 blink | 0.913125 | 0.051475 | 0.062773 | 0.048048 | 0.059778 |
| p02 lateral | 0.273208 | 0.024695 | 0.035932 | 0.058827 | 0.061349 |
| p02 vertical | 0.455667 | 0.070346 | 0.068313 | 0.065001 | 0.062247 |
| p03 blink | 0.958671 | 0.105604 | 0.105517 | 0.044100 | 0.044041 |
| p03 lateral | 0.422304 | 0.045239 | 0.041308 | 0.070062 | 0.060060 |
| p03 vertical | 0.685271 | 0.144703 | 0.144195 | 0.164394 | 0.162117 |

Separate support improves resting preservation in both recordings but worsens
p02 blink/lateral suppression slightly. p03 blink is nearly unchanged; lateral
and vertical improve. It is a trade-off, not a uniformly superior replacement.

Rest RMSE retains original unverified source array units:

| Recording | VMD OR | VMD separate | Raw OR | Raw separate |
|---|---:|---:|---:|---:|
| p02 | 1.196495 | 1.087685 | 1.453975 | 1.312668 |
| p03 | 1.710040 | 1.583876 | 1.956633 | 1.781565 |

Do not label these numbers microvolts without verified source scaling.
Per-trial, per-channel, participant, spectrum and permutation tables remain in
saved artifacts. Two exposed participants cannot establish population safety.

## Decision and next diagnostic

Retain separate support as an unqualified ablation; do not integrate it or
expand its parameter settings merely because some preservation values improve.
The current controlled frontal CC failure and whole15 dB shortfall remain.
This native experiment does not change controlled SNR17.112814 frontal or
13.162704 whole for the VMD-context hybrid.

Next, diagnose the residual resting interventions from saved window fields,
supports and decisions on Kaggle. Reconstruct the five archived outputs without
fitting or VMD, confirm parity, and report association/trigger/support/cancellation
per resting window and record. This describes admission, gain and cancellation patterns before proposing a
repair. Unpaired resting EEG cannot prove field-estimation error, true ocular
contamination or neural loss. No native scored rest is used to select a global recipe.


Diagnostic reporting distinguishes detector-only axis supports from triggered
nonzero-field interventions. Cancellation averages exclude zero-field windows;
the number of contributing windows is saved explicitly. Relative correction
averages exclude zero-input-energy windows with available counts. Overlapping
padded windows are descriptive, not independent population observations.

The diagnostic saves padded-window start and original-trial overlap length.
All cached windows enter output reconstruction, but means/ranked examples
exclude windows contributing zero original samples. Window energy fractions
still describe the full padded window and are not overlap-weighted trial
modification; they cannot be added to or substituted for record preservation.


## Diagnostic launch

`native-axis-diagnostic-001`, Kaggle `satyapaladugu/vmd-native-axis-diagnostic/1`,
published source `5165ee3443297b69ea003c727438663ce62247ba`.
Three versioned parents: native separate-axis/1, readiness/1 and unchanged
frontal fixture/1. No raw dataset attachment. Sol extra-high static review
found no remaining blocker after intervention-count and padded-overlap fixes.
Numerical tests and255 reconstruction checks are pending. Primary monitors
status every45 seconds and retrieves outputs through the bounded downloader.


## Verified cached intervention diagnostic — 13 October 2026

Run `native-axis-diagnostic-001/1`, source
`5165ee3443297b69ea003c727438663ce62247ba`, manifest
`6dae496677ee4ffeb1ab1034a5bfd4676d2bc1dccf694a4e6368e80476fe4b36`.
208 Kaggle tests passed (14 warnings);51 trials and255 exact-family output
parity checks passed. Sixteen retrieved payloads are checksum verified. Diagnostic
runtime12.718s; runner44.121s; peak267,984KiB. No fitting or prediction change.
612 representation-window rows were saved;510 overlap original samples and102
are padding-only. All enter reconstruction; descriptive means exclude the102.

| Record / representation / rest proxy | Contributing windows | Original association-admitted windows | Separate nonzero interventions | Mean OR fraction | Mean separate fraction |
|---|---:|---:|---:|---:|---:|
| p02 raw |45|31|10|0.075217|0.068522|
| p02 VMD |45|31|10|0.061602|0.056295|
| p03 raw |45|45|16|0.162544|0.133332|
| p03 VMD |45|45|16|0.133517|0.110656|

These fractions describe full padded-window correction energy norms, not
record-level modification. Overlapping windows are dependent. Detector activity
is distinct from a nonzero admitted correction. The original paper metrics and
engineering gates are unchanged by this diagnostic.

The largest contributing-window resting VMD example is p03 trial37/window1:
association0.811355, shared gain0.767750, HEOG support0, VEOG support0.782972.
OR/separate fractions0.771501/0.770130. Removing a quiet HEOG contribution barely
changes this dominant VEOG intervention. An association threshold0.8 would still
admit this window. The plotted trace has prominent slow transients, but unpaired
rest cannot establish genuine ocular contamination, neural content, or label error.
Do not relabel it to make the preservation gate pass.

For p02 trial34/window2, both supports overlap; the plot shows nearly identical
OR/separate correction during that interval. Separate support does not solve
correction where both detectors admit the fields.

Resting mean signed cancellation fractions are negative (-0.028209 p02,
-0.019581 p03): the mean cross term is reinforcing, not cancelling, under the
saved convention `-2<HEOG,VEOG>/(E_HEOG+E_VEOG)`. Nonzero-field denominators are
31 and45 windows respectively. This descriptive mean does not exclude individual
cancelling windows, establish field purity, or quantify trial-level loss.

Next: the twelve-setting bounded correction feasibility screen in
[FRONTAL_BOUNDED_FEASIBILITY_PLAN.md](FRONTAL_BOUNDED_FEASIBILITY_PLAN.md).
It freezes K3/2000 and all calibration/detector parameters, reports all native
candidates and selects no global recipe on scored rest. Qualification stays false.
