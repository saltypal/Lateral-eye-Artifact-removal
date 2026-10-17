# Fixed frontal VMD contribution on repaired coupled data

## Evidence and scope

Run `coupled-frontal-review-002`, Kaggle notebook
`satyapaladugu/vmd-fixed-frontal-coupled-review/2`, source
`7f098ff09e74bb154c622d581543316a715ff577` completed all 20 records.
The numerical suite passed 318 tests with 14 warnings in 20.73 seconds;
experiment runtime was 29.821923 seconds, excluding setup/tests/retrieval.
All 95 manifest payloads were downloaded and checksum-verified, with completed
execution and source identity verified. Manifest SHA256:
`6a8393ef23d9f54639caa54d6e6d11408e07ac56edaea0f510897d74adbe33d1`.

Parent: verified `coupled-pilot-002/2`. Two development recipient/donor pairs,
two idealized clean controls and 18 blink/lateral/mixed contaminated examples
at -5/0/+5 dB. These are engineering results, not independent generalization or
native qualification. Regression-derived generation bias remains. No reserved
source, AutoVMD search, trained neural weights or teacher qualification.

## Matched results

Reported contaminated means exclude idealized perfect-reconstruction clean
controls. Channel-first values average channel SNRs within records; the second
SNR column averages whole-scope pooled-energy SNR across records. It is not a
single energy ratio pooled over the entire corpus. This balanced pilot has the
same conditions and input levels for each recipient.

| Scope and method | Channel-first output SNR dB | Mean pooled-window SNR dB | MSE V² | Pearson CC |
|---|---:|---:|---:|---:|
| Whole: frontal ridge + nonfrontal ridge | 18.921014 | 16.193570 | 1.000550e-12 | 0.980987 |
| Whole: frontal VMD + identical nonfrontal ridge | 19.398750 | 15.848823 | 1.069891e-12 | 0.979788 |
| Frontal: direct ridge | 20.973930 | 20.488443 | 4.343346e-13 | 0.992797 |
| Frontal: VMD-informed gain | 23.040945 | 21.768334 | 7.376812e-13 | 0.987553 |
| Posterior: identical ridge in both hybrids | 15.527561 | 14.744282 | 1.966439e-12 | 0.972129 |
| Central/shared: identical ridge in both hybrids | 20.258384 | 20.342242 | 5.954600e-13 | 0.981410 |

Higher mean channel-log SNR can coexist with higher mean squared error because
the summaries weight errors differently. The VMD arm does not win all metrics.
These figures must not be substituted for fresh controlled acceptance evidence.

## Lateral preservation failure

For recipient `lemon:sub-032306`, lateral input -5 dB, frontal channel-first
output SNR was 8.812491 dB with VMD versus 13.486290 dB with raw ridge.
Frontal CC was 0.917674 versus 0.973957. The saved CC deterioration was
0.0562836, exceeding the unchanged 0.005 limit. Other lateral records also fail;
neither the whole-cap average nor blink improvements authorize qualification.

Both idealized zero-driver clean controls pass the recorded clean checks for
the VMD hybrid. This does not demonstrate preservation on native resting EEG.

At severe lateral input, saved shared frontal gains were:

| Recipient | Raw gain | VMD-informed gain |
|---|---:|---:|
| sub-032305 | 1.037937 | 0.971926 |
| sub-032306 | 1.030755 | 0.863411 |

These are observed-input estimates, not comparisons to an oracle gain. Mode
selection varies by channel; the first saved frontal channel of sub-032306
selected only mode 0 in that severe lateral record. No universal ocular-mode
label is justified.

## Next focused comparison

The current VMD gain uses selected-mode activity in the numerator against a
full-field energy denominator. This gives a plausible mechanism for reduced
gain when part of the ocular field falls outside the selected-mode span,
including the residual. It is not yet a demonstrated complete error explanation.

Test existing `calibration_transfer(..., gain_mode="mode_projection")` against
the current shared-gain VMD arm and raw ridge, with the same input, mode cache,
calibration and nonfrontal outputs. Its numerator and denominator use the same
selected-mode subspace. Do not select branches with scored condition labels or
paired targets, change K/alpha, or claim the modification works in advance.
Inspect residual/reference association and temporal error before introducing an
additional interval gate or independently fitted HEOG/VEOG gains.

Native OSF validation and broader grouped development remain necessary after
this engineering comparison. Neural training remains closed.

## Figure inspection

`fixed_frontal_snr.png` was visually inspected after retrieval. Axes, labels and
the two aggregation legends are readable without overlap. The figure shows
contaminated means only; it does not display the condition-specific preservation
failures and must be accompanied by those tables. CSVs and saved decisions are
the numerical authority.
