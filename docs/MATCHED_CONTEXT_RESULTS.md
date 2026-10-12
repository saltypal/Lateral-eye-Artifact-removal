# Matched frontal context results

## Verified pilot — 13 October 2026 IST

Run `context-match-pilot-001`, campaign `vmd-context-match-20261013`,
Kaggle `satyapaladugu/vmd-matched-context-pilot/1`, exact published source
`c8159e17ba857cee2c0bc08284d98c5931fc4fed`. Manifest SHA256
`670cf0577b61c9ce30e39c90b72517ce9cc4060b4de72c8f2776c27c0223f6ce`.
Completion state and 26 critical payloads checksum-verified before interpretation.
The separately tracked waveform/fit receipt records expanded recovery.

180 tests passed, 14 dependency warnings; 150 frozen regional checks,
300 preserved parent predictions and 171,600 preserved metric checks passed.
Five raw-context models were available across 50 examples/five recipients.
Seven arms are saved; only the raw context arm is newly fitted.
All numerical processing and uncertainty calculations ran on Kaggle.

| Context arm | Whole output SNR dB | Central/shared SNR dB | Whole CC |
|---|---:|---:|---:|
| frozen_regional | 10.002235 | 2.323554 | 0.869356 |
| central_frontal_vmd_context | 13.994230 | 11.121655 | 0.945402 |
| central_frontal_raw_context | 13.938976 | 11.002471 | 0.945204 |

Frontal and posterior scored predictions are the same archived VMD/direct
outputs in both context arms. They were not replaced by raw frontal outputs.
Thus this compares central context representation, not two different complete
frontal correction methods. Calibration chunks, joins, transfer parameters
and signed feature order match; no new VMD solver ran.

## Paired uncertainty and interpretation

The table reports candidate minus comparator. This is the declared project
recipient/donor Poisson multiplier approximation, not a paper-native metric or
a reproduced significance procedure. Accuracy excludes clean controls; finite
failure/pass-through observations remain in the complete cohort.

| Scope | Candidate | Comparator | Output SNR difference dB | 95% interval dB | Valid draws |
|---|---|---|---:|---|---:|
| whole | central_frontal_raw_context | frozen_regional | 3.936741 | [3.249903, 5.044596] | 1835/2000 |
| central_shared | central_frontal_raw_context | frozen_regional | 8.678918 | [6.999791, 11.253330] | 1835/2000 |
| whole | central_frontal_raw_context | central_frontal_vmd_context | -0.055253 | [-0.195391, 0.012489] | 1835/2000 |
| central_shared | central_frontal_raw_context | central_frontal_vmd_context | -0.119183 | [-0.420841, 0.027859] | 1835/2000 |

The raw-versus-VMD interval includes zero in both scopes. VMD's small pilot
advantage does not establish a benefit. Both are materially better than central
pass-through in this development experiment. Do not compare pilot averages
with the original 32-recipient means.

The previous full coverage table was also analyzed on Kaggle without repeating
any fits. It contains 32 recipient sources but only five ocular donor sources;
this limits external generalization and the uncertainty method remains approximate.

| Full coverage scope | VMD context minus frozen baseline dB | 95% interval dB | Valid draws |
|---|---:|---|---:|
| whole | 4.068490 | [3.208717, 4.897776] | 1991/2000 |
| central_shared | 8.949300 | [6.964664, 10.766403] | 1991/2000 |

## Preservation, failures and limits

Both context arms leave controlled clean inputs exactly unchanged, with zero
clean alpha/beta and covariance errors. Both retain the failed frozen frontal
CC gate. Native preservation remains unresolved, and the whole target of15 dB
is unmet. `qualification=false`; no neural models or reserved sources opened.

Inherited whole failure-labelled rows are56 per arm, central26, frontal13 and
posterior17. They remain scored, not a successful-only subset. All five new
raw fits were available; inherited labels are distinguished from new failures.
The donor references remain unscaled while controlled EEG ocular strength
changes. No target or recipe amplitude entered correction. The LEMON recipient
is low-ocular reference EEG, not proven artifact-free truth.

Pilot numerical runtime85.652443 seconds; tests18.62 seconds. Exact runner
wall time, peak memory, framework versions and parent identities remain in the
saved run artifacts. The full same-recipe comparison follows numerical/parity/
persistence verification. Native shared-gain/separate-axis support remains the
next safety experiment after this attribution review.
