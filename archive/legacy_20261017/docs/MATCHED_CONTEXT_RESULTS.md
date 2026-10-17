# Matched frontal context results

## Verified pilot — 13 October 2026 IST

Run `context-match-pilot-001`, campaign `vmd-context-match-20261013`,
Kaggle `satyapaladugu/vmd-matched-context-pilot/1`, exact published source
`c8159e17ba857cee2c0bc08284d98c5931fc4fed`. Manifest SHA256
`670cf0577b61c9ce30e39c90b72517ce9cc4060b4de72c8f2776c27c0223f6ce`.
Completion state and 26 critical payloads checksum-verified before interpretation.
Expanded recovery is complete: all50 prediction NPZs,50 decision JSONs and
five raw-context ridge fits are local, with136 manifest-listed payloads verified.

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

| Scope | Candidate | Comparator | Output SNR difference dB | Approximate 95% percentile interval dB | Valid draws |
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

| Full coverage scope | VMD context minus frozen baseline dB | Approximate 95% percentile interval dB | Valid draws |
|---|---:|---|---:|
| whole | 4.068490 | [3.208717, 4.897776] | 1991/2000 |
| central_shared | 8.949300 | [6.964664, 10.766403] | 1991/2000 |

## Preservation, failures and limits

Both context arms leave controlled clean inputs exactly unchanged, with zero
clean alpha/beta and covariance errors. Both retain the failed frozen frontal
CC gate. Native frontal preservation failed on the tested two development recordings;
native central transfer remains untested. The whole target of15 dB
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

## Full run launch

After the numerical/parity/persistence checks, the unchanged recipe was
submitted as `context-match-full-001`,
Kaggle `satyapaladugu/vmd-matched-context-full/1`, exact latest published
source `c285e5398ba5cb31a3cf2163b81c48f2b767bf7e`.
Only documentation changed after the pilot numerical source; recipe and
implementation are unchanged. Full320-example/32-recipient results completed and were verified on 13 October 2026 IST; see the full result below.
No full result is inferred from the five-recipient pilot.

## Independent interpretation review

GPT-6.1 Sol with extra-high reasoning checked the saved pilot and prior full
uncertainty tables read-only. It found no numeric mismatch. An interval
containing zero does not establish equivalence or prove that VMD is unnecessary;
both context arms keep VMD-scored frontal outputs. Five donors remain the
ocular-source replication limit despite32recipients/320mixtures. Valid
bootstrap draw counts are resampling counts, not additional experiments.
The saved `favors_candidate` flag describes the point-estimate direction,
not a significance test. Qualification remains false.


## Full matched comparison: completed and verified

Run `context-match-full-001`, Kaggle `satyapaladugu/vmd-matched-context-full/1`,
source `c285e5398ba5cb31a3cf2163b81c48f2b767bf7e`. Full 320 examples,
32 recipient sources, five ocular donors, seven arms. All 180 tests passed
(14 dependency warnings,20.63 seconds),960 frozen-branch parity checks,
1,920 preserved parent predictions and 1,098,000 archived metric checks passed.
All 32 new raw context fits were available. No new VMD solver, AutoVMD,
model fitting or reserved-source access occurred.

Primary means are channel-first, condition/input balanced within recipient,
then recipient balanced. These are development-exposed controlled results.

| Representation for central transfer | Whole output SNR dB | Central output SNR dB | Whole MSE | Whole temporal RRMSE | Whole Pearson CC |
|---|---:|---:|---:|---:|---:|
| Central pass-through/frozen regional | 9.094213 | 1.954134 | 2.969267e-11 | 0.628874 | 0.859318 |
| VMD frontal artifact context | 13.162704 | 10.903434 | 1.323511e-11 | 0.338343 | 0.934641 |
| Raw frontal artifact context | 13.183201 | 10.949306 | 1.359897e-11 | 0.339202 | 0.934836 |

Both context arms retain the SAME scored VMD frontal output 17.112814 dB and
posterior direct-regression output 13.541703 dB. The new raw context arm does
not remove VMD from the frontal scored correction.

| Difference, raw candidate minus comparator | Whole SNR difference dB | Approximate 95% percentile interval dB | Central SNR difference dB | Approximate 95% percentile interval dB |
|---|---:|---|---:|---|
| Frozen regional baseline | 4.088988 | [3.131588,4.957090] | 8.995172 | [6.862256,10.887332] |
| VMD frontal context | 0.020497 | [-0.089695,0.159625] | 0.045872 | [-0.193234,0.352235] |

Intervals use the predeclared project recipient/donor Poisson multiplier
approximation,1991/2000 valid draws, seed 42; they are not the prescribed
paper-native permutation procedure. Five donors restrict ocular-source
replication. Do not claim a raw or VMD context winner or equivalence. Raw has
slightly higher mean SNR/CC but slightly worse MSE/RRMSE; the SNR interval
includes zero. Both improve coverage compared with central pass-through.

### Condition and input-level results

Raw-context whole output SNR (dB), declared mixture input level:

| Condition | -5 dB | 0 dB | +5 dB |
|---|---:|---:|---:|
| Blink | 12.718306 | 16.122024 | 19.774689 |
| Lateral | 4.369782 | 8.616932 | 12.560167 |
| Mixed | 11.249659 | 14.740758 | 18.496492 |

The lateral condition remains the accuracy bottleneck. Declared input levels
are pooled construction settings; measured channel-first SNR improvement is
reported separately in saved tables and is not output SNR minus that label.

### Preservation, failures, cost and recovery

Both context arms have zero controlled clean modification, alpha/beta error
and covariance error. The inherited frontal CC deterioration 0.007441477
exceeds 0.005 against `calibration_raw_gain`; neither system qualifies.
Native frontal preservation failed on the two previously tested development
recordings. Native central transfer remains untested. Whole SNR 13.183201
and 13.162704 both miss 15 dB. No teacher/model/freeze/export qualification.
Inherited failure-labelled observations remain scored:170 whole,78 central,
39 frontal and 53 posterior per context arm. They are not 32 new fit failures;
all 32 new fits are available.

Numerical comparison runtime 409.274722 seconds; full runner wall 721.381472
seconds; peak resident memory 1,027,120 KiB. The matching log reported no
kernel exception. Critical tables, plots, identity and test outputs were
checksummed before interpretation. Complete recovery now contains320 waveform NPZ files,320 decision JSON files
and32 raw-context fit pairs. All730 locally available manifest payloads passed
checksum verification; the expanded receipt is retained under the run directory.

Independent GPT-6.1 Sol extra-high audit agrees with this interpretation.
Next: fixed shared-gain separate-axis-support native frontal ablation. A
HEOG-only detector hit currently admits both HEOG/VEOG field contributions.
Separating supports tests this mechanism; correlated or cancelling joint-fit
fields can make separate support worse, so this is not a promised repair.
Keep the K3/2000 recipe, shared gain, window support policy, calibration and
matched raw control fixed. Do not select a context architecture from the
near-zero raw-versus-VMD difference alone.
