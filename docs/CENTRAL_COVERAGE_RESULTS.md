# Central/shared coverage: controlled development pilot

Kaggle `satyapaladugu/vmd-central-coverage-pilot/1`, source
`9e4f0ccaf01aac30e13e9091e0fc1bfdfe93895d`: 50 examples, five recipients,
six reporting arms (four primary corrections and two controls). Passed 164
numerical tests and 600 frozen-branch parity checks. Recovered all 50 waveform
shards and five fitted context models; 243 available payloads checksum-verified.

| Primary arm | Whole-cap mean output SNR dB | Central/shared mean output SNR dB | Paired clean preservation |
|---|---:|---:|---|
| Frozen regional, central pass-through | 10.002235 | 2.323554 | exact identity on clean controls |
| Add calibration-frozen EOG ridge | 10.828252 | 4.099535 | fails modification and covariance |
| Add separate-axis gated EOG ridge | 10.291573 | 2.909082 | fails modification and covariance |
| Add frontal VMD artifact-context transfer | 13.994230 | 11.121655 | zero modification/PSD/covariance error on this pilot |

These are channel-first recipient/condition/input-balanced mean output SNRs.
Pilot averages must be compared against the same five-recipient pilot baseline,
not the older 32-recipient whole-cap score. No reserved sources were evaluated.
No claim of superiority to the literature or of VMD superiority over a matched
context method follows from this comparison.

The context arm uses fixed frontal K3/2000 estimated ocular fields for signed
left/right/midline/common/lateral predictors. It fits central transfer on
cross-fitted unscored calibration chunks, then freezes it for scoring. Posterior
and frontal scored outputs stay unchanged. This is an EOG-assisted classical
hybrid, not an EEG-only deployment model.

Reference-only arms face the predeclared construction mismatch: donor references
are not scaled when controlled EEG artifact strengths change, including clean
controls. Whole-cap worst clean modification is 1.577231 for frozen ridge and
1.577036 for gated ridge; covariance error is respectively 4.359356 and 4.358467.
These fail badly and remain in the results. The context arm passes this pilot's
clean-input checks but remains below the 15 dB target and inherits the failed
frontal correlation gate. Native preservation remains unresolved.

Inherited failures remain in channel-level failure columns: 56 whole-scope rows
per primary arm in this pilot are propagated parent labels, not 56 new solver
failures. All five context models were available. Numerical-stage runtime was
78.364 seconds; runner wall time323.249 seconds includes parent checking/tests.
Kaggle setup/output publication and local retrieval time are additional.

A full 320-example/32-recipient run has now been submitted as
`central-coverage-full-001`, Kaggle `satyapaladugu/vmd-central-coverage-full/1`,
exact latest published source `731b2e5daca33585d5cb5ca012271e69d9d19c22`.
The algorithm/recipe is unchanged; the new source strengthens the rest-block
fixture and records fit availability. Full results are now verified in the section below. Luna monitored
status and a separate bounded downloader verifies returned artifacts.

Next after the full review: source-aware uncertainty and native validation,
plus the separately planned frontal shared-gain/separate-axis-support ablation.
Qualification remains false. Models and confirmation stay closed.

## Verified full coverage results — 13 October 2026 IST

Run `central-coverage-full-001`, source
`731b2e5daca33585d5cb5ca012271e69d9d19c22`,
Kaggle `satyapaladugu/vmd-central-coverage-full/1`.
Immutable manifest SHA256
`ad75aa2f3e75b9242c16fc701432ffaa61a8eb16f6ad14520ad23eb4971e97ee`.
The bounded downloader verified the summary/test/metric payloads and actual
complete execution state. Waveform and fit retrieval is tracked separately
in the run's retrieval receipt.

All 164 tests passed (14 dependency deprecation warnings), and 3,840 frozen
branch checks passed. All 32 EOG ridge, detector and VMD-context calibration
fits were available. This is 320 examples across 32 development recipients.

| Arm | Whole SNR dB | Central/shared SNR dB | Whole CC | Whole temporal RRMSE |
|---|---:|---:|---:|---:|
| identity | 3.175566 | 1.954134 | 0.733878 | 1.148284 |
| frozen_regional | 9.094213 | 1.954134 | 0.859318 | 0.628874 |
| central_frozen_ridge | 9.900663 | 3.698831 | 0.855502 | 0.678069 |
| central_axis_gated_ridge | 9.301344 | 2.369821 | 0.848028 | 0.707240 |
| central_frontal_vmd_context | 13.162704 | 10.903434 | 0.934641 | 0.338343 |
| raw_frontal_comparator | 9.337592 | 1.954134 | 0.861007 | 0.621647 |

These are saved channel-first, condition/input-balanced, recipient-balanced
means. Clean controls are scored for preservation separately. Regional scores
cannot be added. The pilot's 13.994230 dB whole result is not the full result.

| Condition | Declared input dB | Whole output SNR dB | SNR improvement dB | CC |
|---|---:|---:|---:|---:|
| blink | -5.0 | 12.575095 | 13.835837 | 0.945895 |
| blink | 0.0 | 16.070493 | 12.331235 | 0.975049 |
| blink | 5.0 | 19.821757 | 11.082498 | 0.989134 |
| lateral | -5.0 | 4.353716 | 7.019327 | 0.790733 |
| lateral | 0.0 | 8.683518 | 6.349129 | 0.884766 |
| lateral | 5.0 | 12.726614 | 5.392226 | 0.941920 |
| mixed | -5.0 | 11.062686 | 12.609635 | 0.930126 |
| mixed | 0.0 | 14.668754 | 11.215702 | 0.967766 |
| mixed | 5.0 | 18.501700 | 10.048649 | 0.986381 |

Declared mixture input levels are whole-montage energy levels; the saved
channel-first input SNR average can differ. SNR improvement is calculated
against each channel's actual input metric, separately from output SNR.

For the context arm, frontal SNR remains 17.112814 dB and posterior SNR remains
13.541703 dB, confirming they were not retuned. All four controlled clean
preservation quantities are exactly zero. Whole CC deterioration against the
archived calibrated raw-frontal/direct-posterior comparator is -0.0736338663;
frontal deterioration is still +0.0074414768 and fails the .005 gate.
Whole output SNR 13.162704 dB fails the 15 dB target. Native preservation
remains unresolved; `qualification=false`.

Inherited failure-labelled channel rows remain scored: 170 whole, 78 central,
39 frontal and 53 posterior for the context arm. These labels propagate from
the parent; they are not 170 newly failed context fits. No raw-context method
was tested in this run. Therefore no VMD-versus-raw-context claim is justified.

Reference-only arms fail clean preservation: worst whole clean modification
as listed below and other spectral/covariance failures
are retained. The reason remains the disclosed controlled construction: donor
EOG references do not scale with the controlled ocular field, including clean
controls. Do not use targets or stored mixture amplitude to repair inference.

Numerical stage runtime was 363.860348 seconds; runner wall time was
570.702088 seconds and peak resident memory 754432 KiB. Runner time includes
parent checking/tests; setup/publication/retrieval costs are additional.

Next is notebook 50's single matched raw-context arm, with unchanged four-chunk
OOF calibration, central ridge and scored VMD frontal/direct posterior outputs.
Source-aware paired uncertainty is computed on Kaggle from the saved full
coverage metric table without repeating its fits. After this attribution
comparison, prioritize native frontal shared-gain/separate-axis support.
No models, confirmation access or AutoVMD run is authorized by completion.

Exact negative-arm worst whole clean quantities from the gates JSON:

- `central_axis_gated_ridge`: `{"alpha_db": 1.3023931215994582, "beta_db": 0.5619537683069719, "clean_change": 1.915830001959485, "covariance_error": 5.625731319344903}`.
- `central_frozen_ridge`: `{"alpha_db": 1.3613739297781697, "beta_db": 0.8029255386055149, "clean_change": 1.9160656639715836, "covariance_error": 5.627902870480572}`.
