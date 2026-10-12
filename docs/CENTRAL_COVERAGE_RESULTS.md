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
fixture and records fit availability. Full results are pending. Luna monitors
status and a separate bounded downloader verifies returned artifacts.

Next after the full review: source-aware uncertainty and native validation,
plus the separately planned frontal shared-gain/separate-axis-support ablation.
Qualification remains false. Models and confirmation stay closed.
