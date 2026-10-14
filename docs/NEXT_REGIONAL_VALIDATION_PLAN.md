# Next regional validation stage: provenance, frontal contribution and native EEG

14 October 2026. Active mode: Super Code for research design, with notebook-driven
implementation. This plan starts from the completed 17.533717 dB development
hybrid. It does not restart AutoVMD or replace the user's regional architecture.

## Starting evidence and unanswered questions

Frontal K3/alpha2000 VMD plus complete calibrated posterior/middle reference
ridge reaches 17.533717 dB channel-first mean, 14.780705 dB pooled-window mean,
MSE 3.546698e-12 VÂ², temporal RRMSE .191564 and CC .972400. The severe lateral
-5 dB stratum remains 13.278714 dB. Reference CCA rank2 is inferior to complete
ridge, with matched CCA-minus-ridge SNR interval [-2.548602, -.943181] dB and
CC deterioration .006414969 > .005. Retain its negative evidence.

These are development-exposed, EOG-assisted, regression-generated results.
Earlier raw-frontal regression is stronger than fixed VMD on frontal channels;
the exact new full-ridge system still lacks that matched frontend comparison.
Earlier native resting modification/covariance gates fail. The old synthetic
producer scales injected ocular fields but leaves supplied EOG arrays unscaled.
An affine, lag-centered field cannot be repaired merely by multiplying raw EOG.

The three questions are: do we have trustworthy source/calibration inputs; what
does frontal VMD add while nonfrontal output is identical; does the retained
hybrid suppress both ocular types while meeting native preservation limits?

## Frozen constraints and information access

200 Hz, 0.5-40 Hz, 1024 samples, hop512; filter segments/trials independently.
Fixed frontal K3/alpha2000, RMS normalization, tolerance1e-6, cap2000. VMD only
verified frontal channels. No broad K/alpha or CCA-rank search. Nonfrontal ridge
retains lags[-10,0,10], penalty.01, association.4 and observed-input gain cap8.
Unknown/masked anatomy passes through. Estimate fields relative to original EEG;
disjoint regions and one subtraction. Keep target/true-artifact/generation scale
outside all denoising and calibration-fitting APIs. Reserved sources stay closed.

All numerical work, tests, fitting and figures run on Kaggle. Local work authors
modules/configuration/notebooks, checks syntax, performs Git and recovers artifacts.
Notebook launches pin published SHAs and immutable parent versions. Every failure
keeps its run, logs and diagnosis; repair means a new version/run, not overwritten
evidence. No neural implementation/training or deployment claim at these stages.

## Ordered work and acceptance

| ID / owner | Input and purpose | Output | Acceptance and dependency |
| --- | --- | --- | --- |
| N01 / primary | corpus002 and CCAfull001 metadata; recover original unscored sources | two deterministic recipient/donor pairs, verified LEMON prefixes and donor calibration | same frozen split/folds, exact archive/header/parent hashes, trial separation; source units explicit; no algebraic unmixing |
| N02 / primary | immutable saved CCAfull/context outputs; hold nonfrontal output fixed | VMD versus raw-frontal matched tables for complete ridge and rank2 controls | new fits0; exact waveform/mask/target/source identity; report channel-first and pooled SNR, all conditions and source-aware uncertainty |
| N03 / primary | N01 source bundles and documented driver baseline | new homogeneous coupled generator plus Kaggle fixtures | F(aR)=aF(R), zero drive, signed scaling, lag joins, units, leakage and reconstruction tests pass; old generator remains unchanged |
| N04 / primary | N03, same two development pairs | separately versioned 20-example clean/blink/lateral/mixed corpus at -5/0/+5 | source hashes, recipient/donor separation, target construction and generated driver units recorded; clean controls explicitly idealized |
| N05 / primary | N04; new input hashes require new frontal modes | matched identity/raw-frontal/VMD-frontal pilot with common complete-ridge nonfrontal arm and pass-through ablation | all failures scored, full reconstruction residual retained, paper metrics, preservation, condition/region diagnostics, solver cost; no qualification from two pairs |
| N06 / primary | intact native OSF sources and frozen eligible calibration/scoring | two-record whole-trial native feasibility followed by eligible development cohort | required .set/.fdt verified first; original HEOG/VEOG unchanged for scoring; suppression, resting RMSE/PSD/covariance, failure counts; no native reconstruction SNR |
| N07 / primary | N02-N06 diagnosis | at most one targeted correction challenger | predeclared grouped development selection, stronger direct comparator, distinct mechanism, no clean-target component/gain selection |
| N08 / primary | qualified classical review and explicit gate evidence | freeze then matched EEG-only students, independent evaluation, export | classical-first checkpoint passes before models; separate controlled/native/teacher/export gates; reserved data opened only after freeze |

N01 and N02 are independent: a source-recovery notebook and a saved-output CPU
review can run separately. N03 depends on N01; N04 on N03; N05 on N04. Native
source/dependency auditing can run alongside N01/N02, but method evaluation waits
for frozen eligible calibration contracts. No broad posterior search is repeated.

N01 selects two recipients lexicographically from the already-scored development
cohort, requiring different donor identities/folds, with first chronological
recipient window and exact donor record from their archived recipe. Selection
must not use SNR, method predictions or failures. Use immutable original LEMON
archive/header hashes. Save source prefix arrays; original raw archives can be
temporary, with source URL/hash and repaired-header evidence retained. OSF cache
physical units absent from source metadata remain unknown: use explicit original
numeric driver units, not an invented volts conversion or physical sensor claim.
This audit recovers prerequisites; it does not repair coupling or qualify accuracy.

## Coupled generator definition

Fit the donor field using only its original unscored calibration, with a declared
per-reference baseline and signed lag operator. Define reference excursions R
relative to that frozen baseline and F(R)=B D^-1 L(R), with valid lag samples
masked independently at trial boundaries. The operator must have no affine
intercept. Generated calibration/scoring each use a single multiplier a:
injected field=aF(R), generated reference excursion=aR. Clean injection uses
zero driver and zero field; its easy identity result cannot establish real false
positive behavior. The recipient reference EEG remains low-ocular, not proven
artifact-free. Regression-derived generation bias remains after coupling repair.

Do not reuse old VMD caches on changed inputs. Do not reconstruct unavailable
recipient calibration from contaminated arrays. Keep source generation evidence
separate from estimator input. A consistent generator might leave gain-adjusted
outputs unchanged; an SNR increase is not an acceptance condition for this repair.

## Decision rules and bounded improvement path

N02 can establish contribution on the legacy corpus, not fresh generalization.
If fixed frontal VMD loses, preserve it as the requested research branch while
reporting the stronger raw control. Do not silently relabel regression gain as
VMD benefit. On N05/6 diagnose gate misses, retained-field direction errors,
amplitude errors, timing errors and clean false positives separately.

Only after that diagnosis choose one challenger: separate signed HEOG/VEOG
time-dependent reference projection for missed lateral intervals, or calibrated
signed frontal-artifact transfer where raw EOG coverage is inadequate. Existing
full-ridge output stays the matched control. No new decomposition parameter
search. Eligible author SGEYESUB needs six explicit movement/rest labels; four
OSF trial codes or HEOG sign are not substitute labels. Generic ASR/ICA/SOBI
queues are deferred unless evidence identifies a distinct unresolved mechanism.

## Evaluation and completion checklist

Paper primary: MSE, temporal RRMSE, ordinary Pearson CC; output SNR and improvement
separate. Additional spectral RRMSE only under its declared paper implementation.
Native suppression: original unchanged-reference HEOG for lateral, VEOG for
blink/vertical, resting RMSE and spectra following the prescribed protocol. Use
participant-level paper inference when eligible; label approximate recipient/
donor intervals as supplemental. Keep failed/low-scoring rows and both pooling
conventions visible. Region dB values cannot be added.

Unchanged engineering limits: worst-source clean modification<=1%, alpha/beta
error<=.5dB, relative covariance<=.02, CC deterioration<=.005 against declared
strongest matched direct comparator. Fresh controlled mean>=15dB,20stretch.
Neither self-comparator CC nor synthetic zero-driver preservation proves native
qualification. Two-pair pilots are engineering checks, not generalization tests.

- [x] N01 Original source/calibration recovery verified on Kaggle:305tests, two distinct donor/fold pairs, all45payloads recovered.
- [x] N02 Exact new hybrid frontend contribution evaluated in repairedversion2:313tests,320examples,549000baseline checks; failedversion1/all651payloads retained.
- [x] N03 Homogeneous coupling and boundary fixtures passed on Kaggle at c6a8420; pilot integration remains separately gated.
- [x] N04 Two-pair new corpus saved and fully recovered: coupled-pilot-002/2, 316 tests, 20 records, all 37 payloads verified.
- [ ] N05 Fixed frontal/matched nonfrontal pilot completed.
- [ ] N06 Native dependencies, trial evaluation and preservation qualification.
- [ ] N07 One evidence-supported challenger, if needed, compared in grouped development.
- [ ] N08 Classical review/freeze before students and final independent evaluation.

Deliverables for each stage: notebook, reusable module/configuration, exact source
SHA, checksummed parents and source/split/calibration identities, saved fits/
predictions, paper metrics/gates/failures, runtime/memory, figures and complete
recovery receipt. Append every stage and repair to Work So Far.md and TASKS.md.

## First execution

Implement and launch N01 as a bounded Kaggle CPU source-coupling audit. Its tests
run before source processing; optional expected checksum checks in the shared
LEMON loader must reject changed archive/header bytes before numerical loading.
This first run saves verified original prefixes/calibration for N03, without
creating new mixtures, VMD decompositions, trained models or reserved access.

## Verified execution update and next concrete interface

N01 completed atfdfdd93. Coupled-field algebra passed312Kaggle tests atc6a8420;
reporting repair passed313 at0b74d48 before N02version2. All source and focused
contract artifacts are recovered. N02 now compares VMD17.533717mean/14.780705
pooled against raw-frontal17.777095/15.126152 with identical nonfrontal output.
VMD-minus-raw SNR interval[-.516659,.043339] crosseszero; frontalCC deterioration
.007441477>.005 still fails. Native qualification and fresh confirmation stayclosed.
This evidence does not justify a VMD superiority claim or automatic model work.

The next producer module should expose:

```text
build_coupled_pilot(verified_source_probe_parent, verified_original_corpus_parent,
                    configuration, output_directory) -> twenty paired records
```

Use the exact source-probe manifest
8618b226331fc8d170b87644dae00e031502fc68ca7a7d0d0f3c9bb2a32fad24
and originalcorpus manifest93da62cb9cec2984ec9687ee27f308d2dbea6dd57abd62541468daafb89d5e0d.
Persist the same source-only two-pair selection. Select the previously identified
recipient scoring start from its recovered window manifest, not a new best-score
window. Resolve original annotated blink/lateral donor windows through the donor
record's window index and corpus native-array identities; verify every consumed
array hash. Mixed uses the declared signed reference combination. No guessed
direction labels or scoring targets may select a correction branch.

Intersect verified unique channel names. For each calibration, choose an eligible
complete donor-trial prefix no longer than the recovered recipient prefix. Fit
the homogeneous field on that exact original unscored donor subset; record its
baseline/scales/coefficients, valid samples, units and source hash. Construct the
mixed calibration with one multiplier on its field and reference excursions.
Recipient background calibration stays producer-only truth, not a denoiser input.

Each pair generates one idealized clean control and blink/lateral/mixed at -5,
0,+5dB: ten records per pair. Arrays are EEG[C,1024], references[2,1024], paired
reference[C,1024], trueartifact[C,1024] and verified metadata/masks. New calibration
arrays retain their trial boundaries and labels. Nominal SNR uses whole available
montage energy; save measured regional input SNR separately. Denoisers receive
only EEG, references and allowed calibration/metadata; labels, trueartifact,
target and generation multiplier stay producer/evaluator-only.

Kaggle acceptance before N05: exact reconstruction, same-multiplier field replay,
finite arrays, source/recipient/donor separation, whole-trial boundaries, no target
fit, and full artifact recovery. Save calibration/scoring field means, valid-lag
coverage and spectral summaries to expose baseline/intercept/edge changes. Do not
demean the paired target or adjust waveform offsets using scoring truth. A field
mean or boundary issue must be diagnosed before comparing denoisers, rather than
being hidden through a changed metric or extra target processing.

Both the low-ocular-reference limitation and regression-generated-field bias
remain. This pilot checks construction and solver contracts; it cannot qualify
fresh generalization, remove native-source uncertainty or prove neural retention.
Once it passes, recompute only fixed frontal K3/alpha2000 modes on its new EEG
hashes and compare matched full-ridge nonfrontal versus pass-through outputs.
