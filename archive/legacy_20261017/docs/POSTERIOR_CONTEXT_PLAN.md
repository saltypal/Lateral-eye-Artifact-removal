# Posterior ICA / GEVD-MWF context experiment

## Question and scope

Does signed frontal information improve posterior blink/lateral cleaning over
posterior-only fitting and a common frontal average, under matched calibration,
inputs, parameter budgets and EEG preservation? This CPU-only experiment runs
alongside the GPU AutoVMD screening. It changes no neural architecture and opens
no confirmation sources. It does not automatically qualify a teacher.

The archived `posterior-001` search compared 304 recipes and 608 calibration
fits with no recorded fit failures. It selected full-montage ICA on the old
development subset. That preliminary search lacked matched common-versus-signed
support for both ICA and MWF and did not establish this context comparison.

## Checklist and interfaces

- [x] Inspect legacy posterior implementation and archived search scope.
- [x] Add reusable `posterior_context.py` experiment module.
- [x] Protect ICA lagged component/reference association at calibration trial joins.
- [x] Build metadata-stable support layouts and explicit missing-frontal flags.
- [x] Author polarity, permutation, leakage and single-subtraction fixtures.
- [ ] Run numerical fixtures and bounded pilot on Kaggle CPU.
- [ ] Inspect fit failures, recorded source exclusions, layout/rank and metric tables.
- [ ] Run the full development grid if the pilot passes software/data contracts.
- [ ] Inspect source-excluded context gains by blink/lateral/mixed and input level.
- [ ] Evaluate selected recipes on native OSF resting/ocular trials.
- [ ] Integrate a qualified posterior expert with frontal VMD using disjoint outputs.
- [ ] Qualify cross-fitted teacher supervision independently of paired learning.

`context_input(data, support)` returns support signals, output-row indices,
original posterior indices and context flags. It consumes verified names,
region/hemisphere codes and masks. A window with zero frontal variance keeps the
same layout as its calibration. Unknown channels never receive guessed regions.

`compose_correction(original, posterior_ids, posterior_artifact, frontal_artifact)`
subtracts one combined artifact. It rejects overlapping frontal/posterior output
masks. The posterior-only experiment supplies no frontal correction; frontal EEG
provides support, not a prior-cleaned input.

## Matched arms

Both extended Picard ICA and Ledoit-Wolf regularized GEVD-MWF use each support:

1. Available verified posterior channels only.
2. Posterior plus a common frontal average.
3. Posterior plus left, right, midline, common and signed right-minus-left summaries.
4. Posterior plus raw available frontal channels.
5. Available full montage, with posterior-only output correction.

Signed context columns are deliberately redundant: common and difference are
linear combinations of left/right/midline summaries. Picard's numerical-rank
reduction and covariance shrinkage handle this differently. Inspect saved ranks
and fits; an advantage cannot be attributed solely to polarity without comparing
these effects. Availability comes from metadata, not amplitudes in scored EEG.

Calibration uses the existing unscored recipient prefix and donor calibration
artifact field, with their recorded trial boundaries. ICA component selection
uses calibration HEOG/VEOG only. A shared scoring-window EOG association gate is
an EOG-assisted algorithm adaptation; EEG-only inference is not claimed here.
No paired clean target enters either fitting or correction.

Include identity and joint HEOG/VEOG ridge controls (penalty .01, instantaneous
lags). This is a declared fixed comparator, not an independently tuned regression
search. Insufficient calibration or fit failure remains a scored pass-through.

## Pilot, full search and source control

Pilot: one deterministic recipient from each of five frozen source buckets;
one clean and 0 dB blink/lateral/mixed window per recipient. Search ranks 1/2,
instantaneous and the declared short-lag bank, thresholds .4/.6 and strengths
.5/1. This is 105 candidates including controls; it is not the full search.

Full: first window per recipient/condition/input-level; all eligible development
recipients; -5/0/5 dB mixtures; ranks 1..4; both lag banks; thresholds
.2/.4/.6/.8; strengths .25/.5/.75/1. This is 737 candidates including controls.
Source/boundary/unit/preprocessing checks are reused unchanged.

Recipe selection excludes each held-out bucket's recipients and donors.
Recipient/donor fold inconsistency fails before sampling. An evaluation source
may use its own unscored calibration, as explicitly allowed by the classical
protocol. The pilot provides very limited participant evidence. Confirmation
data stay closed; no superiority or final SNR claim follows from the pilot.

## Metrics and persistent artifacts

Primary paired metrics use the existing paper-mapped MSE, temporal RRMSE and
ordinary Pearson CC functions. Output SNR and SNR improvement are separate
explicit energy-ratio measures. Calculate per channel before equal recipient
aggregation. Report condition and input-level results separately. Units are
retained recipient volts (MSE in V^2), not arbitrarily rescaled PSNR.

Clean modification, alpha/beta error and covariance are separate engineering
screening gates. This experiment marks these as partial gates: native validation
and the paired-correlation comparator gate still need qualification. It always
saves `teacher_qualified: false`; passing partial gates cannot enable distillation.

Save candidates, exact selected source/window identities, fit diagnostics,
per-example score and compressed per-channel shards, and progress after every
example. Final tables contain source-excluded recipe selections and scores.
The standard notebook runner records SHA, environment, tests, configuration,
parent hashes and completion. Partial outputs are evidence rather than a complete
parent; restart/resume cannot skip source/configuration identity checks.

Fit cost and cache-assisted correction cost are separate. Cached candidate
evaluation times are not fresh deployment latency or a scalability benchmark.
The standard runner also records peak memory and total wall time.

## References and adaptations

- [Somers / exporl MWF authors' code](https://github.com/exporl/mwf-artifact-removal):
  marked clean/contaminated calibration and temporal multi-channel filtering.
  Ledoit-Wolf covariance, bounded excess rank and frontal virtual channels are
  our adaptations, not exact MATLAB replication.
- [Picard authors' implementation](https://github.com/mind-inria/picard):
  extended nonorthogonal ICA, with calibration EOG component selection and
  posterior-only artifact reconstruction added by this project.
- [EEGOAR-Net implementation](https://github.com/dmarcos97/EEGOAR-Net): native
  condition-specific EEG-EOG correlation and resting preservation. Native
  evaluation remains separate from controlled reconstruction SNR.

The controlled reference EEG is low-ocular LEMON, not proven artifact-free.
Regression-derived donor fields may favor EOG-assisted methods. Independent
native evidence is required before interpreting controlled gains as physiological
artifact separation.
