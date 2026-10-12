# Active research task list — 13 October 2026

Current checklist. Dependencies and acceptance checks:
[REGIONAL_NEXT_STEPS_20261013.md](docs/REGIONAL_NEXT_STEPS_20261013.md).
Previous checklist preserved in [TASKS_before_20261013.md](docs/history/TASKS_before_20261013.md).
Older plans/TARGETS.md preserve historical evidence; their old pending/running
statements do not supersede this list.

## Locked scope

- AutoVMD stays stopped. No forty-setting search or neural screening restart.
- VMD only on verified frontal channels: fixed K=3, alpha=2000.
- Posterior and any later verified central correction use non-VMD methods.
- Unknown anatomy is never guessed; combine disjoint original-input corrections once.
- All numerical tests, processing, fitting, evaluation and plots run on Kaggle.
- Recipient/donor folds remain separated; confirmation remains closed.
- No new neural training before the required classical qualification checkpoint.

## Verified starting evidence

- [x] P00 Preserve stopped searches, fixed-frontal results and regional evidence.
- [x] P01 Regional pilot/full completed: full 320 examples, 32 recipients, 53 arms.
- [x] P02 Full run: 126 tests and 19,200 posterior metric-parity checks passed.
- [x] P03 Fifteen retrieved summary/test/metric payloads checksum-verified.
- [x] P04 Preserve negative findings: VMD+direct whole-cap 9.094213 dB;
  VMD+ICA 9.089350 dB; raw frontal+direct 9.337592 dB. Target unmet;
  native preservation unresolved. Full waveform retrieval is not asserted.
- [x] P05 Replace stale root checklist; preserve its historical snapshot.

## Wave 1 — independent audits, then diagnostic execution

- [x] D01 Audit prediction schemas and specify error/coverage diagnostic.
  Owner: Luna `error_audit_plan`; read-only audit complete, no numerical validation.
- [ ] D02 Verify immutable parent waveforms/manifests and source/split compatibility.
  Owner: primary; use versioned Kaggle parents where possible.
  Pilot passed parent contracts for 50 scoring examples; full parent validation pending.
- [x] D03 Implement diagnostic module, fixtures and exact-SHA Kaggle notebook.
  Authored notebook 48, frozen nine-arm diagnostic and eight numerical fixtures;
  syntax/notebook schema checked locally. Numerical validation awaits Kaggle.
- [ ] D04 Run Kaggle fixture/pilot, then full diagnostic after parity passes.
  Pilot complete: 134 tests, 154,440 channel-metric checks and 450 pooled checks.
  Twenty-two payloads checksum-verified after 502-manifest retrieval repair;
  full diagnostic version1 submitted at the unchanged pilot source 220e83d.
- [ ] D05 Publish channel/region error energy, coverage and SNR aggregation report;
  identify the measured bottleneck before choosing new correction coverage.
- [ ] N01 Audit native false corrections, references, resting eligibility and gating.
  Owner: primary; preparation overlaps the independent audit lanes.
- [x] S01 Audit SGEYESUB author implementation, dependencies and OSF calibration prerequisites.
  Owner: Luna `sgeyesub_readiness`; source audit complete, per-record eligibility unverified.
- [ ] S01a Recover sample-level right/left/up/down/blink/rest labels into a separately
  versioned calibration artifact; verify resampling alignment and per-record class coverage.
- [x] W01 Audit selective wavelet–ICA, 200 Hz scales and frontal transfer interfaces.
  Owner: Luna `wavelet_preservation_plan`; read-only audit complete, no numerical validation.

## Wave 2 — bounded classical challengers

- [ ] N02 Declare bounded frontal detection/strength candidates from existing banks;
  keep K/alpha frozen, development-only tuning and trial-safe calibration.
- [ ] N03 Run native preservation pilot; expand only eligible passing recipes.
- [ ] S02 Pin author code and implement eligible SGEYESUB reference path/custom solver
  parity, or record a reproduction blocker; depends on S01a eligibility.
- [ ] S03 Test reference parity on Kaggle; label posterior-only output an adaptation.
- [ ] S04 Run matched controlled/native SGEYESUB pilot with fixed frontal VMD.
- [ ] W02 Implement selective component-interval correction and reconstruction contracts.
- [ ] W03 Run Kaggle fixtures and matched controlled/native wavelet–ICA pilot.
- [ ] W04 Document EOG-assisted/frequency-scale changes separately from paper reproduction.
- [ ] C01 Compare with identity, VMD+posterior pass-through, direct regression and
  full-montage ICA posterior-output controls under matched information access.
- [ ] C02 Reject/repair numerical or preservation failures; retain failed runs.

## Wave 3 — estimated frontal ocular-field transfer

- [ ] F01 Verify estimated frontal artifacts on unscored calibration, with quality flags.
- [ ] F02 Implement blink/common and signed left/right/midline ocular summaries.
- [ ] F03 Fit posterior transfer on calibration only; compare no/common/signed context.
- [ ] F04 Test masks, missing context, failed-estimate pass-through and single subtraction.
- [ ] F05 Run matched Kaggle pilot; consider verified central correction only if D05
  supports it and its policy is declared beforehand. No unknown-anatomy guesses.

## Wave 4 — grouped selection and qualification

- [ ] Q01 Freeze bounded candidates after pilot evidence.
- [ ] Q02 Complete grouped controlled comparison and recipient/donor-aware uncertainty.
- [ ] Q03 Expand native preservation/suppression to eligible development cohort.
- [ ] Q04 Select by paper metrics, preservation and cost; report VMD vs regression fairly.
- [ ] Q05 Publish classical review, selected recipe and machine-checked qualification.
  Failed SNR/preservation targets stay failed; do not relax thresholds.

## Wave 5 — dependency-gated neural and final evaluation

- [ ] M01 After the required checkpoint, compare paired EEG-only compact CNN,
  TCN-only and TCN–BiGRU with matched splits and budgets.
- [ ] M02 Add cross-fitted targets only for qualified teachers and retain only
  supervision that improves the paired-only student.
- [ ] M03 Complete shortlisted multi-seed comparisons; freeze model/preprocessing/analysis.
- [ ] M04 Evaluate reserved controlled and eligible independent real data without retuning.
- [ ] M05 Verify CPU reload, overlap-add, export parity and 10/20/30/40/50-channel scaling.
- [ ] M06 Deliver reproduction notebooks and final results/limitations report.

## Cross-cutting tasks

- [ ] R01 Repair status monitoring: distinguish API errors from kernel failures;
  recover verified status before dependent submission.
- [ ] R02 Every run: commit/push, pin SHA and parents, record identities/environment,
  persist configuration, predictions, diagnostics, progress and completion.
- [ ] R03 Retrieve/checksum required completion outputs before advancing.
- [ ] R04 Append each meaningful experiment/repair/decision to Work So Far.md.

Checkboxes mean the named deliverable is complete. An audit is not a passed numerical
experiment. Record authored/launched/completed/verified/qualified separately.
At checklist creation no new diagnostic, challenger or neural job has launched.
