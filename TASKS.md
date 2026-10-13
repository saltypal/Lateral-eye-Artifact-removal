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
- [x] D02 Verify immutable parent waveforms/manifests and source/split compatibility.
  Owner: primary; use versioned Kaggle parents where possible.
  Full parent contracts passed for all 320 examples; source/split identities verified.
- [x] D03 Implement diagnostic module, fixtures and exact-SHA Kaggle notebook.
  Authored notebook 48, frozen nine-arm diagnostic and eight numerical fixtures;
  syntax/notebook schema checked locally. Numerical fixtures passed on Kaggle in pilot and full diagnostic.
- [x] D04 Run Kaggle fixture/pilot, then full diagnostic after parity passes.
  Pilot complete: 134 tests, 154,440 channel-metric checks and 450 pooled checks.
  Twenty-two payloads checksum-verified after 502-manifest retrieval repair;
  Full version1 completed at unchanged source 220e83d: 988,200 channel, 2,880 pooled and 162 balanced-mean checks passed.
- [x] D05 Publish channel/region error energy, coverage and SNR aggregation report;
  Full report: docs/REGIONAL_DIAGNOSTIC_RESULTS.md. Untouched central/shared contributes 87.9066% mean balanced error share.
- [ ] N01 Audit native false corrections, references, resting eligibility and gating.
  Owner: primary; preparation overlaps the independent audit lanes.
- [x] S01 Audit SGEYESUB author implementation, dependencies and OSF calibration prerequisites.
  Owner: Luna `sgeyesub_readiness`; source audit complete, per-record eligibility unverified.
- [ ] S01a Recover sample-level right/left/up/down/blink/rest labels into a separately
  versioned calibration artifact; verify resampling alignment and per-record class coverage.
- [x] W01 Audit selective wavelet–ICA, 200 Hz scales and frontal transfer interfaces.
  Owner: Luna `wavelet_preservation_plan`; read-only audit complete, no numerical validation.

## Immediate approved coverage experiment

- [x] CC01 Author fixed four-arm coverage module, calibration-context cache, contracts and notebook 49.
- [x] CC02 Run 50-example Kaggle pilot; verify numerical tests, checksum recovery and frozen-branch parity.
- [x] CC03 Run 320-example development comparison after verified pilot: 164 tests and 600 frozen-branch checks passed. Pilot context whole-cap 13.994230 dB vs frozen10.002235; full verified: 13.162704 vs9.094213 dB,164 tests and3,840 parity checks. No qualification.
- [x] CC04 Report preservation failures, amplitude mismatch, per-condition results and explicit qualification status.

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

## Matched context attribution — immediate next stage

- [x] MC01 Publish one matched raw OOF context arm and notebook 50.
- [x] MC02 Verify Kaggle tests, 50-example pilot and checksummed recovery. Run context-match-pilot-001/1 launched at c8159e1; 180 tests,150 branch checks,300 parent predictions and171,600 metric checks passed; critical outputs checksum-verified.
- [x] MC03 Run unchanged320-example comparison and source-aware uncertainty. Full context-match-full-001/1 verified at c285e53:180tests,960branch checks,1,920predictions,1,098,000metric checks;32rawfits. Whole raw13.183201 vsVMD13.162704 dB; approximate interval includeszero.
- [x] MC04 Report VMD contribution, failures and preservation before native frontal repair. No context winner established; whole15dB and frontalCC gates fail. Native frontal preservation previously failed; nativecentral transfer untested.


## Fixed frontal preservation mechanism test

- [x] FP01 Expose jointly fitted axis fields with unchanged shared gain; preserve default parity.
- [x] FP02 Add separate HEOG/VEOG temporal supports; keep existing OR-support policy unchanged.
- [x] FP03 Run Kaggle fixtures including cancelling fields and one-axis isolation. 202 tests passed on Kaggle; source ce6f31d.
- [x] FP04 Run exploratory native pilot with identity and matched raw/VMD OR/separate support. Verified51trials/2records/153predecessorchecks; preservation still fails.
- [x] FP05 Recover, verify and report preservation/suppression before any bounded tuning. All495payloads verified; all51trial/306window/51mode bundles saved. See NATIVE_SEPARATE_AXIS_RESULTS.md. No promotion or tuning.

- [x] FP06 Diagnose remaining resting interventions from saved fields/supports/decisions on Kaggle, without fitting or changing outputs. native-axis-diagnostic-001/1 verified at5165ee3:208tests,255checks,51trials,16retrievedpayloads; dominant VEOG interventions persist.
- [x] FP07 Author bounded native feasibility replay with twelve declared association/strength settings and matched raw/VMD OR/separate paths; no native recipe selection. Notebook53.
- [x] FP08 Run Kaggle fixtures/255baselinechecks/40condition-metricrows and recover all native candidate gates/predictions. Verified215tests/51trials/all122payloads at4719e52.
- [x] FP09 Decide feasibility before any controlled extraction, promotion or dependent integration; all thresholds unchanged. Zero active preservation passes; no uniform recipe in this bank passes both exposed records. Stop promotion; see FRONTAL_BOUNDED_FEASIBILITY_RESULTS.md.


## Next isolated frontal boundary ablation

- [x] BT01 Declare original-trial support alignment, frozen0.4/1 policy and four versioned parent dependencies. See FRONTAL_TRIAL_SUPPORT_PLAN.md.
- [x] BT02 Implement detector reconstruction from exact unscored corpus calibration and require all cached threshold/support parity.
- [x] BT03 Implement original-trial support track alignment and nine-arm cached replay; no new VMD or field fit.
- [x] BT04 Validate fixtures,306detectorwindows/255old-waveformchecks/40conditionrows/204paddingchecks on Kaggle, recover and report all outcomes. Verified224tests/51trials/all194payloads at0e33d87; dataset-free224test check also passed.
- [x] BT05 Assess suppression/preservation without native scored-rest selection or automatic promotion. Zero trial-support preservation passes; VMDseparate worstmod0.154503/cov0.102743. Mixed small suppression changes; no promotion. See NATIVE_TRIAL_SUPPORT_RESULTS.md.


## Fixed frontal artifact-shape hypothesis — completed and rejected

- [x] SC01 Audit exact controlled mode/field/window parent interfaces and publication identities. Saved post-gain fields, selected modes and all320archivedK3 windows available; exact parents pinned.
- [x] SC02 Implement DC-preserving selected-mode artifact-shape helper and numerical fixtures; no gain refit. Verified focused Kaggle run frontal-shape-contracts-001/1:236tests passed at1444b4e; eight manifest payloads recovered and verified.
- [x] SC03 Run fixed four-arm controlled comparison with strongest raw comparator and unchanged baseline parity. Completed320windows/960predictionchecks/112320metricchecks; all663payloads verified at1444b4e.
- [x] SC04 Report paired accuracy/preservation and reject or continue the fixed challenger; no native/teacher promotion from controlled results alone. Newshape2.521295dB versus original17.112814/raw18.193247; rejected. See FRONTAL_ARTIFACT_SHAPE_RESULTS.md.

Contract: FRONTAL_ARTIFACT_SHAPE_PLAN.md. Prior gain-only and component-projection
negatives remain explicit. No broad parameter search or model work authorized
by this planned ablation.

## Frozen-output shape failure diagnosis

- [x] SD01 Validate frozen-parent/prediction/DC/edge/MSE identity and error-decomposition fixtures on Kaggle. Completed243tests/640waveforms/8320MSE/4160DCchecks at5d655c7; all15manifestpayloads verified.
- [x] SD02 Recover per-channel and recipient/condition/input error terms; explain failed projection without selecting another recipe. Dominant discarded-field energy plus positive error interaction explains reconstruction penalty; allninecontaminated strata worse. See FRONTAL_SHAPE_DIAGNOSTIC_RESULTS.md. No promotion.

## Frontal independent-validity checkpoint — audit complete, readiness false

- [x] IB01 Audit an independent paired contamination mechanism and verified frontal source eligibility. Repaired002/2 at27eeefe passed266tests andexactsource/tablechecks; all15payloads verified. Eight audited sources, no eligible candidate; benchmark_ready=false. Failed001 preserved. IB02-IB04held.
- [ ] IB02 Freeze identity/raw/originalVMD-only protocol, source groups, calibration and coupled reference/ocular amplitude contract.
- [ ] IB03 Run matched Kaggle evidence only after generator/data contracts pass.
- [ ] IB04 Review accuracy/preservation/uncertainty without promotion beyond measured scope.

See FRONTAL_INDEPENDENT_BENCHMARK_CHECKPOINT.md. No new benchmark or recipe has launched.

## Proposed reference-coupling repair — unstarted

- [ ] RC01 Audit recoverable unscored source inputs, reference baselines, units and identities.
- [ ] RC02 Author separately versioned homogeneous field/reference generator contract; preserve old corpus.
- [ ] RC03 Run coupling/offset/lag/leakage fixtures on Kaggle.
- [ ] RC04 Create bounded development-source pilot with complete recovery and new hashes.
- [ ] RC05 Run identity/raw/original-frontal-VMD matched pilot after contracts pass.

This is a development consistency repair, not the independent benchmark. See FRONTAL_REFERENCE_COUPLING_REPAIR_PLAN.md.
