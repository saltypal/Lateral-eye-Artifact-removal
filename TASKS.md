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

## Lateral-focused SNR improvement plan — proposed, not launched

- [ ] SI01 Freeze original 17.112814 dB VMD and strongest 18.193247 dB raw evidence.
- [ ] SI02 One saved-output lateral residual diagnosis; target-fitted bounds remain evaluator-only.
- [ ] SI03 Complete the separately versioned RC coupling pilot without replacing old evidence.
- [ ] SI04 Implement at most one challenger supported by the residual diagnosis.
- [ ] SI05 Matched grouped development comparison with strongest raw and no-VMD controls.
- [ ] SI06 Native suppression/preservation qualification with unchanged engineering gates.
- [ ] SI07 Integrate qualified frontal and separately fitted posterior corrections once.
- [ ] SI08 Train matched EEG-only students after the classical-first checkpoint.

See docs/FRONTAL_SNR_IMPROVEMENT_PLAN.md. Rejected 2.521295 dB shape arm is not
the retained method. No AutoVMD, new numerical run or neural training started by this plan.

## User-authorized posterior and central challengers (14 October 2026)

- [x] NF01 Six target-free mechanisms, composition contracts, saved-fit/baseline parity and notebook58 validated on Kaggle:281tests passed at472b217.
- [x] NF02 Exact corpus002/context-match-full001 manifests and all consumed split/waveform/montage/calibration identities verified.
- [x] NF03 nonfrontal-pilot-002/2 completed50examples/5recipients/5donors;900frontal/30fit/85800metric checks, all186payloadsverified. Rank2both whole16.728031 versus matched13.994230dB; not qualified.
- [x] NF04 Same fixed-recipe320-example comparison completed:281tests,32recipients,21arms; whole rank2joint15.155692 versus13.162704dB. All1050manifest payloads and completion verified locally; predictions and fits recovered.
- [ ] NF05 Source-aware review and grouped shortlist evaluation; no scoring-target selection.
- [ ] NF06 Frozen native OSF blink/lateral/rest preservation validation.
- [ ] NF07 Author-code eligible SGEYESUB and ERP/BCI validation after calibration contracts.
- [ ] NF08 Fixed central GEVD-MWF controls, followed by eligible CCA/SOBI/ASR challengers in the documented priority order; retain negative results.
- [x] NF08a Fixed MWF rank1/rank2 pilot completed and rejected:282tests,50examples,all186payloadsverified; whole12.570601/12.619925 vs13.994230, CCgatefails. No MWFfull queue.
- [ ] NF09 Explicit signed frontal-field communication ablation for shortlisted nonfrontal experts; distinguish full-montage raw support from VMD-field transfer.

See docs/NONFRONTAL_CHALLENGERS_PLAN.md. Exploratory regional experiments are
authorized despite current frontal qualification failures; no teacher/model promotion.

- [x] NF10 Completed320-example frozen frontend/aggregation review: exact same nonfrontal output with VMD versus archived raw frontal; channel-first versus pooled SNR; verified middle versus temporal subgroups. No new fitting.

- [ ] NF11 One conservative gain/shrinkage challenger after saved-decision diagnosis; observable inputs only, grouped recipient/donor selection.
- [x] NF12 Fixed EOG-reference CCA pilot/full comparison and reconstruction/trial fixtures completed. Signed frontal-context CCA ablation remains pending under NF09; no claim it was tested here.
- [ ] NF13 Multichannel SOBI selective-source control only if it adds a distinct hypothesis; no nonfrontal VMD.
- [x] NF14 Frozen reviewv1 historical/conservative flag limitation documented; revised provenance fixtures passed within290Kaggle contracts004 tests. Numericalreviewv1 scores remain valid; no active-failure count inferred.

- [x] NF11a Fixedstrength/blend50examplepilot completed292tests; unchangedrank2 retained. Bestblendgain+.037918dB with CIcrossingzero. Fullmatchedcheck and gain/shapebounds follow; no selectedinferencerule.

- [x] NF11b Full320fixedstrength/blendcomparison completed292tests/549000baselinechecks. Blend0.75point15.370676mean/13.826832pooled; gainCIcrosseszero and severelateralworsens. No final replacement or widergrid.
- [x] NF11c Evaluator-onlybounds/mean-varianceaccounting saved onKaggle; severe middle lateral scalarheadroom only5.444→5.570pooled. Prioritize field/coverage and coupling, not scalarstrength alone.
- [x] NF11d Fullgain665payload recovery andcompletionverified; interruptedtransportattempts resumed fromverifiedfiles.

- [x] NF12a CCA source-balanced50pilot297tests/21arms/all188payloadsverified; direct19.601765 versusCCAref2 17.549658 andoldrank2 16.728031dB. v1subgroupcopybugpreserved; revised299contracttests/all8payloadsverified beforefull.
- [x] NF12b Corrected320examplefullCCA/direct/rank2comparison completed299tests; own subgroup preservation, pooledSNR, strongest-direct CC gates and matched uncertainty saved. Direct17.533717mean/14.780705pooled versusCCAref2 15.833574/13.923972; CCA-minus-direct CI[-2.548602,-.943181]dB and CC loss.006414969 fails. No CCA replacement/model promotion.
- [x] NF12c All1,058 fullCCA manifest payloads, completion and source identity verified locally; complete predictions, fits, metrics and logs recovered (run directory3,575,310,609bytes).
- [ ] RC01-RC05 Bounded reference-coupling/source-provenance repair and new fixed-frontal pilot; see docs/FRONTAL_REFERENCE_COUPLING_REPAIR_PLAN.md. Do not reuse old VMD caches on different input hashes.
- [ ] NF15 Match the new complete-ridge nonfrontal candidate with raw-frontal versus fixed-VMD output, then validate native preservation. Keep reserved confirmation sources closed until final freeze.

## Next regional validation stage — 14 October 2026

See docs/NEXT_REGIONAL_VALIDATION_PLAN.md for inputs, outputs, dependencies and
acceptance. Complete ridge17.533717mean/14.780705pooled is a development candidate.

- [x] N00 Freeze evidence and write the dependency-ordered plan; no AutoVMD restart.
- [x] N01 Original prefixes/donor calibration recovered atfdfdd93:305Kaggle tests passed, two donors/folds distinct, exact archive/header/axes/trial checks; all45payloads/completion verified. No coupling repair or accuracy claim.
- [x] N02 Matched new-hybrid frontend review002/2 completed313tests/320examples/549000baselinechecks. VMD17.533717mean/14.780705pooled versusraw17.777095/15.126152; SNR-difference interval crosseszero, frontalCCgatefails. Failed001/all651payloads retained; completedv2/all669payloads recovered andverified.
- [ ] N03 Homogeneous coupled-generator implementation and Kaggle boundary/scaling/leakage fixtures. Field algebra verified within312Kaggle tests/all8payloads recovered; real corpus integration remains pending.
- [ ] N04 Separately versioned two-pair20-example coupled corpus, complete recovery.
- [ ] N05 New fixed-frontal modes and matched regional pilot; old mode hashes are not reusable.
- [ ] N06 Two-record native feasibility then eligible development preservation evaluation.
- [ ] N07 One residual-supported challenger if necessary, grouped development comparisons.
- [ ] N08 Classical qualification/freeze before EEG-only students and independent/export stages.

- [x] 2026-10-15: Author N04 coupled pilot module/config/notebook and boundary/channel fixtures.
- [ ] N04: Pass Kaggle tests, produce twenty records and verify every recovered payload before N05 fixed frontal comparisons.

- [x] N04: coupled-pilot-002/2 passed 316 Kaggle tests, saved 20 records, all 37 manifest payloads verified. Version 1 failed and is preserved.
- [x] N05: Author five-arm fixed frontal review, explicit completed-parent configuration and notebook 69.
- [ ] N05: Complete Kaggle execution, recover outputs, inspect figures and assess frontal preservation before new correction methods.

- [x] Diagnose fixed-frontal review v1: 317 tests passed; stored-mode reconstruction precision failure; all 26 failed payloads recovered.
- [ ] Verify repaired reconstruction fixture and complete frontal review v2 on Kaggle.

- [x] Fixed frontal review v2: 318 Kaggle tests, 20 records/5 arms; all 95 payloads recovered, figure inspected and results assessed.
- [ ] Next: matched-subspace VMD gain versus current shared gain and raw ridge, same modes/calibration/nonfrontal outputs; no label/oracle branch selection.
- [ ] Lateral preservation qualification, native OSF and broader grouped evaluation remain incomplete.

- [x] Re-review lateral literature and document primary-source support, limitations and rejected title matches.
- [x] Implement projected separate-axis frontal gain and notebook70 with six predeclared matched arms.
- [ ] Run lateral-gain-review on Kaggle; inspect lateral gains, MSE/RRMSE/CC and preservation before claiming improvement.

- [x] Diagnose projected-axis ambiguity on rank-collapsed mode support; preserve failed version1 and unchanged failing assertion.
- [ ] Validate common-gain-anchored projected-axis fit on Kaggle before running the six-arm data comparison.

- [x] Complete and recover lateral repair v2:322tests,20records,37payloads; projected-field proposals rejected on measured results.
- [ ] Test residual-aware and raw-anchored VMD arms with all earlier controls and unchanged preservation limits.

- [x] Complete lateral repair v3:325tests,20records/eightarms,all37payloads verified and figure inspected.
- [x] Identify anchored frontal VMD as a provisional balanced pilot candidate; document residual/projection failures and comparison to direct ridge.
- [ ] Broader recipient/donor-separated development evaluation and native OSF preservation before teacher qualification.

- [x] Freeze broader source cohort:32recipients/5donors/5folds, two deterministic16-source shards and exact metadata digest.
- [x] Author original-source recovery module/notebook71/configuration; preserve old two-pair audit guard and add shard/source fixtures.
- [ ] Pass Kaggle contracts, recover both shards and verify every saved artifact.
- [ ] Generate repaired320-record corpus, then fixed five-arm anchored validation and native OSF assessment.


Broader recovery shard0/1 version1 are submitted at 90806870be379bed69b7b86c6d2bfde5bfcefce6 and were RUNNING at the latest live check. No recovery-completion or new accuracy result is claimed. Prepared a strict 32-source union validator, shared unchanged per-pair corpus producer, and five-arm frontal review. Fixed K3/2000 and anchor weight0.25; nonfrontal correction is shared exactly. Predictions precede evaluator-only targets. Save per-record progress and source/fold/exposure tables; source-aware intervals are supplementary project analysis, not paper-native tests. Local AST inspection found the extracted per-pair computational statements unchanged. Numerical fixtures and source recovery must pass on Kaggle before dependent execution.
