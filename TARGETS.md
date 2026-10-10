# Project target checklist

Campaign: `vmd-eog-20261010`. All numerical work runs on Kaggle.
Task states: planned → implemented → Kaggle-tested → research-validated → complete.
User authorizes models automatically only if all classical gates pass.

## Phase A — classical evidence

- [x] T01 Preserve scaffold/draft and publish dedicated Git branch.
- [ ] T02 Create independent notebook workflows and reusable modules.
- [x] T03 Implement latest-published-SHA launch, saved versions and run persistence.
- [x] T04 Pass data-axis, lags, masks, metrics and solver tests on Kaggle (59 tests; classical-fixture-006 and vmd-diagnosis-002).
- [x] T05 Audit original Klados/OSF and historical exposure (45 eligible OSF sessions; audit-001).
- [x] T06 Audit raw LEMON and reserved Magdeburg source metadata (32 raw development sources loaded; external metadata frozen).
- [x] T07 Freeze participant, recipient/donor and cross-fit partitions (39 OSF/40 LEMON identities).
- [x] T08 Validate filtering, calibration boundaries and overlap-add (Kaggle contracts and audit caches).
- [x] T09 Generate coherent controlled mixtures with complete recipe manifests (3,840 examples; corpus-002).
- [x] T10 Run all 40 K/alpha decompositions and preserve every candidate (vmd-001: 5,120 fits, 11,664 candidates).
- [x] T11 Save vectors, centers, overlap, residuals, convergence and cost (673 iteration-cap fits retained; zero failed mode fits).
- [ ] T12 Compare VMD against matched direct EOG regression (Fz comparison complete; full regional comparison running).
- [ ] T13 Validate GEVD-MWF and raw-frontal-supported ICA (analytic/software contracts passed, 608 search fits completed; full regional/native evidence pending).
- [x] T14 Qualify SGEYESUB or record why unavailable (original MATLAB pipeline not qualified).
- [ ] T15 Integrate signed frontal context and single correction subtraction.
- [ ] T16 Complete regional/shared/no-context preservation ablations.
- [ ] T17 Deliver review evidence; machine-check classical gates.
- [x] User conditionally authorizes Phase B when classical gates pass.

## Phase B — only following a passing classical gate

- [ ] T18 Generate source-isolated cross-fitted teacher targets.
- [ ] T19 Implement EEG-only shared TCN–BiGRU and regional heads.
- [ ] T20 Pass identity/masks/permutations/dynamic-channel tests.
- [ ] T21 Train paired-only baseline.
- [ ] T22 Evaluate distillation, spectrum and covariance losses.
- [ ] T23 Evaluate SNR-target and eligible event supervision.
- [ ] T24 Complete bounded architecture/regional/shared ablations.
- [ ] T25 Run grouped multi-seed development confirmation.
- [ ] T26 Freeze model/gate/analysis.
- [ ] T27 Evaluate reserved controlled data once without retuning.
- [ ] T28 Evaluate independent native real-EOG data.
- [ ] T29 Save participant/class/region uncertainty and failures.
- [ ] T30 Verify CPU reload, whole-record inference and measured scaling.
- [ ] T31 Deliver deployment/reproduction artifacts.
- [ ] T32 Complete report addressing K/bins/vectors/regions/complexity/SNR.

## Acceptance

2026-10-10 evaluation correction: primary methods must follow the prescribed
VMD papers + OSF/EEGOAR-Net + Klados. Exact mappings and disclosed adaptations
are in `docs/EVALUATION_PROTOCOL.md`. Old project diagnostics remain supplementary.
Automatic regional/review chaining was stopped while this correction was tested,
then resumed after the Kaggle fixtures passed. The completed VMD/posterior
searches retain their immutable development objectives. Regional-paper-001 is
running at source 8fed92a and adds the prescribed paper tables and statistics.
Models remain closed until paper evaluation is complete as well as project gates.

- [x] E01 Validate the paper metric implementation on Kaggle: `paper-fixture-001`, source `e90857b`, 36 tests passed; saved artifact hashes verified.
  Latest paper contracts `paper-fixture-002` at `ed091f9` passed 40 tests on Kaggle.
  Full classical integration `classical-fixture-004` at `f193e26` also passed
  39 tests and all VMD/posterior/regional/review stages, including the new
  paper-channel tables and paired statistics. Synthetic evidence cannot authorize models.
  `paper-fixture-003` at `ace3a9a` passed 45 tests, including complete-condition
  moments, aligned multi-method overlap-add, eight-second banks and participant
  grouping. All 11 retrieved artifact hashes matched. `classical-fixture-005`
  at `b6b7771` passed 48 tests and all four integration stages; model gate
  correctly remained closed for synthetic evidence.
- [ ] E02 Produce per-channel paired and condition-specific native paper tables.
- [ ] E03 Evaluate source-level paired permutations and Bonferroni families.
- [ ] E04 Qualify eight-second native chance analysis or report it unavailable (fixtures and two-record pilot passed; pilot correctly unavailable with two participants, full evaluation running).
- [ ] E05 Complete paper/protocol review before releasing models.

Fresh controlled mean SNR >=15 dB; 20 dB stretch. Clean modification <=1%,
alpha/beta <=0.5 dB, clean covariance <=0.02, comparator Pearson deterioration
<=0.005. Engineering gates, not biological guarantees. Unknown Klados anatomy,
units and participants remain unknown. Native OSF has no paired clean target.
Historical 17.9272 dB requires runtime VMD and HEOG/VEOG and is development-exposed.
Passthrough failures remain scored. Reserved data cannot select a configuration.
Failed jobs are retained and repaired on Git, with new notebook versions/run IDs.

## Verified evidence snapshot

`contracts-001`: 13 passed; `contracts-002`: 17 passed;
`contracts-003`: 21 passed; `contracts-004`: 22 passed; `contracts-005`: 28 passed; `contracts-006`: 29 passed.
`audit-001`: 375 original files and one supplement verified; all 45 OSF sessions
eligible, 36 development caches and no confirmation caches. Forty raw LEMON
identities are frozen (32 development, eight confirmation). `corpus-001` failed
on stale BrainVision companion names and is retained. `source-fixture-001`
passed the real-source repair and 26 tests. `corpus-002` is the repaired full
run, completed with 3,840 controlled examples, 32 recipient participants and
31 donor participants, zero download failures and 1,346 native/legacy examples.
`classical-fixture-001` passed bounded synthetic VMD/posterior/regional/review
integration and correctly rejected model authorization from pilot evidence.
Full `posterior-001` completed: 304 candidates, 608 fits, zero fit failures.
Its archived selection objective is development evidence. `vmd-001` completed:
40 settings, 5,120 fits, 11,664 correction candidates; 673 iteration-cap fits
and zero failed mode fits. Fourteen retrieved artifacts passed checksum checks.
The global projected-correction selection is K=4, alpha=2000, but the preferred
selection channel available in this corpus was Fz only. This does not establish
all-frontal optimality. Full regional comparison is running; controlled research
SNR and classical acceptance remain unestablished. Neural work remains closed.

The original Fz selected scores are 5.2274 dB for projected VMD versus 10.4617 dB
for direct regression. No VMD benefit is established. A separate notebook 04a
diagnoses reconstruction, mode/residual projection closure and calibration
baseline handling; it will not overwrite the original search. Notebook 07b's
two-record pilot completed with 51 scoring trials and verified publisher
HEOG_lpf/VEOG_lpf references. Its 24 failure events were iteration-cap
pass-throughs only; all 15 retrieved artifact hashes matched. The full native
run `native-protocol-full-001` is launched at `bf7047e`, version 2. The pilot
had fewer than five participants, so its chance analysis correctly stayed
unavailable; the full run evaluates eligibility across development sources.

`contracts-007` at `bf7047e` passed 54 tests on Kaggle, including dimensional
feature invariance, calibration reference baselines and projection linearity.
The separate `vmd-diagnosis-001` run completed at source `fc79b30` with 51 tests.
All 12 saved payload hashes matched after retrieval. Reconstruction and projection
closure errors were below 6e-8. The calibration-baseline control changed Fz SNR
only from 5.2274 to 5.2706 dB; it does not resolve the deficit. Matched lag-zero
direct regression scored 8.6741 dB. The next diagnostic version isolates discarded
projections, residuals, capped fits and threshold sensitivity. Its results must
retain clean-preservation outcomes before any teacher change is accepted.

The final review now includes channel-first primary metric figures alongside
explicitly labelled supplementary pooled-energy figures. Native plots use
complete-condition participant means, original source units for rest RMSE and
adapted chance thresholds only where computed. These additions await Kaggle
execution and visual inspection; authoring alone is not validation.

`classical-fixture-006` and `vmd-diagnosis-002` at `9ed2d35` passed 59 tests
on Kaggle. The fixture passed all four integration stages and primary/native
figure rendering. The diagnostic's 14 payload hashes matched and its figure
was inspected. Lower mode thresholds improved Fz recovery but changed clean
EEG by 35.0809%; residual correction changed it by 2.9008%. Neither passes.
Separate raw-window and mode gates are the next mechanism ablation, with all
three declared lag banks and matched direct comparators. It cannot itself
authorize a teacher or neural training.

## Execution evidence

- Kaggle monitor: GPT-6 Luna high monitors regional-paper-001, native-protocol-full-001 and the latest VMD mechanism comparison. Primary agent owns diagnosis, implementation and repairs.
- Read-only auth/source verification succeeded; source is ready.
- Archived draft has 55 passed/2 failed numerical fixtures, not current validation.
- Current tests, audit, real-source repair and full controlled corpus passed their engineering checks on Kaggle. Classical accuracy/preservation acceptance and all model work remain pending.
- Classical-fixture-006: 70 retrieved payloads matched the immutable manifest; 216 waveform/cache payloads remain available on Kaggle and were not retrieved locally. Its saved source/run identity and completed four-stage log were verified.
- VMD-diagnosis-003: 60 tests passed, 128 preferred-frontal examples, all 14 retrieved payload hashes matched. Two-stage gating at outer 0.6 retained scored clean inputs, but the displayed lag-bank-1/mode-0.2 candidate reached only 8.8051 dB versus 10.4617 dB for matched direct regression. No teacher is approved. A new notebook 04b implements the paper-inspired VMD–SOBI candidate; numerical contracts and its 12-example Kaggle pilot must pass before a full mechanism comparison.
- VMD-SOBI-pilot-001 at `7210c1`: 63 tests passed on Kaggle, including independent joint-diagonalization, identifiable source-recovery and entropy oracles. Twelve development examples completed with zero fit failures. All 14 retrieved payload hashes matched; 12 waveform payloads remain on Kaggle. The saved plot was visually inspected. Best displayed projected-source candidate reached 8.8538 dB versus 8.9368 dB for matched lag-zero regression; worst-recipient scored clean RRMSE was zero for both. Whole-source and entropy-assisted removal were weaker. This is software and bounded mechanism evidence, not teacher approval or a 15 dB result. A 128-example comparison is the next declared check; reserved confirmation remains closed.
