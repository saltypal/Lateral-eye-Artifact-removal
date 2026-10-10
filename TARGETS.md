# Project target checklist

Campaign: `vmd-eog-20261010`. All numerical work runs on Kaggle.
Task states: planned → implemented → Kaggle-tested → research-validated → complete.
User authorizes models automatically only if all classical gates pass.

## Phase A — classical evidence

- [x] T01 Preserve scaffold/draft and publish dedicated Git branch.
- [ ] T02 Create independent notebook workflows and reusable modules.
- [x] T03 Implement latest-published-SHA launch, saved versions and run persistence.
- [x] T04 Pass data-axis, lags, masks, metrics and solver tests on Kaggle (29 tests; contracts-006).
- [x] T05 Audit original Klados/OSF and historical exposure (45 eligible OSF sessions; audit-001).
- [x] T06 Audit raw LEMON and reserved Magdeburg source metadata (32 raw development sources loaded; external metadata frozen).
- [x] T07 Freeze participant, recipient/donor and cross-fit partitions (39 OSF/40 LEMON identities).
- [x] T08 Validate filtering, calibration boundaries and overlap-add (Kaggle contracts and audit caches).
- [x] T09 Generate coherent controlled mixtures with complete recipe manifests (3,840 examples; corpus-002).
- [x] T10 Run all 40 K/alpha decompositions and preserve every candidate (vmd-001: 5,120 fits, 11,664 candidates).
- [x] T11 Save vectors, centers, overlap, residuals, convergence and cost (673 iteration-cap fits retained; zero failed mode fits).
- [ ] T12 Compare VMD against matched direct EOG regression.
- [ ] T13 Validate GEVD-MWF and raw-frontal-supported ICA.
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
- [ ] E04 Qualify eight-second native chance analysis or report it unavailable (complete-trial extension implemented; awaiting Kaggle fixtures/pilot).
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

## Execution evidence

- Kaggle monitor: GPT-6 Luna high resumed after its usage reset and monitors the active regional run. Primary agent owns diagnosis, implementation and repairs.
- Read-only auth/source verification succeeded; source is ready.
- Archived draft has 55 passed/2 failed numerical fixtures, not current validation.
- Current tests, audit and focused real-source repair passed on Kaggle. Corpus and algorithm research gates remain pending.
