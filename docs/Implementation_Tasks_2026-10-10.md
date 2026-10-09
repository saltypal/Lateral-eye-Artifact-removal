# Regional EOG implementation task ledger

Started 2026-10-10. Source modules are authoritative; all numerical processing,
tests, experiments, training and plots run through Kaggle notebooks. Local work
is authoring, static validation, version control, packaging and orchestration.

## Accepted scope

Frontal independent VMD with joint HEOG/VEOG evidence; posterior calibrated
GEVD-MWF/ICA and qualified SGEYESUB comparison; signed frontal context; a single
EEG-only regional TCN-BiGRU student. Preserve legacy models and checkpoints.
No wavelets, state-space models, default transformer, or streaming claims.
The 17.9272 dB legacy result is development-exposed and needs runtime VMD/EOG.

## Tasks

- [x] Confirm Kaggle authentication and private source dataset inventory.
- [x] Inspect existing implementations and agree notebook execution plus Git modules.
- [ ] 01: Version contracts, model registry, and legacy compatibility.
- [ ] 02: Recording manifests and historical exposure ledger.
- [ ] 03: Freeze source/participant/donor/recipient splits and access guards.
- [ ] 04: RunSpec, notebook stage registry, resumable run-specific persistence.
- [ ] 05: Preprocessing/windowing and metric-v2 numerical contracts.
- [ ] 06: Cached K=3..10 / alpha frontal VMD search and lagged projection.
- [ ] 07: Posterior GEVD-MWF and routed ICA.
- [ ] 08: Qualify SGEYESUB against source implementation or explicitly mark unavailable.
- [ ] 09: Signed regional context, teacher composition and constrained search.
- [ ] 10: Source-separated controlled corpus and external source audit.
- [ ] 11: Cross-fitted teacher targets and fitting-identity checks.
- [ ] 12: EEG-only RegionalEEGStudentV2, masks, dynamic channels and identity initialization.
- [ ] 13: Paired training, distillation and bounded architecture/loss searches.
- [ ] 14: Grouped multi-seed development validation and immutable model freeze.
- [ ] 15: Reserved controlled and independent real-data evaluation.
- [ ] 16: Fresh CPU export/reload, measured scaling and reproducibility report.

## Ownership and checkpoints

- Root: integration, training campaign, notebook/controller changes, Git commits,
  source publication, final evidence and task ledger.
- Teacher agent: VMD/reference selection, posterior methods, teacher search/tests.
- Student agent: v2 architecture/losses, model registry, compatible export/tests.
- Data agent: RunSpec/contracts, source/split/corpus governance, metric-v2/tests.
- Luna 6 High: Kaggle readiness, submission monitoring and output retrieval.
- Sol medium reviewer: independent code review and Kaggle testing after integration.

Numerical tests must pass on Kaggle before dependent experiments. Completed code
is not a completed validation task. Retain failure, convergence and eligibility
counts, and score pass-through intervals rather than excluding difficult data.

## Acceptance contract

Target fresh controlled mean SNR >=15 dB (20 dB stretch), accompanied by separate
blink/horizontal results and participant-level uncertainty. Paired-clean worst
participant/record mean relative modification <=1%, alpha/beta <=0.5 dB,
normalized covariance <=0.02, and declared comparator Pearson deterioration
<=0.005. These are engineering bounds. OSF has no paired-clean SNR.
Unknown Klados electrode/participant metadata never becomes inferred anatomy.

## Execution record

No v2 numerical runs have completed at ledger creation. New results must record
exact Git SHA, configuration/source/split/parent hashes, seed, environment,
hardware, checkpoint hashes and saved Kaggle kernel outputs.
