# Fixed frontal VMD campaign: execution and task ledger

Campaign: `vmd-frontal-replication-20261011`, configuration
`configs/frontal_replication.json`. Branch remains
`research/vmd-eog-removal-notebooks`. Every notebook pins an exact published
SHA and persistent parents. Authoring/static inspection is local; all tests,
waveform work, fitting, evaluation and figures are on Kaggle.

## Locked policy

AutoVMD is disabled. On 11 October the authenticated Kaggle status for
`satyapaladugu/vmd-band-search-f0-k4-a1000` was CANCEL_ACKNOWLEDGED.
Historical caches remain immutable. The earlier archived replay ERROR is retained;
the repaired cache-only replay guard17-cache-replay-003 reproduced17.92721 dB.
It remains a separate unknown-montage, EOG-assisted historical benchmark.

All VMD calls are frontal only, with names checked against declared anatomical
metadata. Unknown/E1 and posterior/central/FC names are not assigned frontal.
Fixed K3/alpha2000 and fixed K4/alpha2000 are the only project recipes. No K/alpha
optimizer or neural screening is called. Completed caches are imported by exact
manifest, waveform, solver, corpus, split and scope identities.

## Numerical experiment

Inputs: corpus/2, readiness/1, passing frontal fixture, K3 cache/1 and K4 cache/1.
Pilot: one recipient per frozen bucket, clean and blink/lateral/mixed at 0 dB.
Full heldout evaluation: first deterministic window per
recipient/condition/-5/0/5 dB (320 examples).
Both score **every verified frontal channel** and leave every other channel alone.

Matched controls: identity and direct joint HEOG/VEOG ridge. Mode mechanisms:
whole selected mode, reference-projected mode, temporal interval projection,
mode-domain SOBI projection. Paper adaptations: 2022 selector structure and
2024 fixed-recipe SVM/SOBI/ApEn; exact-reproduction limitations are in
FRONTAL_PAPER_AUDIT.md. No targets or known mixture artifacts enter inference.

Pilot: triggers .4/.6, component thresholds .2/.4, strengths .5/1, instantaneous
ridge .01. Full: declared four association/strength settings for mode arms;
direct comparator additionally covers declared three lag banks and penalties.
This is a bounded correction study, not a restarted parameter search. Direct
regression is explicitly window-adaptive, not falsely described as calibration
coefficient fitting. Event support uses unchanged HEOG/VEOG standardized by
unscored calibration: 3 SD, 100-ms expanded support, tapered edges. These are
declared project choices, not paper constants or native zero-correlation targets.

The full follow-up adds a bounded lag/ridge ablation to the three projected
mode families: trigger .4, component threshold .2, strength 1, all declared lag
banks and ridge penalties. It does not multiply these settings across the full
threshold/strength grid. Projection cache keys include lag and penalty so a
different lag bank cannot accidentally reuse an instantaneous estimate.

Partial selection screens clean modification, alpha/beta and covariance using
training sources only. Family-wise heldout reports include CC deterioration
against the independently selected direct comparator. SVM has one predeclared
K3 correction recipe; nested tuning of crossfit detector predictions is excluded.
Native preservation, uncertainty and final family selection remain required.

In the follow-up, K3 and K4 are separate fixed comparison arms. Correction
parameters are selected within an arm; K is never selected by this workflow.
The single SVM recipe is evaluated as predeclared, without selecting an identity
fallback from other outer-fold detector predictions that may have seen the
current heldout fold. Its preservation failures are reported, not hidden.

Persist candidates, channel scores, per-example recovery shards, fold/exclusion
identities, per-axis association, solver diagnostics, mode frequency/bandwidth
information and failures. The pilot saves every candidate prediction. Partial
or failed runs retain completed shards; resume must match all identities.

## Acceptance and sequence

Output SNR >=15 dB, stretch 20 dB. Preserve worst-recipient mean clean change
<=1%, alpha/beta <=.5 dB, covariance <=.02 and CC deterioration <=.005 versus
direct comparison. These engineering gates are separate from paper metrics.
Clean-gate success alone cannot qualify teacher or posterior integration.

Frontal fixture -> pilot -> full grouped frontal evaluation -> native frontal
validation -> frontal evidence review -> posterior integration -> EEG-only model
comparison -> freeze/external evaluation -> deployment. No queued automatic
transition into posterior or neural work bypasses scientific checkpoints.

## Target task list

- [x] Verify stopped AutoVMD saved-version cancellation; retain its evidence.
- [x] Inspect author code and record reproduction gaps and reference access.
- [x] Author fixed-frontal modules, anatomy guards and explicit adapters.
- [x] Author numerical tests, config and Kaggle notebooks.
- [x] Initial Kaggle numerical checks: frontal-fixture-001, 99 passed at e8bd484.
- [x] Kaggle whole-montage checks: frontal-fixture-002, 101 passed at 9419b73.
- [x] Native interfaces/unit checks: frontal-fixture-003, 104 passed at ef4a4c8.
- [x] Validate separate fixed arms and nonleaking fixed-SVM evaluation on Kaggle.
- [x] Latest foundation checks: frontal-fixture-004, 106 passed at 638d496.
- [x] Strict metric coverage/worst-record reporting: frontal-fixture-005, 108 passed at 08f33ac.
- [x] First pilot completed: 20 examples/72 candidates, zero failure rows.
- [x] Complete separate-K correction screen and calibrated-detector waveform inspection.
- [ ] Resolve or retrieve delayed smaller repaired pilot without replacing its evidence.
- [ ] Complete full source-excluded frontal comparison and uncertainty.
- [ ] Native OSF whole-trial frontal suppression/rest preservation.
- [ ] Obtain full 2020 method; establish author parity where eligible.
- [ ] Deliver frontal results notebook and qualification evidence.
- [ ] Integrate versioned posterior results without repeating their search.
- [ ] Compare no/common/signed plus artifact frontal context; subtract once.
- [ ] Compare compact EEG-only CNN/TCN/TCN-BiGRU and optional qualified teacher.
- [ ] Freeze and evaluate eligible external ocular/BCI/ERP datasets once.
- [ ] Verify CPU/ONNX loading, whole-record overlap-add and 10-50-channel scaling.
- [ ] Final report: K/alpha, mode vectors/bins, preserved EEG, SNR and limitations.

The ledger describes implementation and checkpoints, not measured success.

## Verified evidence

`frontal-fixture-001` (Kaggle `vmd-frontal-contracts`, version 1) completed at
`e8bd4848b462f03222ceac8ca9077bf9ca2feb40`. Retrieved tests report 99 passed,
14 warnings. The synthetic fixture verifies F3-only decomposition, residual
reconstruction and closed confirmation sources. It does not establish measured
ocular cleaning accuracy. The next fixture checks the whole-montage interface
before launching the fixed-recipe pilot.

`frontal-fixture-002` (same kernel, version 2) completed at
`9419b733771431e88fe3f6e7354e43048efec3df`, with 101 passing tests and 14 warnings.
`frontal-pilot-001` was submitted at that same SHA, attached to five exact
parent versions, including contracts/2. It compares 72 fixed-recipe correction
candidates on 20 development examples. Its SVM uses the first absolute-SD
feature; the follow-up replaces that feature with a same-channel unscored
calibration-SD ratio before native transfer. Preserve both versions explicitly.

Native implementation is authored in `frontal_native.py`, notebook 26. It
requires a completed **full** frontal comparison, the original bundle and the
checksum-pinned restoration version. It reuses family-wise fold recipes without
native retuning; source-excluded calibrated SVM checkpoints are mandatory.

`frontal-fixture-003` (contracts version 3) completed at
`ef4a4c873fad5a88f0ea2de0cf10061ec8c6b0b0` with 104 passing tests and 14 warnings.
That verifies unit-invariant detector features, native donor exclusion and cache
scope guards. A later commit additionally separates K arms and removes any SVM
fallback decision from crossfit predictions; its regression check runs in the
next pilot. These checks are software evidence, not native waveform results.

First-pilot source-balanced frontal output SNR: direct 12.6737 dB,
selective VMD 12.2655 dB, projected SOBI 12.2389 dB. Whole-mode removal was
1.9262 dB. This is five-recipient development evidence with provisional earlier
selection, not confirmation accuracy. VMD has not shown benefit over regression;
CC deterioration for projected VMD was .00572, above the .005 engineering limit.
The nominal mixture is 0 dB across the montage; actual frontal identity SNR was
-5.0547 dB. Lateral is weaker than blink/mixed. No target or preservation limit
is relaxed. The second pilot remains separate and fixes units/selection logic.

To bound CPU cost, the follow-up full correction bank is screened on the same
20-example source-balanced development subset, with each outer fold's recipes
selected using the other four buckets. A separate `frontal-evaluation` stage
reuses those recipes unchanged across all 320 eligible examples. This is a
limited training-source selection design, not the five-inner-fold neural search.
The screen saves `evaluation_complete=false`; native validation refuses it.
All 320 selected predictions and channel/record outcomes are saved by evaluation.
Notebook 28 executes the screen with `configs/frontal_screen.json`; notebook 29
executes the full evaluation. Missing artifacts or exposed recipients/donors
stop the dependent stage. No correction choice is retuned on its scoring bucket.

Notebook 27 provides saved mode vectors, residuals, HEOG/VEOG support, selected
waveforms, reusable-interface parity and evaluation-only lag/energy-coverage
diagnostics. Its diagnostic reference design follows the mixture construction
and cannot substitute for native evidence.

`frontal-fixture-004` (contracts version 4) completed at
`638d496c244bdf8a2ae02ce3b4415949e22e029e`: 106 tests passed, 14 warnings.
The repaired 72-candidate pilot and 708-candidate correction screen were both
RUNNING at the live check on 11 October. Their completion and results are not
yet verified. This screen varies only correction parameters within the two
fixed K/alpha arms; it is not AutoVMD.

The full evaluation additionally writes `frontal_controlled_gates.json` and a
readable `frontal_controlled_review.md`. Every arm must cover the identity
example/channel rows, preserve finite accuracy and clean metrics, and satisfy
the unchanged engineering limits. A short bad clean record cannot disappear
inside its participant average. Controlled gates never authorize integration
before native validation. Undefined values propagate through all summary levels.

`frontal-fixture-005` (separate review-contracts kernel, version 1) completed at
`08f33ace0979e589be24969ceeca80071984ed45`, with 108 passing tests and 14
warnings. Retrieved artifacts verify the same run identity and passing checkpoint.
Future notebook setup saves the run identity before clone/dependency installation
and reports each bootstrap phase, so setup failures remain attributable and
recoverable. This does not change a running notebook's pinned source.

## Bounded job orchestration

From `VMD_EOG_Removal`, run:

```powershell
python tools/frontal_queue.py
```

This is local orchestration, not local numerical execution. It checks Kaggle
every 45 seconds and saves append-only events under
`results/vmd-frontal-replication-20261011/orchestration/events.jsonl`.
Completed outputs must match their source/run identity, checksums, passing tests,
completion state and closed-confirmation policy before they become parents.
The first completed repaired pilot or correction screen launches inspection;
the correction screen launches frozen full evaluation; full evaluation launches a two-record native pilot with the
checksum-pinned restoration version. Actual saved parent versions are resolved
from their submission specifications rather than assumed.

The queue ends at frontal review. An API exception or kernel failure stops
dependent submissions for primary-agent diagnosis. It cannot launch AutoVMD,
posterior integration, teacher construction, neural training or confirmation.
Do not start two copies. After diagnosis, rerunning resumes using verified saved
artifacts and existing run IDs; it does not overwrite completed or failed runs.

The completed `frontal-screen-001` evaluated 708 correction candidates on 20
development examples, with zero failure rows and 499.97 seconds numerical-stage
runtime. Retrieved source-balanced output SNR: direct 12.7468 dB, projected
VMD K3 12.1386 dB, projected VMD K4 12.3030 dB. Their CC deteriorations versus
direct were .01392 and .01327, respectively, outside .005. These provisional
five-recipient results do not qualify VMD or meet the 15 dB target. The immutable
full 320-example evaluation was launched at `1d18d81` without retuning.

Inspection may use this completed screen rather than waiting for the smaller
repaired pilot. Because a full screen saves score tables instead of every grid
prediction, the inspection verifies all five paired metric vectors against
archived selected-arm scores, including null/perfect-score statuses and source
units. Pilot NPZ waveform parity remains an additional check when available.

The native job verifies the full development SET/FDT ledger before choosing its
two-record pilot. Reserved records are rejected before verification. Every trial
retains all successful/failing window decisions plus the deterministic first
window's frontal modes and residual, and the complete corrected waveform.
These native outputs remain recoverable evidence even when an accuracy gate fails.

`frontal-evidence-001` completed at `c40721a`, with 109 passing tests and 52
selected-arm metric-parity checks. No new VMD decomposition or AutoVMD ran. Its
four mode-vector archives and plots were retrieved. The preselected lateral Fz
example exposed 6.39% interval-energy coverage versus 94.74% for blink. Anchored
lag correction improved that lateral diagnostic but ungated projection altered
clean EEG; no new inference recipe or gate relaxation follows from this example.

## Repair checkpoint — 11 October 2026

Full version 001 failed at example 292 (291 saved) because float32 summation
failed the unchanged reconstruction tolerance. Audit 001 verified 8,320 channel/
recipe checks: one float32 failure, zero float64 failures, zero solver failures.
Audit 002 exercises the actual repaired reader and regression tests before the
queue can launch full version 002. Its exact completed notebook version is
attached as a repair-evidence parent. Earlier failed outputs remain preserved.
Run python tools/frontal_queue.py to monitor the repair/full/native chain;
do not start a second copy while the PID in orchestration/queue.pid is active.


## Deadline campaign progress

The latest source/data/metric fixtures have passed on Kaggle; the best new
controlled frontal VMD result is17.11281 dB, not a native reconstruction claim.
Native frontal preservation still fails, so regional integration and new
EEG-only models have not begun. Gain-adjusted event support is a bounded repair
experiment at fixed K3/2000; it is not a restarted AutoVMD selector.

Confirmed12:18 UTC: the old AutoVMD search kernel remains
CANCEL_ACKNOWLEDGED. Only completed immutable decomposition caches are reused.
