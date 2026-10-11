# Fixed frontal VMD campaign: execution and task ledger

Campaign: `vmd-frontal-replication-20261011`, configuration
`configs/frontal_replication.json`. Branch remains
`research/vmd-eog-removal-notebooks`. Every notebook pins an exact published
SHA and persistent parents. Authoring/static inspection is local; all tests,
waveform work, fitting, evaluation and figures are on Kaggle.

## Locked policy

AutoVMD is disabled. On 11 October the authenticated Kaggle status for
`satyapaladugu/vmd-band-search-f0-k4-a1000` was CANCEL_ACKNOWLEDGED.
Historical caches remain immutable. The archived guard replay reported ERROR;
it is owned by the separate historical replay workflow, not this campaign.

All VMD calls are frontal only, with names checked against declared anatomical
metadata. Unknown/E1 and posterior/central/FC names are not assigned frontal.
Fixed K3/alpha2000 and fixed K4/alpha2000 are the only project recipes. No K/alpha
optimizer or neural screening is called. Completed caches are imported by exact
manifest, waveform, solver, corpus, split and scope identities.

## Numerical experiment

Inputs: corpus/2, readiness/1, passing frontal fixture, K3 cache/1 and K4 cache/1.
Pilot: one recipient per frozen bucket, clean and blink/lateral/mixed at 0 dB.
Full: first deterministic window per recipient/condition/-5/0/5 dB (320 examples).
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

Partial selection screens clean modification, alpha/beta and covariance using
training sources only. Family-wise heldout reports include CC deterioration
against the independently selected direct comparator. SVM has one predeclared
K3 correction recipe; nested tuning of crossfit detector predictions is excluded.
Native preservation, uncertainty and final family selection remain required.

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
- [ ] Validate the whole-montage interface and expanded checks on Kaggle.
- [ ] Complete pilot; inspect waveforms and recovery/clean modification.
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
