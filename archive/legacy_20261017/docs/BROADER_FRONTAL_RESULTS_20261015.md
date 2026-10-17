# Broader fixed frontal VMD: measured development results

## Scope and provenance

The five-arm comparison completed on Kaggle as
`satyapaladugu/vmd-broader-frontal-review/1`, run
`broader-frontal-review-001`, source
`f126300611889d83d6b3427d4254fdcde731ad40`.
It passed330 tests in23.33s. Stage runtime614.224904s includes decompositions,
correction, evaluation and saving; it is not an inference-only benchmark.

The exact parent is `satyapaladugu/vmd-broader-coupled-corpus/1`, source
`4f1921c7bf2933ac72aa957ca8fb5dc9d5447b4d`, manifest
`9ceae64b9abe885d76526831500aa8588a6e0589fe60b07659061aa93a7a43de`.
All398 corpus payloads were recovered and verified before launch.

There are320 examples:32 development recipients, five ocular donors, and the
five existing source buckets. Each recipient contributes one5.12-second EEG
window with ten variants:idealized clean plus blink/lateral/mixed at nominal
whole-input SNR-5/0/+5dB. These are not320 independent recordings. References
are low-ocular LEMON EEG, not proven biologically clean EEG. The ocular field
is regression-derived, with shared EOG/artifact scaling. Generator bias remains.
All sources are development-exposed; reserved confirmation remains closed.

VMD applies only to verified frontal rows, fixed K3/alpha2000. The anchored
gain is75% raw plus25% selected-mode gain. Posterior/middle correction is the
same separately estimated direct ridge field in every corrected arm. These
results do not select a posterior ICA/MWF winner or validate frontal-context
communication. Unknown anatomy passes through.

## Primary paired results

Values below are copied from Kaggle's saved tables. Channels are averaged
within source examples before source-balanced summaries. Output SNR and SNR
improvement are distinct. The pooled column is mean pooled-window SNR, a
separate aggregation. Regional dB values are never added.

| Frontal method | Frontal output SNR | Whole output SNR | Whole SNR improvement | Whole pooled-window output SNR |
|---|---:|---:|---:|---:|
| Shared direct ridge |18.182199|17.785746|14.564297|15.211023|
| Separate-axis direct ridge |17.071352|17.533194|14.311744|15.068759|
| Original VMD gain |17.079584|17.537370|14.315921|14.852226|
| Anchored VMD gain |18.238703|17.799616|14.578166|15.116150|

The same posterior output SNR is15.473721dB in every corrected arm; the same
central/shared output is19.203094dB. They cannot explain differences between
the frontal arms. Whole-input identity has channel-first SNR3.221449dB and
mean pooled-window SNR approximately0dB; this difference follows aggregation
over unequal channel energies, not a change to injected nominal input levels.

| Method | Frontal MSE | Frontal ordinary Pearson CC | Whole MSE | Whole ordinary Pearson CC |
|---|---:|---:|---:|---:|
| Shared direct ridge |2.535332e-12|0.981665|2.957932e-12|0.974691|
| Separate-axis direct ridge |2.817168e-12|0.978816|3.022279e-12|0.974045|
| Original VMD gain |4.454140e-12|0.974028|3.396606e-12|0.972960|
| Anchored VMD gain |3.559869e-12|0.980432|3.192343e-12|0.974413|

Anchored VMD improves the original VMD arm but does not dominate shared
direct ridge. Its slightly higher channel-first SNR coexists with higher MSE,
slightly lower CC and lower pooled-window SNR. A mean log-ratio score can
obscure a few large errors; those errors remain in all tables.

## Lateral-eye frontal results

Input labels are nominal whole-montage mixture levels; actual frontal input
SNR differs and is saved per record.

| Frontal method | Input-5dB | Input0dB | Input+5dB |
|---|---:|---:|---:|
| Shared direct ridge |11.376975|15.785051|19.376854|
| Separate-axis direct ridge |10.962807|14.833203|17.618509|
| Original VMD gain |9.219142|13.291172|16.743055|
| Anchored VMD gain |11.299087|15.732839|19.450830|

These are channel-first frontal output SNR values. The anchored mechanism
repairs much of the original VMD lateral loss, but shared ridge remains
slightly better at the two more contaminated levels. This is not evidence
that VMD is the superior lateral method.

## Supplementary uncertainty

The predeclared paired recipient/donor Poisson-multiplier bootstrap uses2000
draws and seed42. It is a project approximation, not the paper-native OSF
participant permutation test. There are only five ocular donors.

Anchored minus shared direct ridge:

| Scope | Output-SNR difference | Supplementary95% interval |
|---|---:|---:|
| Whole |+0.013870dB|[-0.080863,+0.128170]dB|
| Frontal |+0.056504dB|[-0.360475,+0.552965]dB|

Both intervals include zero. Incremental VMD benefit over the strongest
matched shared-ridge control is unestablished. Although the comparison with
the weaker separate-axis control has positive SNR intervals, that cannot
establish superiority over shared ridge. The legacy `favors_candidate` column
expresses point-estimate direction, not exclusion of zero by the interval.

## Preservation, solver failures and reporting repair

All idealized clean-record preservation screens passed. These controls have
zero injected ocular references and do not substitute for native resting EEG.
Some per-record CC-deterioration screens failed. Recipient/donor mean
diagnostics remain to be generated; per-record screens will remain visible.
No teacher has qualified, and no neural training follows from this result.

There were4160 frontal channel decompositions and three iteration-cap flags:

| Example | Channel | Solver observation |
|---|---|---|
|sub-032321-study03_p24_prep-blink-snr+0|F3|hit iteration cap;1999 recorded iterations|
|sub-032325-study02_p10_prep-mixed-snr-5|AF3|hit iteration cap;1999 recorded iterations|
|sub-032329-study01_p02_prep-mixed-snr-5|AF4|hit iteration cap;1999 recorded iterations|

The old method passes these rows through. They coincide with large CC drops,
so a separately named raw-regression fallback is a reasonable next mechanism
test. It may act only on solver flags, with all other rows bitwise fixed; the
original five arms remain intact. No higher iteration cap or K/alpha search is
being introduced.

A reporting-contract error also occurred:the new score table used Boolean
False for success, while the existing uncertainty helper counts non-null
failure reasons. Consequently its failure counters incorrectly count successful
rows. The primary metric computation and bootstrap weighting do not depend
on these counters. Preserve the original run as evidence and repair the
contract to null-for-success/nonempty-reason-for-failure. Reproduce the old
paper metrics and uncertainty values on Kaggle before accepting the repair.

## Artifact recovery and next experiment

Summary/table payloads were downloaded with manifest checks. Full1947-payload
recovery is pending after TLS/connection/HTTP failures; this is an artifact
transport problem, not a failed numerical kernel. The watcher correctly
stopped with primary review required after exhausting retrieval attempts.
The retriever has been changed to bounded in-flight work, connection reuse,
safe error diagnostics and progress snapshots; byte checks remain required.

Next:verify all pinned parent bytes on Kaggle before the cached repair tests, run reporting repair and the
solver-failure-only fallback on Kaggle, inspect unchanged thresholds and source
means, then evaluate eligible frozen candidates on native OSF. Original native
trial cache availability and remaining checks are documented separately in
NATIVE_CACHE_REUSE_ASSESSMENT_20261015.md. The17.799616dB development mean does
not satisfy the fresh-confirmation requirement or establish production readiness.


Transport recovery update:1396/1947 parent payloads are locally byte-verified. An explicit cached-review-only metadata route pins exact completed run/SHA/manifest and verifies all present local files. The Kaggle runner must verify EVERY mounted parent payload before tests or analysis; a partial local receipt is never labelled complete. Local download continues separately. No source/split/recipe/acceptance check is removed.
# Cached evidence repair and failure-only fallback: completed follow-up

`broader-cached-review-001/1`, source
`5ad2c35c4070026bb7ac6be995679dd326a6a4a9`, completed with332 tests
(14 warnings,20.39s) and131.191077s stage runtime. It consumed the exact
version1 corpus and five-arm parent, verifying every parent byte on Kaggle.
No new VMD decomposition, calibration fit or recipe selection occurred.
All original paper metrics and original uncertainty point estimates/intervals
reproduced at the declared numerical tolerance. The corrected reporting contract
counts three candidate failure rows and zero raw-control failure rows.

The new arm chooses saved raw-shared predictions only for the three frontal
channels flagged by the solver. These were blink/mixed cases, so lateral
predictions remain unchanged. Original predictions and failed-run evidence stay
archived separately.

| Measure, contaminated development records | Original anchored | Raw fallback arm | Matched raw shared |
|---|---:|---:|---:|
| Channel-first mean whole output SNR,dB |17.799616|17.804727|17.785746|
| Channel-first mean frontal output SNR,dB |18.238703|18.261209|18.182199|
| Whole SNR improvement,dB |14.578166|14.583278|14.564297|
| Mean pooled-window whole output SNR,dB |15.116150|15.200363|15.211023|
| Frontal MSE,source amplitude squared |3.559869e-12|2.617076e-12|2.535332e-12|
| Frontal ordinary Pearson CC |0.980432|0.981111|0.981665|

Fallback-minus-raw channel-first SNR differences remain inconclusive:
whole+0.018981dB, supplemental source-aware95% interval[-0.071359,+0.132357];
frontal+0.079010dB, interval[-0.319093,+0.570154]. These intervals include zero.
There is no demonstrated incremental VMD benefit over matched direct regression.

Worst-recipient mean frontal CC deterioration against raw shared drops from
0.013147884 to0.006224758 (recipientsub-032329), but still fails the unchanged
0.005 limit. Against raw axis the corresponding worst-recipient statistic is
0.004534726. Both comparators remain visible; reporting the passing comparator
alone would hide a failure. Whole-head means cannot excuse frontal loss.
The zero-reference idealized clean controls do not establish native preservation.
Teacher qualification remains false; no new neural training is authorized by
this result.

The six-panel condition plot was visually inspected: labels and curves are
readable; its vertical axis explicitly uses mean pooled-window SNR, which is
different from the channel-first target above. Lateral performance is the
remaining weak condition. Packaging of the original parent completed on Kaggle;
full local recovery is still pending because of download timeouts/HTTP502.
