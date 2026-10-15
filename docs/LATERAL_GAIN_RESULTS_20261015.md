# Lateral gain repair: completed development pilot

## Run provenance

`lateral-gain-review-003/3`, source
`5b59c6dc3f9ed5c50281e7fadf68e6680ab46af8`, completed all 20 examples and eight
predeclared arms. 325 Kaggle tests passed with 14 warnings in 23.00 seconds.
Stage runtime53.787672seconds includes cache verification, diagnostics, scoring
and saving; it excludes runtime VMD because the same archived modes were reused.
All 37 manifest payloads and completion/source identity were recovered and
verified. Manifest SHA256:
`f6695a6fa2faca30edc838f58051b8d4b36077911a41746082082f4614d0147d`.
Figure inspected: labels, seven-method legend and three conditions are readable.

Parents: coupled-pilot002/2 and fixed-frontal review002/2, with exact manifest
identities in configs/lateral_gain_review.json. Zero new decompositions. Posterior
and middle outputs are bitwise matched. Reserved sources remain unopened.
This is a development-exposed two-recipient/two-donor engineering experiment.

## Result and provisional candidate

The best balanced repair candidate is `vmd_anchored`: the final frontal field
gain is 0.75*raw_shared_gain + 0.25*VMD_shared_gain. Both component gains use
observed EEG/references only and the existing [0,8] bound. This is a conservative
project adaptation with a predeclared weight, not a published optimum, label
router, or new VMD algorithm. VMD remains frontal-only K3/alpha2000. No whole
mode or residual is discarded. Full reference-field artifact correction is
composed against the original EEG once.

### Lateral frontal channel-first mean output SNR, dB

Input levels are nominal whole-montage mixture levels, not frontal input SNR.
Each value averages the two eligible development recipient records.

| Nominal input | Old VMD shared | Anchored VMD | Raw shared ridge |
|---|---:|---:|---:|
| -5 dB | 10.655937 | 13.885189 | 14.061627 |
| 0 dB | 14.154767 | 18.488301 | 18.635700 |
| +5 dB | 19.472830 | 22.965294 | 22.682389 |

Mean pooled-window lateral frontal output SNR is respectively
10.142813/13.690511/13.756186 at -5, 13.092838/18.214465/18.467640 at0, and
18.405696/22.688881/22.730702 at+5, ordered old VMD/anchored/raw. These are
separate aggregation conventions; do not choose one opportunistically.

The previously highlighted severe lateral example, lemon:sub-032306, improved
from8.812491 to13.776868dB channel-first (pooled8.254118 to13.042133dB).
Raw shared ridge is13.486290dB channel-first/12.883807pooled on that record.
Pearson CC improved from0.917674 to0.974111, versus raw0.973957.

### Contaminated-condition aggregate

| Scope/method | Channel-first SNR dB | Mean pooled-window SNR dB | MSE V² | Pearson CC |
|---|---:|---:|---:|---:|
| Frontal, old VMD | 23.040945 | 21.768334 | 7.376812e-13 | 0.987553 |
| Frontal, anchored VMD | 21.971702 | 21.327173 | 4.011316e-13 | 0.993456 |
| Frontal, raw shared | 20.973930 | 20.488443 | 4.343346e-13 | 0.992797 |
| Whole, old VMD hybrid | 19.398750 | 15.848823 | 1.069891e-12 | 0.979788 |
| Whole, anchored VMD hybrid | 19.150952 | 16.271499 | 9.929115e-13 | 0.981140 |
| Whole, raw shared hybrid | 18.921014 | 16.193570 | 1.000550e-12 | 0.980987 |
| Whole, raw axis hybrid | 18.963855 | 16.226222 | 9.949560e-13 | 0.981001 |

The anchored arm trades some high blink/mixed log-SNR of the old VMD arm for
better lateral behavior and lower aggregate MSE/higher correlation. It is not
the winner of every metric or condition. Its advantage over direct regression
is small and not established as statistically generalizable by two pairs.

## Preservation and rejected candidates

No saved anchored-arm region/record CC check fails against either raw shared or
raw separate-axis controls. Both idealized zero-driver clean controls have zero
recorded modification, alpha/beta error and covariance error. This does not
establish preservation on native resting EEG or a fresh held-out cohort.

Reference-supported residual completion improves some lateral outcomes but still
fails lateral CC checks, including0.0269568 deterioration on the severe second
recipient and0.0193499 on the first recipient at nominal0dB. It is not qualified.
The projected-field scalar and axis proposals from version2 remain rejected;
their complete negative results stay in this run's tables and predictions.

## Interpretation, literature and next checkpoint

The residual diagnostics show that mode-only gain can omit reference-associated
activity. They do not establish that the entire residual is ocular. The successful
anchor reduces the effect of an incomplete representation by borrowing the more
stable raw-field gain. Most of its correction strength is explicitly regression-
derived. VMD's incremental benefit must still be tested with broader grouped
development, not inferred from the architecture name.

Literature evidence and adaptation boundaries are in
LATERAL_REPAIR_REVIEW_20261015.md. Papers support treating ocular generators
separately and avoiding whole-component removal; none proves the weight0.25 or
this particular hybrid superior. Next: broader source-separated development
evaluation of the anchored candidate versus both direct controls, then native
OSF blink/lateral suppression and resting preservation. Do not tune on reserved
confirmation sources. Teacher qualification and neural training remain closed.
