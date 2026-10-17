# Paper-grounded evaluation protocol v1

User choice, 2026-10-10: **VMD papers + OSF/EEGOAR-Net + Klados**.
This supersedes descriptions that presented project preservation gates as
literature-standard evaluation. Older runs remain immutable development evidence.
The running searches use their archived selection objective. They are not a
completed replication of the protocol below. No metric changes are retroactively
attributed to their source SHA.

## Sources and exact mapping

| Evidence | Source location | Implemented metric | Status |
|---|---|---|---|
| Paired VMD recovery | [SVM-IVMD-SOBI, Sensors 2024](https://www.mdpi.com/1424-8220/24/5/1642), section 3.1, equations 15–18; [publisher PDF](https://mdpi-res.com/d_attachment/sensors/sensors-24-01642/article_deploy/sensors-24-01642-v2.pdf?version=1709543478), pages 9–10 | RRMSE, MSE; standard Pearson plus separately identified literal printed equation; explicit-scale PSNR | Verified equations; inconsistencies disclosed below |
| VMD real EEG | Same paper, section 3.2, equation 19 | Absolute PSD change `abs(PSD_in - PSD_out)` | PSD-estimator choices explicitly adapted |
| Native ocular suppression | [EEGOAR-Net author manuscript](https://uvadoc.uva.es/bitstream/handle/10324/78224/Calibration-free-ocular-artifact.pdf?isAllowed=y&sequence=1), section 2.4.1 | Ordinary absolute Pearson EEG–HEOG for lateral; EEG–VEOG for blink and vertical | Verified section; no lag maximization |
| Native rest preservation | Same section | RMSE between corrected and original resting EEG; resting PSD; relative delta/theta/alpha/beta1/beta2 power | Verified durations and bands; unspecified estimator choices disclosed |
| Native statistics | Same section | Two-sided paired permutation, 10,000 repetitions, Bonferroni | Participant-level summaries; declared comparison family required |
| Native calibration/scoring | [Kobler author evaluation demo](https://github.com/rkobler/eyeartifactcorrection/blob/master/demo_main.m), lines 116–145 | Rest RMSE; condition-specific ordinary Pearson with low-pass EOG derivatives | Author code verified; no reconstruction SNR |
| Paired ground truth | [Klados and Bamidis, 2016](https://doi.org/10.1016/j.dib.2016.06.032), sections 1–2 | Compare correction against provided pre-contamination EEG | Dataset paper; does not define a new SNR/CC/RRMSE formula |

## Equations and distinctions

For one channel, target `s`, correction output `y`, input `x`, and `N` samples:

```
MSE           = mean((s-y)^2)                           [IVMD Eq17]
RMSE          = sqrt(MSE)                               [EEGOAR rest; paired derivation]
RRMSE_time    = sqrt(mean((s-y)^2) / mean(s^2))          [IVMD Eq16]
Pearson CC    = cov(s,y) / sqrt(var(s)*var(y))
SNR_energy_dB = 10*log10(sum(s^2) / sum((s-y)^2))        [explicit project SNR definition]
SNR_gain_dB   = SNR_energy_dB(y,s) - SNR_energy_dB(x,s)
PSNR          = 10*log10(peak^2 / MSE)                  [IVMD Eq18]
native |r|    = abs(Pearson(corrected EEG, original condition-specific EOG))
rest RMSE    = sqrt(mean((corrected rest - original rest)^2))
```

Output SNR, SNR improvement, PSNR and a linear input RMS ratio are separate
quantities. The 15/20 dB user target refers to **energy-ratio output SNR**, never
PSNR. It is a project acceptance target, not a claim about a published threshold.
Report each channel first, then channel means for each example. Also preserve
example/record tables. Participant-balanced estimates are supplementary research
summaries with their aggregation specified; they cannot be described as the
single-channel VMD paper's exact sampling protocol.

The IVMD paper's printed Eq15 repeats target variance in the denominator. The
literal expression is `cov(s,y)/var(s)` and can exceed one. Our primary CC is
ordinary Pearson; the literal expression is separately available and labelled.
This correction is disclosed rather than silently called an exact replication.
Eq13 is a **linear RMS input ratio**, not dB. Eq18 uses an 8-bit peak of 255.
Applying it directly to volt-scaled EEG gives unit-dependent, inflated-looking
numbers. PSNR therefore requires an explicit peak and scale justification and
is marked unavailable in the production protocol until that scaling is resolved.

The IVMD experiments use 500 examples per input-ratio level and mean/SD per
level. Our controlled donor/recipient corpus, input levels and source partitions
differ. Results will be labelled an adaptation; no published number is treated
as a directly comparable benchmark without matching those conditions.

## Native OSF evaluation

Use separate rest/lateral/vertical/blink labels and original HEOG/VEOG. Both
before/after correlations use the unchanged EOG. Keep per-channel results and
verified regional summaries. Do not compute paired SNR/RRMSE against the
uncorrected native EEG or a teacher target and describe them as reconstruction.

EEGOAR averages four regions with channel assignments in supplementary section
S.II. That supplement could not be retrieved. Our frontal/posterior/shared
algorithm regions are not asserted to reproduce its four regions. Keep
per-channel paper tables and whole-montage summaries; exact four-region
replication remains unverified rather than guessing the channel sets.

The EEGOAR manuscript describes five folds with nine test and 36 training/
validation participants per fold. Our audit found 45 sessions but 39 global
OSF participant identities. We retain grouped verified participant identities;
we do not assume sessions are independent people to match the paper's numbers.

The Kobler demo concatenates condition-specific scoring trials before ordinary
Pearson/RMSE evaluation. Computing each 5.12-second cache window and averaging
is an **adapted window protocol**, not identical to that concatenation. Saved
tables must state which unit is used; overlapping windows are not independent
participants. Existing caches lack the 8-second windows used in EEGOAR's chance
analysis. They cannot establish its chance threshold.

EEGOAR specifies Welch 2-second segments and 1-second overlap, with delta
1–4, theta 4–8, alpha 8–13, beta1 13–19 and beta2 19–30 Hz. Our explicitly
declared estimator uses Hann, constant detrending and half-open frequency bins,
with relative power normalized over 1–40 Hz. These details are adaptations where
the inspected section does not specify exact implementation. Save full PSD and
before/after relative powers, not only a thresholded summary.

The paper's chance analysis uses five randomly selected participants, 8-second
ocular and resting windows, 5,000 repetitions and the 95th percentile. Its exact
aggregation over those five participants is not established from the inspected
text. Chance-level equivalence will remain **not established** until a separate
8-second eligible-source implementation declares and checks that choice. Do not
substitute a fixed arbitrary correlation threshold.

For significance, use one paired participant summary per comparison, 10,000
sign-flip permutations, a two-sided mean-difference statistic and Bonferroni
over a predeclared family. Monte Carlo sign flips and the conservative +1
probability estimate are disclosed choices. Confidence intervals are supplementary;
they do not replace the paper's comparison method.

## Klados provenance

The source paper describes 54 recordings from 27 participants, 19 electrodes,
200 Hz, 0.5–40 Hz EEG, 0.5–5 Hz EOG, and `pure + a*VEOG + b*HEOG` mixtures.
Our local conversion has 53 records and 5,401 samples with no verified subject
or channel labels. A list in the publication does not prove the conversion's
array order. Keep its paired results legacy/development-exposed, report record
counts and missing mapping, and never fabricate participant-level independence.
RMSE/MSE require known units; report unknown original units as unknown.

## Supplementary project diagnostics and model gate

Lag-maximized EOG association, joint-EOG R², clean relative modification,
alpha/beta dB error and relative covariance error remain **supplementary
project diagnostics**, with no attribution to these papers. The earlier 1%,
0.5 dB, 0.02 and 0.005 thresholds are project criteria, not published standards.
Preserve them as user-requested engineering gates and report them separately.
No neural progression may rely on the old metrics alone: paper-grounded evaluation
and its source/aggregation evidence must be complete first.

The complete-trial extension is implemented in `native_protocol.py` and notebook
07b. It preserves the source's LPF EOG derivatives if present and otherwise
labels raw-reference fallback. It applies the campaign's additional EEG filtering
per trial; it does not assert identical preprocessing to the original paper.
Its Welch average is weighted by the number of within-trial segments. The null
analysis uses contiguous non-overlapping eight-second banks, five distinct
participants per draw, uniformly sampled session/window banks, and 5,000 draws.
The statistic is a channel mean followed by a participant mean; this is explicitly
a whole-montage adaptation and cannot establish the paper's four-region chance
equivalence. Ineligible conditions remain unavailable with an explicit reason.

Review qualification is computed from complete immutable artifacts, source
coverage, unchanged fold recipes and declared adaptations. It cannot be enabled
by a hard-coded flag. The >=15 dB output-SNR gate and matched VMD benefit use
the channel-first source-balanced paper tables; archived pooled-energy gates
remain additional project checks. The statistical tables do not themselves
prove biological preservation or authorize model development.

Dora/Biswal 2020 and Saini 2020 remain algorithm references. Their full evaluation
equations have not been verified in this correction. Do not attribute equations
to them or claim a replication until the primary methods are available.
