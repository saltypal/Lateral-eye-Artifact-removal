# Frontal support and EOG-guided VMD: development results

The classical extension is implemented and has passed its bounded development
gate. It has not established held-out deployment quality or complete removal.
Numerical execution was exclusively on Kaggle.

## Processing path

Raw frontal EEG supplies independent VMD cleaning and support observations to
posterior ICA. Posterior outputs receive the ICA correction; frontal outputs
receive the VMD correction. Both branches operate against the original aligned
EEG. Central/temporal/unknown channels pass through in this diagnostic.

The selected VMD branch uses separate signed correlations with HEOG and VEOG.
The absolute maximum correlation controls a soft gate above 0.6. It subtracts
the gated ridge projection of each selected mode onto the two EOG references,
keeping the remainder and the VMD residual. References are required at runtime.
The ridge penalty is 0.01, K=3, alpha=2000, normalized tolerance=1e-6 and the
maximum numerical budget is 2000 iterations. Correction strength is one;
individual mode weights remain between zero and one.

A channel whose decomposition reaches that budget contributes no artifact
estimate and passes through unchanged. It does not enter clustering or
correlation selection. Training decompositions used for clustering must fully
converge. Failed attempts remain in the diagnostic ledger and every original
EEG row remains in evaluation.

## Paired Klados development evidence

Source run: `e046fc1aa07eca95390aeeaa0b07c4d519147e82`,
[reference-safe Kaggle kernel](https://www.kaggle.com/code/satyapaladugu/region-aware-eog-reference-safe).

The sweep fitted selectors on 37 training recordings/all 19 rows, first window
only, and evaluated 96 settings on eight validation recordings/all 19 rows,
three nonoverlapping windows starting at 2000/3024/4048. It used no held-out
target for selection or scoring. Two/three FCM clusters plus correlation were
compared with correlation-only selection and both whole-mode and projected
mode correction. Correlation-only projected correction was selected.

| Metric | Raw / selected result |
|---|---:|
| Record-macro validation SNR | 3.2455 / 10.2121 dB |
| Record-macro validation RMSE | 7.9613 / 3.5737, unverified source units |
| Mean per-record RMSE reduction | 52.7519% |
| Mean clean-input modification | 0.0633% |
| Worst-record mean clean-input modification | 0.5063% |
| Accepted correction convergence | All corrected channels converged |

SNR is computed against paired clean EEG per window, averaged within each
recording and then across recordings. It is not a pooled-energy SNR. The clean
guard applies the same frozen cleaner to paired clean EEG with the same EOG
references; its worst-record window-mean limit is 1%, and mean alpha/beta
spectral-error limits are 0.5 dB. Those development limits passed. They are not
the full research-contract noninferiority test.

Two of 912 validation channel decompositions remained capped: one dirty and
one clean input. They passed through unchanged. All 703 training channel
decompositions converged. The earlier 500-iteration and 2000-iteration runs
without this fallback accepted no candidate and remain archived.

Klados row and participant identities remain unverified. These paired figures
evaluate the mode cleaner over all rows, not an anatomically verified regional
hybrid or a subject-independent deployment test.

## Named OSF regional diagnostic

One session from each of the four studies was evaluated on identical annotated
rest, horizontal, vertical and blink intervals. Five whole independent trials
were excluded for calibration; fit-copy filtering preceded static sample
pooling. All twelve posterior-only, frontal-supported and full-montage ICA
calibrations converged. The frontal-supported fit used up to four frontal
electrodes. ICA correlation threshold 0.8 and strength 0.25 were fixed before
OSF evaluation. No OSF result selected these parameters.

| Combined regional method: mean correlation reduction | Frontal | Posterior |
|---|---:|---:|
| Blink interval, VEOG | 0.0865 | 0.0957 |
| Horizontal-eye interval, HEOG | 0.1123 | 0.0238 |

Posterior-only ICA made no correction at the tested threshold. Frontal-supported
ICA made a measurable correction; this does not guarantee ocular separation.
Mean rest-window modification relative to raw was 1.031% frontal and 1.742%
posterior. OSF has no paired clean target, so these are suppression/preservation
proxies, and no OSF SNR is claimed. No sessions/trials/channels were excluded
by this four-session correction diagnostic.

## Verification and remaining work

The final source run passed 33 numerical tests in 9.89 seconds and original /
restored source integrity checks. Tests cover reference sign/unit invariance,
mixed-mode neural retention, train-only selector reload, name-based routing,
independent VMD parity and explicit failed-channel identity passthrough.
Saved outputs include validation tables, cluster evidence, VMD diagnostics,
ICA calibration/routing, regional proxies, runtime records and environment/SHA.
The later removal of averaged record/start identifiers from summary metadata
does not change fitting, selection or numerical metrics.

Before deployment: evaluate frozen settings with full grouped/multi-seed
validation, all-session OSF isolation, unseen-cap channel subsets, covariance
preservation and realistic runtime. Train an EEG-only student using paired
truth and accepted teacher residuals; retain signed left/right frontal context.
Validate that student's own preservation and device performance. Neural
training/deployment is not completed by this classical study.
