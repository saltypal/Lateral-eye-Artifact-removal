# Frontal support for posterior ICA and EOG-guided VMD

This is the active classical-method development plan, requested on 2026-10-08.
Deep learning is deferred until the classical target has passed its quality gates.
All numerical tests, decompositions, fitting and evaluation run on Kaggle.

## Processing contract

1. Keep the same reference, preprocessing and sample alignment across branches.
2. Frontal channels use independent RMS-normalized VMD. Keep the decomposition
   residual; it is not automatically ocular noise. Use the verified rolling
   engine only after checking its saved parity gate.
3. Each mode has separate signed HEOG and VEOG correlations. Compare a soft
   correlation gate against fuzzy C-means plus that gate. FCM uses the six
   existing signal descriptors and two absolute EOG correlations. Its scaler,
   centers and cluster EOG evidence are fitted on training data only.
4. Compare attenuating a selected whole mode against attenuating only its
   ridge projection onto the two EOG references. Neither correlation nor
   projection certifies that all removed activity is ocular.
5. Posterior ICA sees posterior channels plus raw frontal support electrodes.
   Support selection uses verified names, not artifact labels or held-out scores.
   Back-project the identified ocular sources and retain only posterior rows.
   Do not feed cleaned frontal channels, concatenate VMD estimates as extra
   electrodes, or subtract an artifact twice.
6. Central/temporal channels remain explicit. In the first regional diagnostic,
   nonfrontal channels outside the posterior dictionary pass through. Their
   treatment is a separate routing comparison, not a guessed anatomical claim.

## Modules and acceptance

- `reference_guided.py`: aligned correlation, train-only FCM, soft mode weights,
  ridge projection and frontal support routing. Tests cover zero references,
  sign invariance, mixed neural/ocular modes, duplicate electrode names, aligned
  routing and preservation of channels outside the correction target.
- `reference_guided_study.py`: matched Klados development comparison and a
  named OSF regional diagnostic. Saves raw tables, selected settings, cluster
  evidence, convergence failures and calibration provenance.
- Existing runner/controller/notebook: exact Git revision, clean runtime,
  full input hashes, fresh numerical tests and persistent Kaggle outputs.

## Bounded experiment

Klados uses the existing duplicate-clean grouped engineering split. Participant
and electrode-row identities remain unverified. Fit mode clustering on the
first scored window of every training recording/all 19 rows. Compare cluster
counts 2/3 with correlation-only selection. Reuse the previously selected
normalized K=3, alpha=2000 setting; this is not a new claim of global optimality.
Use all eight validation recordings, all 19 rows, and three nonoverlapping
1,024-sample windows starting at 2000/3024/4048. Search correlation threshold
0.2/0.4/0.6/0.8 and correction strength 0.25/0.5/0.75/1.0 for whole-mode and
EOG-projection correction. Clean counterfactuals receive the same available EOG
references; never refit the cleaner to the clean target. Choose minimum
record-macro RMSE only among configurations whose worst-record mean clean
change is <=1%, alpha/beta error <=0.5 dB and whose decompositions converge.
Do not examine held-out targets to choose settings. This phase does not score
the already inspected held-out set again.

OSF initially uses the first alphabetically listed original session of each of
the four studies as a development diagnostic. Exclude its first five complete
trials for input-only ICA calibration. Filter each fit-copy trial independently
before pooling samples for static ICA fitting. Compare posterior-only ICA with
posterior plus up to four frontal support electrodes and full-montage ICA.
Use fixed correlation threshold 0.8 and strength 0.25 for this support ablation;
these are hypotheses, not OSF-tuned deployment settings. Evaluate identical
genuinely annotated intervals for rest, horizontal, vertical and blink, and
report separate regional proxies. Do not select a winner on these proxies.

## Deployment path and limitations

The EOG-guided classical experiment requires HEOG/VEOG at runtime. It is
distinct from an EEG-only student. A future student can consume EEG, masks and
region metadata and learn from paired targets and accepted teacher residuals.
Training must include channel subsets/permutations and separate blink/lateral
examples. Learning requires validated training or updates; deployment alone
does not improve a network. Measure device memory, latency and quality before
claiming the student is easier to deploy.

The first diagnostic cannot certify N-channel accuracy, complete removal or
SNR >10 dB. Full grouped multi-seed validation, all-session OSF isolation,
covariance preservation and streaming remain separate gates.

Reference: MNE's official ICA documentation describes EOG correlation scoring,
channel selection and component reconstruction:
https://mne.tools/stable/generated/mne.preprocessing.ICA.html
