# SNR-target neural campaign: completed 2026-10-09

The conservative shared-support model reached **17.9272 dB mean SNR** on
eight reused test recordings/all 19 rows/three windows per recording. It
made **zero correction to every scored paired-clean input**. This meets
the 15-dB development target and the recorded preservation guard. The
benchmark is development-exposed, not fresh confirmation; one recording
remains at 14.2829 dB. Runtime VMD, HEOG and VEOG are required.

All numerical tests, VMD, training, evaluation and charts ran on Kaggle CLI
jobs. Source: `583b6942b1aaccaccc54f290750e98b7cdf7233c` for `snr-target`;
the shared-support refinement uses
`2ec0b3cd3da636cc3aed3b476ff890a6f56e2627` for `snr-context`, followed by
`0b183e6c09f6ecfbb3ed4818028eb5abbdf842fc` for `snr-guard`.
See [SNR_Target_Protocol.md](SNR_Target_Protocol.md) for the fixed split,
explicit inputs, loss, parameter sweep, selection and preservation gates.

## Completed first sweep

Four identity-weight/rate configurations per model, seed42, 160 epochs each.
All 37 numerical tests passed in 8.30 seconds. Every original source hash
passed; the restored OSF supplement was verified. Outputs/checkpoints/caches
and both rendered comparison figures were retrieved and visually inspected.

Mean SNR averages three windows within each of the eight reused test records,
then averages recordings in dB. It is not a pooled-energy SNR or a new unseen-
subject confirmation. The same all-19-row windows are scored for all methods.

| Model / frozen selection | Mean test SNR dB | Mean clean change | Worst-record mean clean change |
|---|---:|---:|---:|
| Raw | 2.9021 | Not separately scored | Not separately scored |
| Frozen reference-gated VMD | 9.4057 | Not scored in this baseline table | Not scored in this baseline table |
| EEG + VMD, validation preservation selection | 4.5194 | 0.6408% | 3.0956% |
| EEG + VMD, unconstrained validation selection | 12.0296 | 12.3198% | 29.6968% |
| EOG + VMD gain network, validation preservation selection | 5.2096 | 0.6306% | 0.9217% |
| EOG + VMD gain network, unconstrained validation selection | 17.8832 | 6.7942% | 9.9971% |

The first 17.88-dB result is within the requested range but does not pass the
clean-preservation requirement. It is an explicitly retained diagnostic,
not the selected deployment model. EEG-only conditioning did not meet the
15-dB target; its validation-safe checkpoint also failed preservation on the
reused test set. The old 10.212-dB VMD number was a validation result; its
matched reused-test score here is 9.406 dB.

The two networks contain 10,596 and 2,402 trainable parameters respectively,
independent of electrode count. Both require their input's VMD modes; the
second additionally requires aligned runtime HEOG and VEOG. A fixed number
of learned weights does not establish accuracy for untested cap sizes or
wearable latency. Four-study frozen OSF proxies ran with no diagnostic
exclusions; OSF supplies no clean target and cannot supply an honest SNR.

## Shared-support refinement

Validation behavior motivated sharing ocular support evidence across channels
before reviewing the first campaign's final test table. The refinement
prefers verified frontal support, with an all-observed-channel fallback for
unlabelled Klados caps. It controls intervention for the whole cap but keeps
separate signed HEOG/VEOG correction gains per channel. The same four
configurations, losses, source split and preservation limits are retained.
Source cache signals and keys are verified before reuse. Its 39 numerical
tests passed in 6.77 seconds on Kaggle. Its validation-preservation-selected
checkpoint scored 18.0249 dB mean SNR, with 0.1933% mean clean change and
1.3785% worst-record mean clean change on the reused test set. The worst
record exceeded the 1% limit; this is not the final accepted cleaner.

## Completed conservative deployment gate

The original epoch110/config2 neural weights are frozen. Twenty inference
rules were compared on validation under a tighter 0.25% worst-record clean-
change limit. The winning rule is a hard threshold at the model's learned
support-evidence threshold, offset zero, correction strength one. It leaves
below-threshold windows unchanged instead of applying a small sigmoid tail.
This is a fixed input-only rule; no clean target enters inference. Validation
SNR is 18.4312 dB with zero modification of scored clean inputs.

| Matched reused-test metric | Result |
|---|---:|
| Raw / frozen reference VMD / guarded neural SNR | 2.9021 / 9.4057 / 17.9272 dB |
| Guarded neural RMSE | 1.5076, unverified original units |
| Mean Pearson correlation with paired clean | 0.9900 |
| Mean / worst-record mean clean modification | 0% / 0% |
| Mean paired-clean alpha / beta error | 0 / 0 dB |
| Mean contaminated reconstruction covariance error | 0.0346 relative norm |
| Model parameters | 2,499 |

Individual recording SNR values are 17.3063, 20.2679, 14.2829, 20.0310,
20.6897, 16.4658, 18.3618 and 16.0123 dB for recording IDs
5/14/16/30/41/45/48/49 respectively. The target is met on the mean; it is
not a claim that every channel, window or recording reaches 15 dB.

All 40 numerical tests passed in 9.13 seconds. Original source hashes and
source-cache signal/key checks passed. The completed Kaggle kernel outputs,
checkpoint, predictions, per-window/per-record tables, inference controls,
environment and plots were retrieved; both plots were visually inspected.
Four-study frozen OSF proxies ran without exclusions. Those data still
cannot supply paired-clean SNR or prove complete blink/lateral removal.

The gate refinement was motivated by a reused-test preservation failure.
Although thresholds/strength were selected on validation, this makes the
benchmark development-exposed. A fresh locked set and grouped multi-seed
confirmation remain required for a generalization claim. EEG-only models
have not reached the 15-dB target under preservation limits.

Reproduce from the research checkout:

```powershell
python tools/kaggle_campaign.py submit --phase snr-guard --sha 0b183e6c09f6ecfbb3ed4818028eb5abbdf842fc
python tools/kaggle_campaign.py retrieve --phase snr-guard
```

The final selected checkpoint is
`results/snr-guard/results/snr-guard-0b183e6c09f6/eog_vmd_context_student/best_safe.pt`.
Restore the checkpoint's inference gate mode, offset and strength alongside
its neural weights. The saved checkpoint's protocol records source lineage.

## Original concerns and limits

K=3 and alpha=2000 reuse the previous normalized VMD development setting.
This neural sweep does not establish a global best K, a reason to choose
five modes, or five canonical EEG bins. A VMD vector is a full time-domain
mode from the verified solver; the neural model uses its waveform together
with correlation/energy descriptors. HEOG and VEOG remain separate reference
axes, not labels assigning each mode exclusively to blink or lateral motion.

The shared TCN and gain MLP do not require a transformer. Parameter count is
independent of N, while VMD work, feature construction and neural processing
grow with the observed EEG-channel count. Deployment still needs VMD and,
for the high-SNR comparison, actual eye-reference sensors. Distillation into
a VMD-free/EEG-only model remains a separate experiment. The neural result is
not literal posterior ICA, and regional ICA/frontal VMD results from the
previous study remain a separate classical comparison.

Full grouped/multi-seed validation, subject provenance, all-session OSF
isolation, matched recent baselines, streaming behavior and device performance
are still not established. No complete-removal claim follows from these
development scores.
