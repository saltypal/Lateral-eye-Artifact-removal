# 15–20 dB neural development target

User authorization: train/evaluate through Kaggle CLI; retain VMD preference,
cap-independent processing, frontal context, both ocular sources, and report
actual performance. Source modules remain in Git. All numeric tests, VMD,
training, evaluation and figures execute on Kaggle.

The original frozen duplicate-clean-group split remains 37/8/8 recordings.
All 19 rows and three nonoverlapping 1024-sample windows at 2000/3024/4048 are
used. Windows never cross records. These eight test recordings have all been
evaluated by previous campaign revisions; they are reused record-held-out
data, not a fresh or subject-independent confirmation set. No test score
selects model parameters, epochs, strength or a training configuration.

The paired signal-to-noise ratio is
`10 log10(sum(clean^2) / sum((prediction-clean)^2))` per window, followed by
averaging windows within recordings and recordings in dB. The training loss
combines negative SNR divided by 10 and softplus of the 20-dB shortfall divided
by 5. It is scale-sensitive, not SI-SNR or correlation. A target-relative
1e-8 numerical floor limits loss SNR to 80 dB; reporting uses the original
float64 paired metric, without an optimistic substitute.

Supporting losses: target-RMS-normalized MSE weight 0.1, derivative weight
0.05, trusted VMD teacher distillation weight 0.02 and an identity barrier
`mean(log1p(clean_correction_energy / clean_energy / 0.005^2))`. Teacher
acceptance checks paired training improvement, <=1% clean change and mean
alpha/beta errors <=0.5 dB. No validation/test truth enters teacher fitting.
The teacher uses the frozen accepted reference-safe settings and actual
input VMD vectors, K=3, alpha=2000, tolerance=1e-6, budget=2000. Failed
decompositions receive zero correction, with every EEG row still scored.

Two explicit deployment contracts are compared:

1. `eeg_vmd_student`: EEG plus its own VMD vectors at inference. Shared TCN
   weights, permutation-invariant masked context and full-rate mode waves.
   This requires VMD computation at deployment; it is not the earlier
   VMD-free distilled student. The model supports a frontal encoded-feature
   pool, but Klados region identity is unknown, so that pool is not trained
   or claimed validated. OSF uses the trained shared/unknown setting.
2. `eog_vmd_gain_student`: EEG plus VMD plus measured HEOG/VEOG at inference.
   A shared neural MLP learns separate signed two-reference projection gains
   using VMD correlation/energy features. It cannot be called EEG-only.

For each contract, identity weights {0.02,0.2} cross learning rates
{0.001,0.0003}; seed 42, 160 epochs, AdamW weight decay 1e-4, cosine decay
to 1e-5, batch eight, gradient clip five, and 15% random training-channel
dropout. At least one electrode remains observed. Strengths
{0,0.25,0.5,0.75,1} are compared at epoch one and every ten epochs.
Maximum validation record-macro SNR selects a preservation-safe checkpoint
under <=1% worst-record window-mean clean modification and <=0.5 dB mean
alpha/beta errors. A separate unconstrained validation winner is retained
as a diagnostic and must never replace the safe deployment claim.

No held-out inference or target-dependent calculation runs until every
configuration in both contracts has finished. Frozen raw and reference-VMD
baselines use identical scoring windows. Final outputs include all window/
record metrics, safe and unconstrained checkpoints, histories, numerical
diagnostics, cache keys, original/source hashes, environment/hardware/Git SHA,
and four-study frozen OSF proxies. OSF has no clean target; no OSF SNR is
claimed. GPU throughput is not a certification for a wearable processor.

The model components are original adaptations, not a replication of a paper.
The paired-supervision design is informed by
[EEGdenoiseNet](https://arxiv.org/abs/2009.11662), which supplies known clean
targets for controlled denoising comparisons. Our actual data remain Klados
and OSF. This development sweep cannot establish full five-fold/three-seed
success, unseen-participant performance, cap-transfer accuracy or complete
ocular removal. The target is a measurement criterion, not a promise.
