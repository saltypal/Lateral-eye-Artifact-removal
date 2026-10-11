# Frozen guarded VMD result reproduced

Kaggle run `guard17-cache-replay-003` completed on 11 October 2026.
Legacy source `0b183e6c09f6ecfbb3ed4818028eb5abbdf842fc`;
replay source `6286aa841afc0cb6664cbb2ba92e3740995c47ff`.
Kernel `satyapaladugu/guarded-vmd-frozen-archive-replay-fix/1`.

| Method | Mean output SNR dB | Pearson CC | RMSE | Mean clean modification |
|---|---:|---:|---:|---:|
| Original archived input | 2.902146 | 0.860895 | 8.060284 | 0% |
| Frozen guarded VMD hybrid | 17.927210 | 0.989990 | 1.507619 | 0% |

The historical score was 17.927211 dB. The replay reproduces it to approximately
0.0000015 dB with the exact source checkpoint, K3/2000, hard gate, zero offset
and correction strength one. No training or parameter selection occurred.
Mean output SNR is window energy SNR over all 19 channel rows, averaged within
eight records and then across records in dB. It is not SNR improvement or the
current channel-first frontal aggregation. The 24 prediction windows, identity
outputs and frozen weights are downloaded locally with verified artifact hashes.

Runtime HEOG/VEOG and VMD-mode features remain required. Electrode and
participant identities are unknown; this is an archived development benchmark,
not a fresh frontal/posterior or EEG-only qualification result. Paired covariance
error was .034605; that describes contaminated-input reconstruction, not the
separate clean-input covariance gate. Report both without conflating them.

## Source/cache mismatch investigation

The prior original-recording replay failed its strict `array_equal` check.
The diagnostic kept every difference visible and never edited cached signals.
Across train/validation/test comparisons, only a few samples differ: maximum
relative RMS difference 1.195e-10, with channel/window correlations effectively
one. In test raw EEG, only three samples differed and relative RMS difference
was 3.062e-15. These magnitudes are consistent with floating-point roundoff;
the exact environment/implementation cause is not established.

The reproduced score uses the explicitly declared immutable archived cache
with SHA256 `8202b73390d31f587945b9a1891df95fa15f2b07c42744cbbbf4d64a1b9f7db4`.
It does not assert bitwise equality with the newly preprocessed original arrays.
Checkpoint SHA256 is
`4bb7c1b0cfa0e8341c4053ab3099efb16832f045f99e37afde1904a93c2cd0a9`.

## Current frontal transfer

Run `frontal-frozen-transfer-001`, source `2d3d5f4`, is a separate experiment:
apply the same frozen gain mechanism only to verified frontal K3/2000 modes
on the 320 development examples. Keep all nonfrontal EEG unchanged. Disable
the anatomy feature never trained in the historical model. Compare source-
excluded selected direct regression and identity using the current paper metrics
and engineering gates. No new weights are trained and no 17 dB claim transfers
automatically. Native validation and posterior integration remain pending.
