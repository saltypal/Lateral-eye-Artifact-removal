# What is established, adapted and unavailable

Verified 17 October 2026. This file governs claims in the replacement notebook.

## Klados

The [original article](https://doi.org/10.1016/j.dib.2016.06.032) describes a
semi-simulated benchmark: original reference EEG plus weighted horizontal and
vertical EOG. The [publisher version 4](https://data.mendeley.com/datasets/wb6yvr725d/4)
contains four MATLAB files. Our earlier hash-verified source audit exactly
matched 53 local exports to publisher records after float32 conversion and the
start crop `[0:5401]`, without row reordering or amplitude scaling. Recording
45 is missing from the exports; its omission remains unexplained.

The article lists 19 electrodes but does not explicitly establish the row order
of these MATLAB matrices. The source files contain no channel-name or participant
fields. The list cannot be silently used as a verified row map. Participant
independence cannot be claimed from record IDs. An earlier coefficient audit
found no stable participant pairing from the published contamination weights.
The exact-match mapping and publisher checksums are copied into `provenance/`.

Therefore the new paired Klados methods use unknown channel rows. No new VMD
decomposition or regional mask is invented for those rows. The requested historical
VMD replay retains its original all-row scope and is shown separately. If a
verified row map becomes available, a new frontal experiment can use it.

## Historical 17.9272 dB

This was an `EOGContextGainStudent`, a 2,499-parameter MLP with input-window VMD
features, explicit HEOG/VEOG projections and a support gate. It was neither VMD
alone nor an EEG-only CNN/GRU. The checkpoint and archived cache are hashed.
The replay uses fixed weights and hard gating with zero offset and strength 1.
The exposed eight-record result uses pooled-energy SNR per window, averaged
within record, then across records in dB. A mean of individual channel SNRs is a
different number. Earlier preprocessing/cache-source discrepancies remain
historical limitations; the replay does not assert equality to newly filtered
original signals. Restricting a checkpoint to named OSF frontal inputs would be
an out-of-dataset adaptation requiring its own validation, not the same result.

## VMD and successive VMD

[Original VMD](https://doi.org/10.1109/TSP.2013.2288675) decomposes a signal into
K adaptive narrowband vectors with optimized center frequencies. Alpha penalizes
bandwidth; it is not a cutoff frequency. K=3, alpha=2000 is the user's fixed
historical starting recipe, not a proven optimum or five canonical EEG bands.
Our reviewed rolling solver follows vmdpy 0.2 with RMS normalization, uniform
initialization, tolerance 1e-6 and a 2000-iteration cap. Its source and MIT
attribution are preserved. Residuals are retained; only estimated artifact is
subtracted. Capped/invalid decompositions pass through and remain counted.

[SVMD](https://doi.org/10.1016/j.sigpro.2020.107610) extracts modes successively
without supplying K. It is a different algorithm. The later ocular
[SVMD–FCM article](https://doi.org/10.1016/j.bspc.2025.109434) reports a blink
application, but its full parameter specification was not available through the
publisher page in this audit. No guessed settings or replication claim are used.
Its lateral performance is not established by the accessible abstract. It is
deferred while the fixed, inspectable implementation is validated.

## ICA regression and MWF

[Mannan et al. 2016](https://doi.org/10.3389/fnhum.2016.00193) combines component
identification with regression. Our bounded challenger uses FastICA, EOG
correlation selection and ridge projection within components. It is an adaptation,
not reproduction of the article's entropy/RLS procedure. Whole-component rejection
is a separate preservation-risk control. Both fit only on calibration data.

[Somers et al. 2018](https://doi.org/10.1088/1741-2552/aaac92) uses artifact and
nonartifact segments to estimate multichannel Wiener filters. The
[author toolbox](https://github.com/exporl/mwf-artifact-removal) permits supplied
artifact masks and optional temporal embedding. Our adaptation is instantaneous,
rank-two GEVD with covariance shrinkage and correction strength 0.5. OSF supplies
calibration conditions; Klados uses explicitly approximate high/low EOG-activity
masks within its prefix. Low EOG activity is not certified clean EEG.

GEVD–MWF is a classical covariance filter. It is unrelated to the neural model
name EEGOAR-Net. Claiming the same implementation would be incorrect.

## EEGOAR-Net and native OSF

[EEGOAR-Net author code](https://github.com/dmarcos97/EEGOAR-Net) includes a
montage-masked neural model and pretrained weights. Its example uses 128-sample
epochs and a declared 64-channel label set. Applying it correctly requires
verified channel names, sampling/scale handling and an exposure audit. It is
not silently substituted for ICA or MWF. An adapter is a separate follow-up,
not a reason to hold up the first classical comparison. ASR likewise requires
its own validated calibration and implementation; it is not implemented by
renaming a covariance threshold.

The OSF original readme identifies trial conditions, signed sample annotations,
block numbers and named EOG derivatives. Source records are independently
filtered by trial; block 1 calibration and block 2 scoring remain separate.
HEOG_lpf/VEOG_lpf evaluation references are kept unchanged except resampling.
Correlation is computed for every EEG channel against both axes. Resting RMSE,
spectral change and covariance change are preservation proxies. Native OSF has
no paired artifact-free target; reconstruction SNR and clean-target CC are not
available. Neither zero correlation nor fitting A/B weights proves pure ocular
removal. These limits also explain why the removed LEMON generator could not
establish that its estimated mixing fields were ground-truth ocular signals.
