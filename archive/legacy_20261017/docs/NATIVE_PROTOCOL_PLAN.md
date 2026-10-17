# Native paper evaluation implementation

This Phase A extension preserves the existing source split and search outputs.
It closes the gap between short cache-window proxies and the condition-level
evaluation in the Kobler author demo. No reserved source is opened.

1. Validate streaming condition-level Pearson/RMSE against actual concatenation.
2. Validate multi-method overlap-add, with EEG and references aligned and every
   original sample covered. Filter each trial before any concatenation.
3. Read only development OSF recordings named in the completed corpus ledger.
   Reuse its calibration arrays, scoring trial IDs and held-out-fold recipes.
4. Correct complete trials using identity, direct regression, regional VMD–MWF
   and regional VMD–ICA. Save trial identities, corrections and failures.
5. Evaluate concatenated-condition Pearson and rest RMSE per channel. Prefer
   the publisher's exact HEOG_lpf/VEOG_lpf channels when present; disclose raw
   HEOG/VEOG fallback instead of inventing the missing derivative.
6. Compute Welch spectra separately within each rest trial. Combine spectra
   by the number of eligible two-second segments, never across trial joins.
7. Construct contiguous eight-second rest/ocular banks. Bootstrap five distinct
   verified participants for 5,000 repetitions and retain all random draws.
   Whole-montage averaging over channels then participants and non-overlapping
   banks are declared adaptations; this is not an exact four-region EEGOAR
   replication or an equivalence test.
8. Run a focused Kaggle fixture and two-record original-source pilot. Launch
   the full development job only after these checks pass.

Primary source: https://github.com/rkobler/eyeartifactcorrection/blob/master/demo_main.m
lines 121–145. EEGOAR section 2.4.1 supplies the eight-second chance analysis,
Welch durations and paired-permutation design; unresolved choices remain in
docs/EVALUATION_PROTOCOL.md. Numerical execution and plotting remain Kaggle-only.
