# Simple execution plan

## First: frontal comparison

Compare these methods through the same interface: EEG in, cleaned EEG and
artifact estimate out. No new invented method names or broad parameter search.

| Order | Method | What we implement | Current status |
|---|---|---|---|
| 1 | Historical VMD model | Frozen EOG-assisted 2499-parameter checkpoint, K3/alpha2000; seven frontal rows only | All-19-row 17.927210 dB replay verified; frontal result must be measured separately |
| 2 | SVMD + FCM | PySDKit successive solver, EEG-demo parameters, declared moving-window FCM adaptation | Implemented; Kaggle numerical check pending |
| 3 | VME + GMETV | PySDKit VME, original GMETV Algorithm 3 with explicit diagonal C | Implemented adaptation; unavailable ocular settings remain a limitation |
| 4 | EEGOAR-Net | Author architecture/pretrained initialization; explicit seven+padding montage; Klados training/validation only | Frozen 64-label mapping lacks FP1/FP2; custom-montage fine-tuning adaptation under Kaggle validation |
| 5 | GEVD–MWF | Rank-two spatial filter; covariance masks from training EOG only | Existing tested core; matched frontal evaluation pending |
| 6 | BandRouteNet v2 | Pinned author BRAANetV2, paired training, validation-only checkpoint selection | Training adapter implemented; Kaggle GPU smoke then 15 epochs |

Identity and direct HEOG/VEOG regression remain the two simple controls.
Reproductions and adaptations are labelled separately. Missing paper details
are never silently replaced with guessed parameters. Both unavailable full PDFs
were requested; the user does not have them. Continue available implementations
and record exact remaining reproduction gaps.

## Required tests and output

- [x] Preserve old code, AutoVMD, caches and historical results.
- [x] Move legacy modules/notebooks/configurations under `archive/`.
- [x] Verify dataset checksums and preserve OSF confirmation identities.
- [x] Run meaningful numerical contracts on Kaggle: 20 passed in pilot v2.
- [x] Reproduce the historical 17.9272 dB checkpoint on its exact archived cache.
- [x] Run one native OSF session and save every frontal mode's H/V correlation.
- [x] Finish six method adapters and Kaggle numerical contracts (29 passed).
- [x] Complete one-window/one-epoch GPU integration run for all six, no method exceptions.
- [x] Run the six frontal methods on identical archived Klados records/windows.
- [x] Download full Klados executed notebook: 6/6 code cells, 13 figures, 5 tables, 0 errors.
- [ ] Independently recompute paired-clean CC and embed labels in NPZ copies.
- [ ] Complete frontal-only OSF before/after HEOG/VEOG and resting-preservation evaluation.
- [ ] Save per-record/channel SNR, improvement, MSE/RMSE, RRMSE and Pearson CC.
- [ ] Save lateral/blink correlation, resting preservation, runtime and failures.
- [ ] Inspect waveform, mode, correlation, PSD, preservation and metric plots.
- [ ] Publish one measured comparison and select a frontal method only from evidence.

No native OSF reconstruction SNR is reported because no clean target exists.
Use Klados paper-listed order as explicitly requested: frontal rows
0/1/2/3/10/11/16 = FP1/FP2/F3/F4/F7/F8/Fz. Save the assumption because row-to-list
binding has not been independently verified. The historical score is development-
exposed and requires runtime VMD plus HEOG/VEOG. It is not VMD alone.

## Then: posterior

Start with posterior pass-through, GEVD-MWF and ICA component regression. Use
named OSF electrodes and whole calibration trials; keep one simple implementation
of each, without five different montage-support variants.

After the frontal winner is established, test one additional transfer route:
use its estimated frontal artifact to form a common blink signal and a signed
right-minus-left lateral signal. Fit their posterior transfer on calibration
only. Compare with the same posterior method without frontal context. A strong
frontal artifact does not automatically authorize strong posterior subtraction.

## Then: central

Use pass-through as the default. Compare conservative calibrated EOG regression
and the same GEVD-MWF implementation. Keep correction only if it improves ocular
suppression while preserving resting EEG and neural spectra/covariance. Central
means verified FC/C/CP/temporal electrodes under our declared grouping; unknown
rows remain unknown. There is no extra algorithm called shared.

## Final regional combination

Choose one frontal, one posterior and one central rule. Their masks are disjoint.
Each estimates artifact from original input; subtract the combined estimate once.
Do not add regional dB values. Report original paper metrics and the existing
engineering preservation gates separately. Failed targets remain failures.
