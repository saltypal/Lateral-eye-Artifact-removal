# Simple execution plan

## First: frontal comparison

Compare these methods through the same interface: EEG in, cleaned EEG and
artifact estimate out. No new invented method names or broad parameter search.

| Order | Method | What we implement | Current status |
|---|---|---|---|
| 1 | Our fixed VMD | K=3, alpha=2000; inspect every mode against HEOG and VEOG; retain residual | Kaggle pilot passed |
| 2 | Historical VMD model | Frozen EOG-assisted 2499-parameter checkpoint; separate replay and frontal adaptation | 17.927210 dB replay verified; native adaptation failed resting preservation |
| 3 | SVMD + FCM | True successive decomposition, paper features and moving-window fuzzy clustering | Full ocular-paper settings unavailable; author solver and clearly labelled adaptation under audit |
| 4 | VME + GMETV | Extract ocular-dominated mode, recover EEG inside it, add EEG back | Full settings unavailable; verify original extraction and GMETV equations before implementation |
| 5 | EEGOAR-Net | Pinned author architecture and pretrained weights; verified frontal channel mask | Author repository located; adapter next |
| 6 | BandRouteNet v2 | Pinned published architecture/author code; paired Klados training and OSF frontal inference | Author repository located; code/split audit next |

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
- [ ] Finish paper adapters and method-specific tests.
- [ ] Run all eligible frontal methods on identical OSF scoring trials.
- [ ] Run eligible paired methods on identical Klados records/windows.
- [ ] Save per-record/channel SNR, improvement, MSE/RMSE, RRMSE and Pearson CC.
- [ ] Save lateral/blink correlation, resting preservation, runtime and failures.
- [ ] Inspect waveform, mode, correlation, PSD, preservation and metric plots.
- [ ] Publish one measured comparison and select a frontal method only from evidence.

No native OSF reconstruction SNR is reported because no clean target exists.
No Klados electrode anatomy is guessed. The historical score is development-
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
