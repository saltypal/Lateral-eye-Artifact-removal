# Six-hour execution checklist — 17 October 2026

Objective: a small, understandable Klados/OSF-only experiment with saved real
results. The 17.9 dB historical number is reproduced honestly, never guaranteed
for another channel subset or dataset. No LEMON records or generated mixtures
enter the replacement campaign.

| Order | Task | Output | Acceptance |
|---|---|---|---|
| 1 | Preserve legacy implementation and AutoVMD | exact-SHA source archive, inventory, hashes | archive verified; no old result removed |
| 2 | Verify source identities and anatomy | Klados publisher audit, OSF named groups, checksums | unknowns remain unknown; missing inputs fail visibly |
| 3 | Numerical contracts on Kaggle | saved test report | reconstruction, masks, lag/axes, calibration separation pass |
| 4 | Matched Klados paired comparison | per-row/window/record scores and predictions | identical input/target preprocessing and scoring |
| 5 | Fixed frontal VMD on OSF | modes and HEOG/VEOG evidence for every mode | no posterior/central VMD; failed decompositions counted |
| 6 | Nonfrontal ICA/regression/MWF | regional and full-output results | original-input artifact estimates, one subtraction |
| 7 | Historical frozen replay | checkpoint/caches/hashes and score | old data exposure and unknown anatomy declared |
| 8 | Figures and final review | notebook outputs, report, recovery receipt | labels/units/scope checked; no native reconstruction SNR |

Initial bounded pilot: all 53 Klados records, historical eight-record comparison
reported separately, and four OSF development sessions. Expand OSF only after
the loading, calibration and scoring contracts pass. Existing confirmation
participants stay excluded; the new campaign does not manufacture a fresh holdout
from previously exposed Klados records.

Methods: identity, direct EOG ridge, fixed VMD reference correction, ICA component
regression and GEVD–MWF. Regional hybrids combine the same frontal VMD estimate
with each nonfrontal estimate, plus a frontal-pass-through comparison. Central
means verified FC/C/CP/temporal electrodes under the declared engineering grouping;
it is not a hidden third algorithm or an anatomical discovery.

No broad K/alpha search, guessed Klados montage, VMD–SOBI, anchored-gain variants,
or large neural training campaign is needed to make this first comparison work.
The archived model can be replayed without retraining. Any later supervised
model has its own explicit source split and result label.

## Gates

- Paired metrics: MSE, RMSE, temporal RRMSE, ordinary Pearson CC, output SNR and
  input-to-output SNR improvement, with explicit aggregation.
- Native metrics: unchanged-reference signed/absolute HEOG and VEOG correlation,
  separate lateral/vertical/blink/rest, resting RMSE, relative modification,
  alpha/beta dB error and covariance change. Rest is a preservation proxy.
- Existing engineering thresholds are reported separately: 1% worst-record mean
  clean modification, 0.5 dB alpha/beta error, 0.02 relative covariance change and
  0.005 paired CC deterioration against the declared comparator. Passing software
  tests does not pass these scientific gates.
- Save failures and partial progress per recording. A failed candidate cannot be
  removed to improve its mean. Raw input is its explicit fallback.

## Timing budget

First hour: source/architecture cleanup and focused Kaggle readiness. Next two:
paired and native pilot execution, diagnosis and repair. Next two: complete
matched comparisons and recover artifacts. Last hour: inspect figures, write
measured conclusions, identify unfinished work. Times are budgets, not claims
that hardware or numerical failures cannot change them.
