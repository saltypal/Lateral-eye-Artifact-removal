# Frontal-only development evidence

Snapshot: 11 October 2026. This is a development report, not final qualification.
Read the append-only orchestration events for newer job states.

## What is implemented and validated

- Verified F/AF/FP electrode masking; posterior/shared/unknown channels stay original.
- Two independent fixed decomposition arms: K3/alpha2000 and K4/alpha2000.
- Joint HEOG/VEOG regression, selective/interval/whole-mode correction, projected
  SOBI, and explicitly labelled 2022 selector and fixed 2024 SVM/ApEn adaptations.
- Recipient/donor exclusion, same-unit unscored calibration scaling, reconstruction
  residual retention, numerical fallback and immutable cache verification.
- Exact Git/versioned-parent notebook launches and recoverable per-example outputs.
- Kaggle fixture 005: 108 passed, 14 warnings, zero failures at `08f33ac`.
- Kaggle waveform inspection: 109 passed, 14 warnings, zero failures at `c40721a`;
  52 selected-arm metric-vector parity checks passed on four predetermined examples.

AutoVMD cancellation was verified and its historical artifacts retained. The
new correction bank is not a K/alpha selector. Author-code parity remains
unestablished; the 2020 full specification remains unavailable. See the paper audit.

## Completed correction screen

Run `frontal-screen-001`, source `638d496c244bdf8a2ae02ce3b4415949e22e029e`,
Kaggle `satyapaladugu/vmd-frontal-correction-screen/1`:

- 708 correction recipes; 20 examples; five recipients/five donors; 13 verified
  frontal channel names; clean/blink/lateral/mixed; dirty examples at nominal 0 dB.
- Recipes selected outside the heldout recipient/donor bucket, separately per
  fixed K arm. SVM uses one predeclared recipe without a crossfit-derived fallback.
- Zero failure rows. Numerical-stage runtime 499.97 s; runner wall time 694.81 s.
- Peak resident memory 649,184 KiB. Cache creation cost is excluded and cannot
  be presented as runtime-VMD deployment cost.

| Selected fixed arm | Mean output SNR dB | Mean SNR improvement dB | Pearson CC | CC deterioration vs direct |
|---|---:|---:|---:|---:|
| Direct HEOG/VEOG regression | 12.7468 | 17.8016 | 0.97217 | 0 |
| VMD projected K3 | 12.1386 | 17.1933 | 0.95825 | 0.01392 |
| VMD projected K4 | 12.3030 | 17.3577 | 0.95890 | 0.01327 |
| SOBI projected K3 | 12.1348 | 17.1895 | 0.95819 | 0.01397 |
| SOBI projected K4 | 12.2936 | 17.3483 | 0.95863 | 0.01354 |
| Identity | -5.0547 | 0 | 0.49977 | 0.47240 |

Nominal input SNR uses the whole montage. Actual frontal input SNR is -5.0547
dB here. A 17.8 dB **improvement** is not a 17.8 dB **output SNR**.

Both projected VMD arms remain below 15 dB and exceed the .005 CC deterioration
limit. No qualified teacher, superior VMD claim or integration approval follows.
Whole-component rejection and the entropy adaptations performed substantially
worse; their complete results are preserved rather than excluded.

Approximate recipient/donor Poisson-multiplier 95% intervals for the output-SNR
means: direct [11.9759, 13.2619], VMD K3 [10.5479, 14.0168], VMD K4 [10.6150,
14.1107]. Matched VMD-minus-direct intervals cross zero. These are explicitly
project statistical approximations on five sources, not paper-native tests or
confirmation evidence. Primary paper metrics remain MSE, temporal RRMSE and
ordinary Pearson correlation.

The clean-input plot shows zero mean modification for the selected arms. Native
resting EEG and full-source clean preservation remain necessary; this plot alone
does not establish retention of all neural activity during ocular contamination.

## Active and pending work

- Frozen full evaluation: all 320 development examples at -5/0/5 dB plus clean
  controls, all eligible recipients, both fixed VMD arms. Run `frontal-full-001`
  launched at `1d18d81a6b93bafef085c42258d4e33dd58ffae6`; RUNNING at this snapshot.
- Mechanism inspection: first sorted example per condition, mode vectors/centers,
  reconstruction residuals, actual waveform/score parity, interval-energy coverage,
  lag mismatch and calibration-anchored mean-offset diagnostics. Run
  `frontal-evidence-001` launched at `c40721a6c2c459a025834a29361fcc56731e4f16`;
  COMPLETE and retrieved. Targets are diagnostic-only and never enter inference.
  The screen saved tables rather than every grid prediction; NPZ prediction-parity
  checks are therefore zero, while all 52 metric-vector checks passed. Four mode
  vector/residual archives and blink/lateral/mixed/clean plots were also downloaded.
- Smaller repaired pilot `frontal-pilot-002` remains RUNNING; its logs have not
  yet matched the requested run. That is unavailable progress evidence, not a
  numerical success or confirmed exception. The screen supplies current recipes
  without waiting for this pilot.
- Native OSF pilot follows verified full evaluation. It uses the pinned restoration
  bundle, audits all development SET/FDT files, preserves trial boundaries and
  compares unchanged HEOG/VEOG association and rest RMSE/PSD/preservation.
- Posterior integration, EEG-only student training, external evaluation and export
  remain pending. The local queue stops at frontal review and cannot launch them.

## First mechanism evidence

These observations use the first sorted Fz example in each condition, not an
optimized or representative population estimate. The lateral example's 3-SD
interval support covered 6.39% of known ocular-artifact energy; blink support
covered 94.74%. This explains a plausible weakness of interval-only correction,
but support thresholds still require grouped preservation/accuracy validation.

For that lateral example, ungated instantaneous projection scored 7.7897 dB,
short-lag projection 10.6197 dB, and calibration-anchored short-lag projection
11.6112 dB. Mean-bias squared decreased from 9.6009e-14 to 2.3347e-15 source
units squared under anchoring. However, the clean example's ungated lag
projection scored only 12.1585 dB against unchanged input. Those diagnostic
corrections therefore cannot be adopted without the association/preservation
gates. Blink/mixed examples did not improve under extra lag terms. More lags are
not automatically better, and no target was used to fit any projection.

The source-normalized Fz plots show adaptive K3/K4 center frequencies and retained
residuals; they are not five physiological bands. This is mechanism evidence on
the regression-generated corpus, not independent native EEG or final SNR evidence.

The posterior report was inspected read-only. Its immutable future integration
parent is `satyapaladugu/vmd-band-posterior-full/1`, source `a25acb9`, with the same
corpus/2 and preprocessing. Its best-candidate search table is not a source-excluded
winner estimate and teacher qualification is false. It will not be silently treated
as a qualified ICA/MWF result or rerun as another broad posterior search.

## Recover and inspect

From the project root:

```powershell
python tools/kaggle_campaign.py status --run-id frontal-full-001
python tools/kaggle_campaign.py retrieve --run-id frontal-screen-001 --file-pattern '(summary|frontal_oof|frontal_search|frontal_uncertainty|frontal_comparison|tests)'
```

Only one copy of `python tools/frontal_queue.py` should run. Its append-only
events and hidden-process logs live in
`results/vmd-frontal-replication-20261011/orchestration/`. Failed kernels or API
errors stop dependent submissions for diagnosis. Every numerical stage remains
on Kaggle and records an exact published source SHA and parent versions.

## Failure and repair checkpoint â€” 11 October 2026, 09:01 UTC

Full evaluation frontal-full-001 failed after 291/320 examples at the strict
archived mode-plus-residual reconstruction check. Saved progress and individual
example outputs remain evidence; this run is not complete and cannot qualify a
method or supply the native parent. The dependent queue stopped as designed.

Focused Kaggle notebook frontal-cache-audit-001 (source efda250, kernel
satyapaladugu/vmd-frontal-cache-audit/1) inspects both fixed archives over all
320 development examples without any VMD fitting or cache modification. It
compares float32 and float64 summation at the unchanged tolerance. The potential
float32 accumulation explanation remains a hypothesis until its outputs verify it.

Native validation, posterior integration and neural training remain blocked at
this checkpoint. The Luna monitoring agent became unavailable due to its usage
limit; the primary agent continues diagnosis and the local diagnostic watcher
provides 45-second status polling.

### Cache diagnosis confirmed and bounded repair

Read-only audit 001 completed at efda250: 109 tests passed; all 320 examples,
8,320 channel/recipe checks. One float32 sum failed: K4, F7,
sub-032339-000-blink-snr+5. Float64 accumulation passed every check under the
unchanged rtol=1e-5/atol=1e-11 contract. No solver failures, cache mutation or
new decompositions occurred. Maximum relative RMS reconstruction error was
3.992e-8. This is accumulation precision, not a failed VMD decomposition.

Repair changes only verification accumulation to float64, preserving all mode
vectors and residuals. A cancellation/damage regression fixture and full
actual-reader Kaggle audit 002 must pass before frontal-full-002 starts.
The failed full version 001 is retained. Scientific accuracy and preservation
thresholds remain unchanged.

## Deadline campaign: frozen frontal transfer

Archived guard17-cache-replay-003 reproduced 17.92721 dB with the original frozen K3/2000 reference-assisted model on its pinned, development-exposed Klados cache. This is historical whole-cap reconstruction, not a new verified frontal result.

Frontal-gain-hybrid-002 completed 320 examples / 32 recipients / 13 frontal channels; 115 numerical tests passed in the follow-up evidence stage. Frozen transfer output SNR was 8.76523 dB; anchoring 8.90208 dB; separate-axis support 7.84840 dB; combined 7.94956 dB. Matched direct regression reached 11.80749 dB, but its worst clean modification was 0.21764 and clean covariance error 0.23846. No arm passed qualification gates. Saved condition plots show lateral correction is the main weakness. Frontal-hybrid-evidence-001 saved condition tables, approximate recipient/donor uncertainty and inspectable figures, without changing predictions.

Bounded next tests preserve K3/2000, splits and targets: (1) consistent model input-unit conversion to diagnose absolute floors; (2) signed lagged calibration transfer with a matched raw versus VMD-supported pooled correction gain. The calibration hybrid is a project extension, not author reproduction. It closely matches the synthetic regression-field construction; native validation remains mandatory. Neither test trains weights, performs AutoVMD or decomposes posterior EEG.

The second reconstruction check inside FrontalCorrector also needed float64 summation. The previous reader-only audit did not exercise that layer; it cannot be called a full end-to-end repair. Source e5dd5a8 fixes both checks with the original tolerance and adds a regression fixture. The running full002 uses its older pinned source; any resulting failure stays archived and requires a new version.

Unit audit001 measured 8.76494 dB for microvolt-adapted frozen transfer versus 8.76523 dB unscaled, and 8.90178 dB anchored versus 8.90208 dB unscaled anchored. The numerical-floor hypothesis did not explain the observed performance gap. Preserve this negative control rather than describing unit conversion as an improvement.

## New calibrated frontal result

frontal-calibration-hybrid-001, source96c6cfd7c5109b6984c154d432348069c6c69db0, completed all 320 development examples and 116 numerical tests. K3/2000 calibration-mode gain achieved output SNR 17.112814 dB, improvement 22.585995 dB, temporal RRMSE 0.192181 and Pearson CC 0.973700. Three channel/example decomposition failures remain scored pass-through. All controlled engineering gates passed, with zero synthetic clean modification, alpha/beta change and clean covariance error.

The matched calibrated raw-gain control achieved 18.193247 dB, RRMSE 0.160311 and CC 0.981142 with no failures, also passing controlled gates. Therefore VMD has not demonstrated added benefit over that matched calibration control. Fixed windowwise direct regression scored 11.807489 dB and failed clean gates. The historical frozen-model transfer scored 8.765227 dB.

This is verified frontal-only, participant-balanced controlled development evidence on regression-generated ocular fields. It does not improve the archived 17.92721 score on an identical benchmark, establish independent generalization, qualify posterior integration or provide an EEG-only deployment model. Unscored calibration learned signed lagged transfer; scoring input alone fitted correction strength; paired targets entered only evaluation. Native OSF validation and matched uncertainty are next.

The completed evidence notebook reports approximate recipient/donor Poisson
multiplier uncertainty (32 recipients, five donors): VMD mean interval
14.4481-19.7451 dB. VMD minus matched calibrated raw gain is -1.0804 dB,
interval -2.4207 to +0.2482 dB. This does not support VMD superiority.
Condition/input-level VMD outputs are blink 16.84/20.06/22.11, lateral
8.95/13.13/16.62 and mixed 15.99/19.10/21.20 dB at -5/0/+5 dB input.
These approximate intervals are supplemental project statistics, not the
prescribed papers' native participant permutation procedure.

Native calibration version001 failed at strict JSON decision saving, after
the first trial prediction was produced. Its outputs are preserved. Version002
repairs scalar serialization without changing numerical settings. Separate
signed HEOG/VEOG gain fitting is evaluated as a fixed mechanistic challenger
in frontal-axis-hybrid-001; its result remains pending.

## Native preservation failure and detector extension

Native calibration002 completed 117 tests and 51 trials from two development-exposed OSF recordings. VMD blink correlations dropped from 0.913/0.959 to 0.051/0.106; lateral correlations from 0.273/0.422 to 0.025/0.045. These are suppression proxies, not reconstruction SNR. Rest-relative modification was 0.318/0.456, alpha error 1.030/1.934 dB and covariance error 0.171/0.392, failing the declared preservation gates. Do not authorize posterior integration or neural training from the controlled SNR alone.

Signed-axis hybrid001 passed 119 tests: VMD axis gain scored 15.396155 dB versus unchanged shared-gain 17.112814; matched raw-axis gain 17.091567 versus shared raw18.193247. The axis challenger does not improve mean accuracy.

New event-support comparison fits a reference-only logistic detector on each unscored calibration recording using rest/ocular trial labels. It thresholds above all calibration-rest logits and probability0.9, then restricts and tapers artifact correction within detected intervals. No scored condition or paired target enters inference. Settings are fixed; labels/detector fitting use only the authorized calibration. This is a new development extension motivated by native preservation failure, not independently validated evidence.


Gain-adjusted event support checkpoint: the unscaled event detector reached only 12.080396 dB (VMD) and 12.406342 dB (matched raw), failing controlled SNR and correlation gates. Native event validation is blocked. Saved severe-lateral gains were mostly below the existing bound of 8, so gain clipping does not explain most misses. Controlled mixtures change EEG ocular strength without changing donor EOG amplitude. A fixed gain-adjusted detector adaptation will test this mismatch, retaining the original correction, references for evaluation, K3/2000 and all gates. This is a hypothesis, not an achieved improvement.

Gain-adjusted logistic support001 completed120 tests, 320 examples and full score coverage. VMD scored11.463423 dB, matched raw12.615663 dB; both failed SNR/correlation gates. The unchanged shared controls remained17.112814/18.193247. No native follow-up launched. A fixed monotonic axis detector now compares separate HEOG/VEOG rest-calibrated amplitude thresholds, with and without the same gain adjustment. This tests a simpler interval policy without changing correction parameters, VMD, source access or qualification criteria.

Monotonic axis support001 completed121 tests and all320 examples. Unscaled temporal VMD16.079988 dB/raw17.109785 passed every controlled gate; gain-adjusted VMD14.823904/raw15.836632 failed (raw correlation deterioration0.027446). Ungated controls stayed17.112814/18.193247. Native unscaled validation launched as frontal-native-monotonic-001, source437198c, with identical detector and original evaluation EOG. A separate bounded controlled comparison adds whole-window routing and selected-mode subspace-normalized gain; it preserves all controls and the fixed decomposition. No accuracy improvement or native qualification is claimed for pending variants.

## Completed native monotonic and subspace checkpoints

Native monotonic001 completed both development records. Event-restricted VMD worst-record resting relative modification0.185129660, alpha0.360003534 dB, beta0.152969872 dB, covariance0.127834427. Alpha/beta pass, modification/covariance fail. Ungated VMD worst modification0.456386041. This improves preservation proxies but does not qualify integration or training.

Subspace-window001 completed320 examples. Selected-mode projected gain scored10.890616 dB and failed accuracy/correlation; rejected. Whole-window event routing scored17.112814 dB, identical to ungated VMD, so no benefit shown. Matched raw remains18.193247. The temporal monotonic VMD arm remains16.079988 and passes controlled gates but fails native preservation. None improves the archived17.92721 on an identical benchmark. All failures remain preserved.
