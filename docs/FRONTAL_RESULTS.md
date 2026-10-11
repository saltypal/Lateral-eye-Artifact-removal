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
  RUNNING at this snapshot. Targets are diagnostic-only and never enter inference.
- Smaller repaired pilot `frontal-pilot-002` remains RUNNING; its logs have not
  yet matched the requested run. That is unavailable progress evidence, not a
  numerical success or confirmed exception. The screen supplies current recipes
  without waiting for this pilot.
- Native OSF pilot follows verified full evaluation. It uses the pinned restoration
  bundle, audits all development SET/FDT files, preserves trial boundaries and
  compares unchanged HEOG/VEOG association and rest RMSE/PSD/preservation.
- Posterior integration, EEG-only student training, external evaluation and export
  remain pending. The local queue stops at frontal review and cannot launch them.

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
