# Frontal results: paired Klados and native OSF

All processing is frontal-only. No LEMON or new AutoVMD search is active.
The full Klados and native OSF notebooks have completed and their actual
executed notebooks, tables, figures, predictions and modes are downloaded.

## Paired Klados: correlation with clean EEG

Eight archived test records, three windows each, seven frontal channels.
Mean Pearson CC with paired clean EEG changes from **0.698520 before cleaning**
to **0.984644 after the historical VMD + EOG model**. Its record-balanced
pooled-window output SNR is **16.134624 dB**; improvement is **16.928025 dB**.
The distinct mean-channel SNR is 17.496008 dB. The older 17.927210 dB used all
19 rows. These are different scopes/aggregations and cannot be substituted.

The paper-list mapping is zero-based rows 0/1/2/3/10/11/16 =
FP1/FP2/F3/F4/F7/F8/Fz. Labelled NPZ copies embed channel_names and
source_row_indices alongside predictions and historical modes. This uses the
user-requested paper list order; independent MATLAB-row binding is unverified.
No participant identity is inferred from record numbering.

An independent NumPy Pearson calculation verified all **1512** saved
method/window/channel clean-EEG correlations against the original table.
The mode audit separately covers **2005 component rows**, including retained
residuals, with clean EEG, HEOG and VEOG correlations. Clean targets are used
for diagnosis only, never runtime mode selection.

For example, at FP1 in the historical K3/alpha2000 decomposition:

| Mode | Measured center (Hz) | Mean absolute clean-EEG CC | HEOG CC | VEOG CC |
|---|---:|---:|---:|---:|
| 1 | 1.154 | 0.248 | 0.461 | 0.903 |
| 2 | 7.623 | 0.576 | 0.065 | 0.149 |
| 3 | 16.633 | 0.408 | 0.004 | 0.013 |

All three CC columns in this example are mean absolute coefficients, with equal
record weighting. Signed coefficients are also saved. A high EOG association
does not prove a mode contains only artifact; low CC with full-band clean EEG
does not prove a narrowband neural mode should be removed.

## Native OSF: unchanged-reference before/after correlations

Four development participants (p02–p05), 32 scored trials, with separate
unscored calibration. Every method sees the same five exact frontal names:
F3/F4/F7/F8/Fz. OSF FPz is not substituted for absent FP1/FP2. Posterior and
central EEG are not corrected. Frozen Klados neural weights are transferred;
EEGOAR-Net has two missing montage inputs, an explicit transfer limitation.

The blink column uses VEOG; the lateral column uses HEOG. Absolute correlations
are computed before averaging, then summarized within participant and across
participants. Full signed and absolute HEOG/VEOG values for every condition,
channel, trial and method are saved.

| Method | Blink abs VEOG CC after | Lateral abs HEOG CC after | Rest relative RMS change |
|---|---:|---:|---:|
| Before / identity | 0.918516 | 0.358906 | approximately 0% |
| Historical VMD + EOG model | 0.328867 | 0.214818 | 63.63% |
| SVMD–FCM adaptation | 0.367851 | 0.137215 | 64.55% |
| VME–GMETV adaptation | 0.367017 | 0.045383 | 49.76% |
| EEGOAR-Net, Klados fine-tuned | 0.889118 | 0.377380 | 14.59% |
| GEVD–MWF | 0.559549 | 0.301306 | 61.43% |
| BandRouteNet | 0.375989 | 0.276730 | 29.13% |
| Direct EOG regression control | 0.044528 | 0.063976 | 70.26% |

**The historical model reconstructs paired Klados well, but its native OSF
resting-input preservation fails.** Rest is not paired clean ground truth;
63.63% signal modification is not a measured percentage of brain activity lost.
It is nevertheless large and does not support deployment qualification. Low
EOG CC alone does not establish better cleaning: direct regression illustrates
the trade-off particularly strongly. EEGOAR-Net slightly worsens lateral CC.
Native OSF has no paired clean EEG, so no reconstruction SNR or clean-target
correlation is claimed for it.

## Recovered notebooks and provenance

- [Klados comparison, actual outputs](notebooks/executed/frontal-six-full-001.ipynb):
  source c196374, 29 contracts, 922 verified payloads, 13 figures, 6/6 executed
  code cells, zero error cells, zero method exceptions; 16 SVMD solver-cap
  warnings remain in evidence.
- [OSF correlations, actual outputs](notebooks/executed/frontal-osf-correlations-001.ipynb):
  source b9df0e6, 31 contracts, 240 verified payloads, 16 figures, 5/5 executed
  code cells, zero error cells and method exceptions.
- [Mode audit version 1, actual outputs](notebooks/executed/frontal-mode-clean-audit-001.ipynb):
  source 405d7aa, 32 contracts, 22 verified payloads, three figures, 3/3 code
  cells, zero error cells. Its paired mode correlations are complete.

- [Mode audit version 2, actual outputs](notebooks/executed/frontal-mode-clean-audit-002.ipynb):
  source 0876931, 32 contracts, 25 verified payloads, three figures, 3/3 code
  cells, zero error cells. All **1280 native prediction rows** independently
  match their saved before/after scoring table. **5071 native component rows**
  are recomputed against unchanged HEOG/VEOG.

The original prediction table already uses unchanged HEOG_lpf/VEOG_lpf
correctly. Its separate mode CSV used model-preprocessed references. Audit
version 2 repairs that diagnostic reference mismatch using unchanged references
embedded beside the saved modes. No denoising, training or original scoring
result changed. Original artifacts and audit version 1 remain preserved.
The three final notebooks contain 32 figure outputs in total (13 + 16 + 3).

Artifacts reside below results/<campaign>/<run_id>/ with hash-verified
retrieval_receipt.json files. SVMD–FCM and VME–GMETV are explicit adaptations:
unavailable full ocular-paper parameters prevent faithful reproduction claims.
All current Klados scores are development-exposed, not fresh confirmation.
The next decision is preservation repair and matched native validation before
regional integration; none of these results establishes production readiness.

---

# Klados frontal six-method results

Run `frontal-six-full-001`, SHA `c19637465e5a5c29991c2a988f25e6527a156f09`.

Smoke: **False**; 8 records, 24 windows, seven frontal rows.

| method               |   pooled_window_snr_db |   pooled_snr_improvement_db |    rmse |     mae |   rrmse_t |   rrmse_s |   pearson_cc |
|:---------------------|-----------------------:|----------------------------:|--------:|--------:|----------:|----------:|-------------:|
| historical_vmd       |              16.1346   |                    16.928   | 1.79144 | 1.34748 |  0.163901 |  0.210996 |     0.984644 |
| svmd_fcm_adaptation  |               0.952856 |                     1.74626 | 9.12081 | 6.13321 |  0.896051 |  1.99512  |     0.698607 |
| vme_gmetv_adaptation |               3.84742  |                     4.64082 | 7.08315 | 5.3012  |  0.645974 |  0.78392  |     0.757794 |
| eegoarnet_finetuned  |               9.02984  |                     9.82324 | 3.89531 | 2.95702 |  0.365776 |  0.452547 |     0.919062 |
| gevd_mwf             |               6.21072  |                     7.00412 | 5.51    | 4.18532 |  0.517064 |  0.873557 |     0.853614 |
| bandroutenet         |               6.93877  |                     7.73218 | 4.8914  | 3.71939 |  0.462799 |  0.600538 |     0.882719 |

Pooled-window SNR averages windows within record, then records in dB. Other metrics average channels/windows within record, then records. The old all-19-row 17.927210 dB is separate.

## Reference access

- historical_vmd: HEOG and VEOG required at inference.
- svmd_fcm_adaptation: EEG only; references used for evaluation only.
- vme_gmetv_adaptation: EEG only; references used for evaluation only.
- eegoarnet_finetuned: EEG only; author pretrained initialization, Klados training on explicit seven-channel montage.
- gevd_mwf: training EOG activity defines covariance masks; no runtime EOG.
- bandroutenet: paired Klados training; EEG-only inference.

## Engineering preservation, separate from paper metrics

Worst-record mean clean change<=1%, alpha/beta error<=0.5dB, covariance error<=0.02. The declared CC comparator here is matched direct EOG ridge, with worst-record mean deterioration<=0.005. Exceptions fail qualification.

| method               |   worst_record_mean_clean_change |   worst_record_mean_alpha_error_db |   worst_record_mean_beta_error_db |   worst_record_mean_covariance_error |   worst_record_cc_deterioration_vs_direct_ridge |   mean_output_snr_db |   failed_windows | preservation_gates_pass   | development_15db_target_pass   | production_qualified   |
|:---------------------|---------------------------------:|-----------------------------------:|----------------------------------:|-------------------------------------:|------------------------------------------------:|---------------------:|-----------------:|:--------------------------|:-------------------------------|:-----------------------|
| historical_vmd       |                         0        |                           0        |                          0        |                             0        |                                        0.001143 |            16.1346   |                0 | True                      | True                           | False                  |
| svmd_fcm_adaptation  |                         0.620222 |                           4.28865  |                          0.137621 |                             0.638233 |                                        0.536932 |             0.952856 |                0 | False                     | False                          | False                  |
| vme_gmetv_adaptation |                         0.732827 |                           0.016835 |                          0.006356 |                             0.583435 |                                        0.384766 |             3.84742  |                0 | False                     | False                          | False                  |
| eegoarnet_finetuned  |                         0.357802 |                           0.564426 |                          0.72838  |                             0.412456 |                                        0.135351 |             9.02984  |                0 | False                     | False                          | False                  |
| gevd_mwf             |                         0.778242 |                           1.46917  |                          1.15281  |                             0.73577  |                                        0.29669  |             6.21072  |                0 | False                     | False                          | False                  |
| bandroutenet         |                         0.365698 |                           1.16623  |                          0.989853 |                             0.453353 |                                        0.197426 |             6.93877  |                0 | False                     | False                          | False                  |

Failed method windows: 0. Solver-cap/nonconvergence warning entries: 16. See saved evidence; SVMD inner convergence is not exposed by its library.

## Interpretation limits

Historically exposed Klados record split; participant IDs unknown. No fresh confirmation or production claim. Frontal assignment uses the requested paper-list order with unverified MATLAB row binding. SVMD–FCM and VME–GMETV are documented adaptations with unavailable ocular-paper settings. EEGOAR-Net is fine-tuned on an explicit custom montage, not frozen electrode aliases. No LEMON or AutoVMD search. Runtime rows disclose cached VMD cost and device. Actual executed notebook, predictions, modes, H/V evidence, checkpoints and figures are saved. Smoke scores are not final accuracy.
