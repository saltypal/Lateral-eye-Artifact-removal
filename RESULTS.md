# Current results and frontal correlation status

The full Klados frontal comparison is complete. All **922 artifacts** were
retrieved and verified. The executed notebook contains **6/6 executed code
cells, 13 inline figures, five result tables and zero error cells**.

Open the [executed Klados notebook](notebooks/executed/frontal-six-full-001.ipynb).

Correlation with paired clean EEG rises from **0.698520 before cleaning** to
**0.984644 after the historical frontal VMD model**. Its pooled-window,
record-balanced output SNR is **16.134624 dB**. The distinct mean-channel SNR is
17.496008 dB; these aggregations cannot be substituted. The historical
17.927210 dB result used all 19 rows and remains a separate result.

The frontal OSF run has completed four participants and 32 scored trials.
Its artifacts are being retrieved. A separate CPU audit checks every saved
frontal mode against clean EEG, HEOG and VEOG. Native OSF has no paired clean
EEG target; its results use unchanged-reference ocular correlations and resting
preservation proxies. All methods use the five exact shared frontal names
F3/F4/F7/F8/Fz, without substituting FPz for FP1/FP2.

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
