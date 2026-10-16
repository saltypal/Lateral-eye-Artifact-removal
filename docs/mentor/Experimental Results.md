# Regional EOG Artifact Removal Experimental Results

Research results through 16 October 2026

## 1 Summary of measured outcomes

The latest controlled development comparison achieved 17.804727 dB whole-head channel-first mean output SNR and 18.261209 dB frontal mean output SNR using an anchored frontal VMD correction with a solver-failure raw fallback and matched nonfrontal regression. The direct-regression control achieved 17.785746 dB and 18.182199 dB, respectively. The incremental whole-head difference was +0.018981 dB with a supplemental 95% interval spanning zero. VMD superiority is therefore not established.

Native OSF evaluation showed reduced ocular association but excessive resting-signal modification. The classical teacher and EEG-only deployment model remain unqualified. This document presents measured results and their evaluation scope; historical branches use different corpora and must not be treated as one comparable leaderboard.

| Evidence set | Scope | Interpretation |
|---|---|---|
| Historical Klados replay | 24 windows from eight reused records; electrode identities unknown | Development-exposed EOG-assisted hybrid; not VMD alone |
| Latest controlled LEMON OSF corpus | 320 examples; 32 recipient windows; five ocular donors | Constructed contamination added to low-ocular reference EEG; not independent native ground truth |
| Latest native OSF pilot | Two development recordings; 51 scoring trials | Ocular suppression and resting preservation; no reconstruction SNR |
| Neural pilots | Five epochs on bounded development data | Preliminary learning results; no final model qualification |

Output SNR compares retained reference energy with reconstruction-error energy. Channel-first mean SNR and mean pooled-window SNR use different aggregation and are reported separately. Native EEG has no paired clean target and receives correlation and preservation measures. All numerical results were produced on Kaggle; this document assembles saved evidence.

<!-- PAGE -->


## 2 Literature and the fixed frontal correction screen

| Reference | Idea affecting our design | Implementation status |
|---|---|---|
| Dora and Biswal 2020 [6] | Detect contaminated segments, estimate ocular content with VMD, then regress | Full numerical settings unavailable; exact reproduction deferred |
| Liu and Zhang 2022 [7] | VMD followed by SOBI and reference/entropy source selection | Independent selector and solver adaptation; author helpers missing |
| Xiong et al 2024 [8] | Separate SVM detection, optimized VMD, SOBI and approximate entropy | Fixed-parameter adaptation; GA and mRMR deferred |
| Schlögl et al 2007 [9] | Reference regression as a legitimate ocular correction method | Our lagged ridge/gain is an extension, not exact replication |
| Plöchl et al 2012 [10] | Distinct ocular contributions and spatial patterns | Motivates signed axis evidence; not eye-tracker-guided replication |
| Issa and Juhasz 2019 [11] | Repair contaminated component intervals | Wavelet/ICA adaptations later tested; not full published-method parity |

The 2022 author repository lacks required FuzzyEntropy/SNR helpers and uses VMD/SOBI conventions different from ours. Our observed HEOG/VEOG cannot be equated to an unspecified supplied or true simulated artifact reference. The 2024 paper’s printed optimum conflicts with its declared parameter ranges. We recorded the discrepancy rather than inventing a justified setting. Missing full settings prevent exact-replication claims.

### Completed correction screen

708 correction recipes were evaluated on 20 examples from five recipients/five donors and 13 verified frontal channel names. Fixed K3/K4 decompositions were used; this was a correction-bank screen, not a resumed AutoVMD search. Selected results below are matched on that screen.

| Fixed method | Frontal output SNR dB | Pearson CC |
|---|---:|---:|
| Direct joint HEOG/VEOG regression | 12.7468 | 0.97217 |
| Projected VMD K3 / K4 | 12.1386 / 12.3030 | 0.95825 / 0.95890 |
| Projected SOBI K3 / K4 | 12.1348 / 12.2936 | 0.95819 / 0.95863 |

Both VMD arms miss 15 dB and exceed the declared CC-deterioration limit. Added separation did not establish a benefit. Whole-mode/entropy variants remained in the negative-result tables.

Evidence: [E2], [E5].

<!-- PAGE -->

## 3 Historical replay and calibrated frontal transfer

### What actually produced the historical 17 dB result

The guarded replay reached 17.927210 dB on 24 windows from eight reused Klados records, using all 19 rows with unknown anatomy. It required K3/2000 VMD, HEOG/VEOG, a 2,499-parameter EOGContextGainStudent MLP and a support gate. It was not VMD alone, an EEG-only deployment model, or verified frontal/posterior integration. The archived aggregation differs from the newer regional protocol. An 18.0249 dB alternate guard failed its clean-change criterion and was not the retained safe recipe.

Direct transfer of that mechanism did not reproduce 17.93 on the newer controlled corpus: frozen frontal and anchored variants scored about 8.77/8.90 dB. Consistent-unit replays did not rescue them. This rejected a simple “reuse K/alpha and recover the old score” argument.

### New calibration-based field estimate

Using unscored same-record EEG and HEOG/VEOG, we fit a signed ocular transfer. Reference rows use lags −10/0/+10 samples, or ±50 ms at 200 Hz; ridge penalty is 0.01 and trial-boundary lag columns are excluded. Observed scoring EEG provides a bounded gain. Association 0.4 controls intervention; selected modes use declared 0.2 reference evidence. These are project settings, not universal paper optima.

| Earlier matched controlled frontal method | Output SNR dB | MSE V² | Ordinary CC |
|---|---:|---:|---:|
| Calibrated VMD mode gain | 17.112814 | 4.298972e−12 | 0.973700 |
| Matched calibrated raw gain | 18.193247 | 3.001322e−12 | 0.981142 |

Calibration transfer substantially improved this controlled branch. VMD still did not beat the raw control. Its worst-source frontal CC deterioration was 0.00744148 versus the 0.005 project limit. Idealized clean controls had zero measured modification; that did not establish preservation on native rest.

**Bottleneck:** severe lateral contamination remained much weaker than blink. VMD frontal lateral output was 8.9544/13.1292/16.6229 dB at nominal whole-input −5/0/+5. We investigated timing and axis-specific correction rather than treating the aggregate as sufficient.

Evidence: [E6], [E7].

<!-- PAGE -->

## 4 Temporal support and separate ocular axes

The hypothesis was that correction should act during contaminated intervals rather than throughout any window containing ocular activity. Calibration-derived amplitude/derivative features and separate HEOG/VEOG support were tested. Scoring condition labels were not used to route inference.

| Earlier controlled frontal adaptation | VMD output SNR dB | Finding |
|---|---:|---|
| Separate-axis gain | 15.396155 | Worse than shared gain |
| Logistic support | 12.080396 | Lost reconstruction accuracy |
| Gain-adjusted logistic support | 11.463423 | Proposed amplitude repair failed |
| Monotonic temporal support | 16.079988 | Controlled screens passed; native preservation failed |
| Whole-window support | 17.112814 | Identical to ungated result; no added benefit |

On native rest, monotonic VMD reduced worst-record modification to 18.51%, but covariance error remained 0.127834; both fail the 1%/0.02 limits. A 48-active-arm replay varied association and correction strength without new VMD or fitting. No active setting passed all preservation limits.

:::figure bounded 6.65 2.42

**Figure 3.** Earlier two-record native feasibility experiment, not the latest frozen pilot. Horizontal axis is correction strength; dots/crosses represent raw/VMD and shared/separate support across association settings. Dashed lines are unchanged engineering limits. More conservative correction reduces modification, but low modification alone is insufficient: conservative endpoints also lose substantial blink/lateral suppression.

Original-trial support, calculated before reflected overlap-add padding, changed VMD separate worst modification from 15.8564% to 15.4503% and covariance from 0.108051 to 0.102743. This isolated boundary change did not qualify the method. The hypothesis that window support alignment alone solves preservation was not supported.

Evidence: [E8], [E9].

<!-- PAGE -->

## 5 The rejected mode shape restriction

### Hypothesis

Perhaps a calibrated ocular estimate included neural content outside the selected VMD span. We tested projecting its centred variation into selected mode vectors while retaining the artifact mean, with fixed gain/detection/lag settings. Original modes, calibration and nonfrontal outputs remained matched.

| Earlier full controlled frontal method | Output SNR dB | Temporal RRMSE | Pearson CC |
|---|---:|---:|---:|
| Original calibrated VMD | 17.112814 | 0.192181 | 0.973700 |
| DC-preserving shape restriction | 2.521295 | 0.777803 | 0.644872 |
| Calibrated raw control | 18.193247 | 0.160311 | 0.981142 |

The numerical run passed 236 tests and reproduced archived baselines. The new correction failed scientifically. Later frozen-output diagnosis found that useful correction content had been discarded; it did not establish that a restricted mode span was pure ocular space.

### What this failure means

The 2.52 dB value belongs to this rejected challenger. It is not the final score of every VMD method. Conversely, a failure in this one adaptation does not prove every mode-constrained approach is impossible. The tested mode span was incomplete for the correction task under this protocol.

We preserved the residual precisely because VMD’s finite modes need not account for every useful waveform contribution. Residual activity is not automatically neural and not automatically ocular. Its content must be evaluated rather than thrown away or blindly removed.

### Why the diagnostic was useful

The decomposition, correction gain, support policy and subtraction rule are different mechanisms. An experiment should change one mechanism while holding others fixed so a failure can be interpreted. Here, the evidence rejected the particular shape restriction and prevented an unsupported native/model promotion.


**Evidence:** [E10]. The report tables retain every condition, not only the aggregate failure.

<!-- PAGE -->

## 6 Posterior ICA and GEVD MWF with frontal support

The other chat’s versioned posterior comparison is part of this research record. It compared posterior-only, common frontal average, signed regional summaries, raw frontal channels and full-montage support. The fitted decomposition can use broader support while correcting posterior outputs only. No posterior VMD is active in the current scope.

ICA estimates statistical components; selected-component correction must preserve reconstruction residuals. GEVD–MWF uses ocular/rest covariance information and regularization to estimate artifact-related spatial directions [12]. Our implementation uses changed covariance estimation and declared masks, so it is an adaptation of the reference approach.

Signed context includes left/right/midline summaries, common frontal activity and right-minus-left activity. A common average can cancel opposing lateral patterns. This motivates retaining polarity, but does not prove every signed-context recipe superior.

| Best candidate in earlier posterior development search | Posterior output SNR dB |
|---|---:|
| Direct EOG regression | 13.542 |
| Full-montage ICA | 13.523 |
| Raw-frontal-supported ICA | 13.168 |
| Signed-summary ICA | 12.918 |
| Common-frontal ICA | 12.516 |
| Posterior-only ICA | 11.815 |
| Signed-summary GEVD–MWF | 12.335 |
| Posterior-only GEVD–MWF | 11.786 |

The completed run recorded 737 candidates, 320 examples, 32 recipients, 1,440 calibration fits and no fit failures/exclusions; 92 tests passed. The candidate rows above are partial-gate search evidence, not unbiased final winner estimates. The fixed regression comparator did not receive the same broad optimizer, which limits claims about algorithm families.

**Bottleneck:** neither posterior point estimates nor clean synthetic controls establish a native-qualified teacher. We proceeded to controlled regional integration with matched calibrations and information access.


**Evidence:** [E11].

<!-- PAGE -->

## 7 Regional integration exposed the missing middle

Initial integration combined fixed frontal VMD with posterior regression or ICA and passed central/shared channels through. Whole-cap output was 9.094213 dB with posterior regression and 9.089350 dB with posterior ICA; matched raw frontal plus posterior regression was 9.337592 dB. Regional scores cannot be added to obtain these values.

:::figure errors 6.65 2.83

**Figure 4.** Earlier frozen regional diagnostic. Vertical axis is mean per-example fraction of reconstruction-error energy, then source/condition balanced. Central/shared pass-through contributes 87.9066%; frontal 2.9326%; posterior 9.1608%. This is not a globally pooled energy ratio and does not prove that removing all central error is feasible without neural loss.

| Earlier full regional composition | Whole output SNR dB | Central/shared SNR dB |
|---|---:|---:|
| Central pass-through | 9.094213 | 1.954134 |
| VMD frontal artifact context for central transfer | 13.162704 | 10.903434 |
| Raw frontal artifact context for central transfer | 13.183201 | 10.949306 |

Both context arms retained the same scored VMD frontal and direct posterior outputs. “Raw context” did not mean replacing the frontal denoiser or an EEG-only method. The raw-minus-VMD context whole difference was 0.020497 dB with an interval crossing zero. Context coverage helped; unique benefit from VMD-derived context remained unproven.

Evidence: [E12], [E13].

<!-- PAGE -->

## 8 Nonfrontal challengers and the stronger simple control

We tested six fixed mechanisms in posterior-only, central-only and joint compositions, preserving frontal predictions. This included ICA rejection, component regression, wavelet–ICA, selective wavelet–ICA and reference-field rank reduction. Wavelet adaptations were actually evaluated; earlier “reviewed only” descriptions do not describe this later run.

| Joint nonfrontal method with frozen frontal VMD | Whole mean output SNR dB |
|---|---:|
| Saved VMD-context control | 13.162704 |
| ICA component ridge | 14.735326 |
| ICA rejection | 12.541144 |
| Reference-field rank 1 / rank 2 | 12.453789 / 15.155692 |
| Wavelet–ICA / selective wavelet–ICA | 12.396655 / 12.333200 |

A separate 50-example fixed GEVD–MWF rank-2 joint control scored 12.619925 dB against a matched 13.994230 dB context control. This rejects those fixed recipes, not all published MWF methods. A later fixed blend of the rank-2 field reached 15.370676 dB mean but 13.826832 dB pooled-window SNR; its gain interval crossed zero and severe lateral output worsened from 8.050232 to 7.758159 dB.

### CCA versus complete lagged reference regression

CCA seeks directions shared with references. The full 320-example comparison tested rank-one/two reference-side and EEG-side versions. Removing reference directions did not beat retaining the complete regularized field.

| Frozen frontal VMD plus nonfrontal method | Whole mean dB | Mean pooled-window dB |
|---|---:|---:|
| Complete calibrated lagged ridge | 17.533717 | 14.780705 |
| Reference-side CCA rank 2 | 15.833574 | 13.923972 |
| EEG-side CCA rank 2 | 13.597170 | 12.168657 |
| Previous reference rank 2 | 15.155692 | 13.480444 |

CCA reference-rank2 minus complete ridge was −1.700143 dB, supplemental 95% interval [−2.548602, −0.943181], and failed its paired-CC gate. A later matched frontend audit gave raw frontal plus identical complete ridge 17.777095 mean/15.126152 pooled versus VMD 17.533717/14.780705. Simpler controls must remain visible.

Evidence: [E14].

<!-- PAGE -->

## 9 Corpus repair and the anchored lateral correction

### Why source and reference coupling required an audit

Earlier controlled construction had a field/reference scaling mismatch that limited interpretation of calibration transfer. We recovered original recipient windows and unscored calibration from pinned sources, then constructed a separately versioned homogeneous field/reference scaling corpus. This was not algebraically extracting clean EEG from contaminated scoring data. Old results remain evidence under their old protocol; they are not silently overwritten or merged with the repaired corpus.

### The lateral gain hypothesis

The old mode-supported gain could under-estimate correction when useful ocular-associated activity lay outside selected modes or in the residual. In one severe lateral example, selected-mode gain was 0.863411 versus raw gain 1.030755. This supported a mechanism experiment; it did not prove all residual content ocular.

Matched-subspace scalar/axis projections were tested and rejected. A fixed anchored gain used 75% raw gain plus 25% VMD gain, with the same observed-data gate and bounds. This is our declared adaptation, not a published optimum. It remains one full-field artifact estimate subtracted from the original.

| Two-pair pilot frontal lateral output | Old VMD dB | Anchored dB | Raw shared dB |
|---|---:|---:|---:|
| Nominal whole-input −5 dB | 10.655937 | 13.885189 | 14.061627 |
| Nominal whole-input 0 dB | 14.154767 | 18.488301 | 18.635700 |
| Nominal whole-input +5 dB | 19.472830 | 22.965294 | 22.682389 |

The anchor improved old-VMD lateral performance and some MSE/CC summaries, but did not win every condition or aggregation. Both recipients were development-exposed. This small pilot justified broader fixed-recipe evaluation, not a general superiority claim. Its 75% raw contribution makes the role of regression explicit.

The broader repaired corpus fixed 32 recipient windows, five donors and ten variants per pair. New inputs required new modes tied to input hashes. Nonfrontal ridge was computed once and matched across all corrected frontal arms. No renewed K/alpha search or native-label tuning occurred.

Evidence: [E15], [E16].

<!-- PAGE -->

## 10 Latest controlled results and the size of the VMD benefit

The broader run processed 320 examples, 4,160 frontal decompositions and three solver-cap cases. A cached follow-up reproduced the original metrics and intervals, repaired failure-count reporting, and replaced only failed frontal rows with matched saved raw predictions. It performed no new VMD or calibration fit.

| Latest method | Frontal output dB | Whole output dB | Whole pooled-window dB |
|---|---:|---:|---:|
| Shared direct ridge | 18.182199 | 17.785746 | 15.211023 |
| Original VMD gain | 17.079584 | 17.537370 | 14.852226 |
| Anchored VMD | 18.238703 | 17.799616 | 15.116150 |
| Anchored with solver-only raw fallback | 18.261209 | 17.804727 | 15.200363 |

:::figure conditions 6.65 4.36

**Figure 5.** Broader cached development comparison. Horizontal axis is nominal whole-input SNR; vertical axis is mean pooled-window output SNR, not the channel-first table above. Columns separate blink/lateral/mixed; rows separate frontal/whole output. Lateral remains weakest. The fallback does not change lateral outputs because its three affected examples were blink/mixed.

**Inference:** fallback-minus-raw whole mean difference is only +0.018981 dB, supplemental 95% interval [−0.071359, +0.132357]. Worst-recipient frontal CC deterioration remains 0.006224758 > 0.005. Neither VMD superiority nor teacher qualification follows. Evidence: [E17].

<!-- PAGE -->

## 11 Native OSF validation is the decisive current checkpoint

The frozen native run completed all 51 scoring trials from two development recordings, 306 windows and 4,284 frontal K3/2000 decompositions. Four solver-flagged channel windows remained visible. All 341 tests passed and all 744 output payloads were recovered and independently hash-checked. Posterior channels passed through unchanged.

| Native condition | Original absolute EEG–EOG r | Direct raw | Anchored plus fallback |
|---|---:|---:|---:|
| p02 lateral, HEOG | 0.273208 | 0.058827 | 0.047653 |
| p03 lateral, HEOG | 0.422304 | 0.070063 | 0.055387 |
| p02 blink, VEOG | 0.913125 | 0.048049 | 0.043421 |
| p03 blink, VEOG | 0.958671 | 0.044100 | 0.047265 |

These are source/condition means of channel correlations against unchanged publisher derivatives. Lower association supports ocular suppression; it is not clean reconstruction accuracy. Two exposed participants do not support a population superiority claim. Full vertical and other-method tables remain in [E18].

| Worst record mean resting statistic | Raw shared | Anchored plus fallback | Required limit |
|---|---:|---:|---:|
| Relative RMS modification | 0.569527 | 0.541158 | 0.01 |
| Alpha error dB | 2.097657 | 2.076375 | 0.5 |
| Beta error dB | 0.967177 | 0.967189 | 0.5 |
| Relative covariance change | 0.433347 | 0.425106 | 0.02 |

Every active arm fails all four resting criteria. Identity passes by construction and removes no artifact. Anchored/fallback modification is 36.55% for p02 and 54.12% for p03. These are relative RMS changes, not percentages of neural activity proved lost. Rest may contain residual ocular activity; large changes nevertheless fail our declared engineering criteria. Worst statistics can come from different records.

**Decision:** teacher remains unqualified. The controlled 17.80 dB does not overrule native preservation. No reconstruction SNR is valid against native contaminated input. A dedicated latest-pilot native figure was not generated; this report uses the saved numerical tables rather than inventing a new plot.

Evidence: [E18].

<!-- PAGE -->

## 12 Neural experiments and what deployment would require

### What was implemented and actually trained

The earlier band-routing adaptation includes a shared width-32 encoder, two scales, Inception kernels 3/7/15/31, group normalization, GRU bottleneck and component attention: 222,211 parameters. A separate shared TCN–BiGRU student has five dilated blocks, 32 units per direction and 47,427 parameters. Software fixtures exercised model arms, CUDA optimization and fresh CPU reload. Those are implementation checks, not denoising qualification.

| Five-epoch development pilot | Output SNR dB | Worst-source clean relative error |
|---|---:|---:|
| Fixed-band MSE | 7.772145 | approximately 0.479 |
| Student MSE | 7.364268 | 0.478943 |
| Fixed-band preservation loss | 7.458374 | 0.554103 |
| Student preservation loss | 7.619747 | 0.460719 |

All four pilots failed clean preservation. They used a bounded development fold/seed, not completed multi-seed grouped final training. The preservation loss added identity, log-PSD and covariance terms with project weights 1.0/0.1/0.05. The results do not establish that longer training, a larger model or a larger SNR penalty will solve preservation. A diagnosis dispatcher failed after 78 tests and was repaired before the replacement diagnosis; this was an integration failure, not a cleaning score.


## 13 Evaluation interpretation and evidence access

Primary paired outcomes are MSE, temporal RRMSE, ordinary Pearson correlation, explicitly defined output SNR and SNR improvement. Native outcomes use ordinary absolute EEG HEOG correlation for lateral movement, EEG VEOG correlation for blink and vertical movement, resting RMSE and spectral preservation. EOG references remain unchanged between before and after calculations.

Project engineering limits are separate from published metric definitions: worst-source mean relative clean or resting modification at most 1%, alpha and beta power error at most 0.5 dB, relative covariance error at most 0.02, and paired correlation deterioration at most 0.005 against the declared comparator. The latest candidate fails preservation and is not production qualified.

The latest native run passed 341 numerical and source-contract tests and all 744 output payloads were recovered and hash verified. Test passage verifies implemented contracts, not scientific efficacy. No completed reserved confirmation, independent waveform benchmark or final EEG-only deployment qualification is claimed.

Evidence identifiers follow the companion Full Research Report and its evidence register. Source reports are retained in docs/FRONTAL_RESULTS.md, docs/GUARDED_REPLAY_RESULTS_20261011.md, docs/NATIVE_SEPARATE_AXIS_RESULTS.md, docs/FRONTAL_BOUNDED_FEASIBILITY_RESULTS.md, docs/FRONTAL_ARTIFACT_SHAPE_RESULTS.md, posterior results.md, docs/REGIONAL_RESULTS.md, docs/REGIONAL_DIAGNOSTIC_RESULTS.md, docs/NONFRONTAL_CHALLENGERS_RESULTS.md, docs/LATERAL_GAIN_RESULTS_20261015.md, docs/BROADER_FRONTAL_RESULTS_20261015.md and docs/NATIVE_FROZEN_FRONTAL_RESULTS_20261015.md. Evaluation equations and paper mappings are retained in docs/EVALUATION_PROTOCOL.md. Full run histories and detailed tables remain in Work So Far.md.

The latest native source commit is c48af6a0cda4d6bd905856613a3cf0c01f5bdcd2. Its artifact manifest SHA256 is ccc9253ee7b71f676bfacf805c5d7aabf3b4fed3cae147abad707c04ff071111.
