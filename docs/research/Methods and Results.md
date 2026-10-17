# Regional EOG removal: methods, evidence and results

Research record • 17 October 2026 • Klados paired EEG and native OSF eye-artifact EEG

This document separates measured results, implementation adaptations and proposed work. **VMD is our preferred frontal representation. The historical VMD–EOG hybrid achieved 16.13 dB frontal output SNR versus 15.23 dB for the original direct-regression control.** This supports that specific comparison, not universal VMD superiority. A newer calibrated ridge control scored 17.49 dB with lower clean-target correlation; both controls are reported below. No current method is production-qualified. All numerical results and scientific figures were produced on Kaggle; this document assembles saved evidence without rerunning signal processing.

## 1. Datasets and electrode provenance

**OSF.** The [eye-artifact dataset associated with SGEYESUB](https://doi.org/10.17605/OSF.IO/2QGRD) contains named EEG channels, horizontal and vertical EOG references, and annotated eye-movement/rest conditions, making it useful for native ocular suppression and preservation assessment. Our completed native frontal comparison uses four development participants, p02–p05, and 32 scored trials across blink, lateral, vertical and resting conditions, with separate unscored calibration. Five exact common frontal names are F3/F4/F7/F8/Fz: FPz was not substituted for absent FP1/FP2. The central/temporal study examines 21 FC/C/CP and six FT/T/TP channels in the available sessions. HEOG_lpf and VEOG_lpf remain the scoring references, with necessary resampling disclosed. Native recordings have no paired clean EEG: reduced EEG–EOG correlation is a suppression proxy, and resting modification is a preservation proxy. Neither is ground-truth reconstruction accuracy, so native OSF receives no output reconstruction SNR or clean-target CC.

**Klados.** The [Klados publication](https://doi.org/10.1016/j.dib.2016.06.032) describes a semi-simulated dataset with paired reference EEG and EEG contaminated with ocular activity; the [publisher data release](https://data.mendeley.com/datasets/wb6yvr725d/4) supports reproducible paired evaluation. The audited local export contains 53 records with 19 EEG rows and 5,401 samples per record, corresponding to the publisher's initial samples; publisher recording 45 is absent. The split is 37/8/8 records, but participant identities are unknown and the eight test records are historically development-exposed. At the user's instruction, the publication's electrode-list order was assigned to local rows and saved in labelled NPZ copies with `channel_names` and `source_row_indices`. Crucially, the publication lists electrodes but independent binding of its MATLAB row numbers to that list remains unverified. Regional findings are conditional on that mapping assumption; the original arrays did not independently establish anatomical labels.

| Zero-based rows | Assigned names | Reported region |
|---|---|---|
| 0, 1, 2, 3, 10, 11, 16 | FP1, FP2, F3, F4, F7, F8, Fz | Frontal |
| 4, 5, 17 | C3, C4, Cz | Central |
| 12, 13, 14, 15 | T3, T4, T5, T6 | Temporal, reported separately |
| 6, 7, 8, 9, 18 | P3, P4, O1, O2, Pz | Posterior |

Main processing uses 200 Hz, 0.5–40 Hz, 1,024-sample windows and a nominal 512-sample hop. Archived comparisons use their explicitly saved windows; the later calibrated regional studies score starts 2000/3024/4048 after a separately processed 2,000-sample calibration prefix. Filtering is independent across trial and calibration/scoring boundaries. Method-specific resampling is documented below. Scores from different window/calibration protocols must not be silently pooled.

**Retired dataset work:** earlier LEMON–OSF mixtures used low-ocular recipient EEG and regression-derived ocular fields. Low-ocular EEG is not proven artifact-free, and that construction could favor methods related to the mixing model. LEMON is excluded from the active comparisons. Its historical experiments remain archived rather than being relabelled as Klados or native OSF evidence.

## 2. VMD, AutoVMD and the historical model

### VMD: what K and alpha actually control

[Variational Mode Decomposition](https://doi.org/10.1109/TSP.2013.2288675) decomposes a channel into K adaptive, frequency-concentrated time-series vectors: `x = sum(u_k) + residual`. Each mode has the same sample count as the input; it is not a scalar feature or a predetermined EEG band. K controls the number of modes, while alpha controls the bandwidth penalty; increasing alpha generally encourages narrower spectral concentration, rather than setting a cutoff in hertz. The modes' measured spectra, center frequencies, bandwidths and reference associations determine their interpretation afterward. Our historical recipe uses K=3, alpha=2000, input-RMS normalization, uniform initialization, tolerance 1e-6 and a 2,000-iteration cap. These values preserve the historical recipe; they are not a demonstrated global optimum. Close centers diagnose possible overlap but do not alone prove duplication. Fixed delta/theta/alpha/beta/gamma bins are a different representation and do not justify choosing K=5. Artifact removal subtracts an estimated ocular contribution from the original signal while retaining reconstruction residuals.

### AutoVMD: what was planned and what actually happened

AutoVMD was a parameter-selection campaign over K=3–10 and alpha=250/500/1000/2000/4000: eight K values times five alphas, **40 configurations**, not 50. Decompositions were cached with identities, modes, residuals, center frequencies and numerical diagnostics. Planned neural screening was five epochs for all settings, fifteen epochs across three inner folds for eight shortlisted settings, and up to eighty epochs for the best two, followed by outer-fold evaluation. That complete selection process did not finish and no optimal recipe was established. Cache availability must not be confused with completed routing-model training. The historical K3 archive contains 1,600 verified NPZ entries; failed and repaired cache jobs, including K5, remain preserved. **AutoVMD is stopped and is not an active search in this document.** K3/2000 is the fixed historical continuation, with K4/2000 retained only as an existing diagnostic comparison.

### Historical VMD–EOG hybrid

The historical cleaner is `EOGContextGainStudent`, a 2,499-parameter shared per-channel MLP with dimensions **17 → 48 → 32 → 2** and GELU hidden activations. It consumes mode/reference associations, mode energies, amplitude/statistical descriptors and pooled ocular-support context. Explicit HEOG/VEOG projections estimate reference-explained activity; the network predicts bounded correction gains, and a support gate controls subtraction. In plain language, **projection means estimating the part explained by the reference; gain means the multiplier applied to that estimate**. Runtime requires EEG, HEOG, VEOG and VMD. This is neither VMD alone nor the planned EEG-only deployment student, and its use of regression must remain explicit.

| Historical evaluation | Output SNR dB | RMSE | Clean EEG Pearson CC | Scope |
|---|---:|---:|---:|---|
| Raw replay input | 2.902146 | — | — | All 19 rows |
| Frozen classical reference-VMD baseline | 9.4057 | — | — | Historical all-row comparison |
| Guarded historical VMD–EOG model | **17.927210** | **1.507619** | **0.989990** | Eight reused records, all 19 rows |
| Historical model in later frontal comparison | **16.134624** | **1.791442** | **0.984644** | Seven frontal rows, 24 windows |

The historical replay passed 40 numerical contracts and its scored clean-input modification was zero under the frozen gate. An unconstrained candidate reached 18.024929 dB but exceeded the 1% clean-change gate (1.37846% worst-record change), so it was not the accepted guarded result. These are development results, not fresh generalization. The later OSF transfer failure shows why the paired replay alone cannot establish native preservation.

### How we identify an ocular-associated mode

We inspect the mode waveform, spectrum, center, timing, and signed/absolute correlations with HEOG and VEOG. Paired clean EEG correlations are diagnostic only; clean targets never select a mode at runtime. At FP1, the saved record-balanced mean absolute associations were:

| Mode | Center Hz | Clean EEG CC | HEOG CC | VEOG CC |
|---|---:|---:|---:|---:|
| 1 | 1.154 | 0.248 | 0.461 | 0.903 |
| 2 | 7.623 | 0.576 | 0.065 | 0.149 |
| 3 | 16.633 | 0.408 | 0.004 | 0.013 |

Mode 1 is strongly VEOG-associated, consistent with blink contamination, but it is not proven pure artifact. A narrowband neural mode can have low correlation with full-band clean EEG and still be useful. The component audit covers 2,005 paired component rows, including residuals. Native audit version 2 independently verifies 1,280 prediction-correlation rows and recomputes 5,071 component rows against unchanged references. It repairs a diagnostic reference mismatch in the older native mode CSV; original prediction scores were already correct.

## 3. Regional splitting: rationale and limits

We separate frontal, central, temporal and posterior electrode groups because ocular contamination and preservation requirements vary spatially. Blink fields are often strongest near the eyes, making frontal channels a sensible priority for per-electrode VMD; lateral movements can also affect temporal channels strongly. Central means FC/C/CP where available, with temporal FT/T/TP reported separately, and posterior includes the verified parietal/occipital group. These are scalp-electrode groups, not uniquely localized brain sources. VMD is a good architectural fit for independent frontal processing and variable channel counts because it need not fit a full spatial mixing model. That makes it our preferred frontal representation, not proof that it is universally best. Nonfrontal correction must be independently justified: distance from the eyes does not automatically mean negligible contamination, and low contamination increases the cost of unnecessary correction. Unknown anatomy is never assigned a guessed region.

## 4. Why direct regression is insufficient as the final solution

Direct regression estimates `artifact = beta_H HEOG + beta_V VEOG` and subtracts it, optionally using lagged references and ridge regularization. It is fast and provides an important control, but it is a poor **sole basis for claiming physiological cleaning**: reference electrodes can contain neural activity; genuine EEG can be correlated with ocular behavior; calibration coefficients can change across movements and sessions; and reducing reference correlation can remove useful activity. With no paired clean native signal, good coefficients cannot be certified merely from a small residual correlation. Our OSF direct-regression control reduced blink VEOG association to 0.0445 yet modified resting input by 70.26%, demonstrating the danger. Calling all regression scientifically bad would be incorrect: published REG-ICA and our own historical hybrid use it selectively. The defensible criticism is indiscriminate or insufficiently validated regression, not the mathematical tool itself.

## 5. Frontal candidates: mechanism, paper, dataset and implementation

### 5.1 Historical VMD–EOG model

**Why selected:** it is our strongest historical paired frontal reference and allows an exact replay rather than an invented replacement. Its mechanism and reference requirements are specified in Section 2. It is a project-developed model, not a published standalone VMD denoiser. Klados is the paired evaluation dataset; native OSF tests transfer and preservation. It gives the best clean-target CC in the original frontal table, but native preservation fails.

### 5.2 Successive VMD plus fuzzy C-means

**Why selected:** it tests an EEG-only alternative to prescribing K and manually selecting an ocular mode. The paper, [EOG-free EEG using successive variational mode decomposition and fuzzy c-means clustering](https://doi.org/10.1016/j.bspc.2025.109434), uses successive extraction and feature-based clustering for ocular reduction. Accessible source auditing identifies synthetic, BCI Competition IV and SP300 BCI data; lateral-specific superiority is not established. Full ocular-paper definitions and optimized settings were unavailable, so our run is explicitly an adaptation.

**Mechanism:** extract a mode, account for it, then seek another mode while discouraging overlap with those already extracted. The residual after k modes is `r_k = x − sum(u_j)`. Features such as kurtosis, mean absolute amplitude, standard deviation and Higuchi fractal dimension summarize mode segments. FCM minimizes `sum_i sum_c membership_ic^m × squared_distance(feature_i, center_c)`, alternating between cluster-center and membership updates. Membership is similarity to a cluster, not an EOG probability. A declared semantic rule labels an artifact-like cluster. Our adaptation uses two clusters, fuzzifier 2, 200-sample windows/100 hop, threshold 0.8 and low-frequency eligibility up to 8 Hz. PySDKit SVMD uses demo-based maxAlpha1000/tau0/tolerance1e-6 with a visible maximum of 16 modes and iteration cap300. Sixteen solver-cap warnings remain in the comparison; no claim of verified inner convergence is made.

### 5.3 VME–GMETV

**Why selected:** it addresses EEG retained inside an artifact-dominated component, instead of deleting that entire component. The paper [EOG Artifacts Suppression From single channel EEG Signals by VME-GMETV model](https://doi.org/10.1016/j.bspc.2023.105622) combines variational mode extraction with generalized Moreau-enhanced total variation. Its exact dataset and full optimized ocular settings were not verified from available material. We do not invent them or claim an exact reproduction.

**Mechanism:** VME extracts a targeted waveform `u = ocular + retained EEG`. A regularized estimate identifies the ocular contribution within it. Ordinary TV balances `0.5||u−a||² + lambda||Da||_1`, where D computes consecutive differences; GME modifies the difference penalty to reduce ordinary TV's amplitude bias while preserving overall convexity under suitable conditions. Our declared implementation follows [Selesnick et al.'s GME-TV formulation](https://doi.org/10.1007/s10851-019-00937-5), with VME alpha20000, center0Hz, tau0, tolerance1e-7, cap300; normalized GMETV lambda0.5 and gamma0.8 with an explicitly constrained matrix construction. These are a valid implemented family member, not verified ocular-paper optimum settings. Correction is `x − estimated ocular waveform`, retaining the rest of the mode and residual.

### 5.4 EEGOAR-Net

**Why selected:** it is a montage-oriented learned ocular correction using a classical teacher and has an available [author implementation](https://github.com/dmarcos97/EEGOAR-Net). The [paper](https://doi.org/10.1016/j.bspc.2025.108147) develops a calibration-free, montage-independent learned model using SGEYESUB-derived targets from native eye-artifact data. The teacher supplies operational corrected targets, not independently known clean EEG; teacher errors can be inherited.

**Our run:** author architecture and pretrained initialization, then Klados-only paired fine-tuning on an explicit seven-channel-plus-absent-column montage. The released 64-label list lacks FP1/FP2, so arbitrary frozen electrode aliases were rejected. We use the required 128Hz/128-sample representation and a resampling-only control, input RMS for both input and target, AdamW learning rate0.001, weight decay0.0001, batch8 and 15 epochs with validation-only checkpoint selection. The measured model has 595,882 parameters. Native transfer has two missing montage inputs. This is a fine-tuned adaptation, not unchanged pretrained reproduction.

### 5.5 GEVD–MWF

**Why selected:** it supplies a compact multichannel covariance-based comparator, distinct from per-electrode VMD. [Somers, Francart and Bertrand (2018)](https://doi.org/10.1088/1741-2552/aaac92) propose a generic EEG artifact remover using a low-rank artifact covariance estimate derived through generalized eigenvalue decomposition. Its scope is multichannel artifact filtering rather than only frontal blink cleaning. The full original dataset composition is not reconstructed in this project report; the comparison dataset here is explicitly Klados, followed by OSF.

**Mechanism:** contrast artifact-rich and reference covariance matrices. The equation `R_art w = eigenvalue × R_ref w` finds weighted channel combinations with unusually large artifact-period variance. A Wiener estimator uses the resulting artifact covariance to estimate the unwanted signal with minimum expected squared error under the model. Our frontal run uses shrinkage-stabilized instantaneous rank2 filtering, strength1, and training EOG top/bottom30% activity as approximate masks. It is not the complete temporally embedded author pipeline. The central/posterior pilots use rank1. Reference covariance is not guaranteed clean; calibration brain-state differences can be removed inadvertently.

### 5.6 BandRouteNet / released BRAANetV2

**Why selected:** it tests learned frequency- and time-selective correction against explicit VMD features. [BandRouteNet v2](https://arxiv.org/html/2604.24428v2) uses band routing and full-band conditioning for EOG/EMG/mixed artifact reduction on EEGdenoiseNet: 4,515 EEG, 3,400 EOG and 5,598 EMG segments at 256Hz/512 samples. That benchmark is not equivalent to our regional Klados comparison, so paper headline scores are not placed in our ranking as if matched.

**Our run:** pinned author BRAANetV2 implementation, measured 205,123 parameters, 200Hz input and two512-sample blocks per1,024-sample window, paired input-scaled MSE, AdamW0.001/0.0001, batch8 and15epochs. Bands follow the implementation boundaries .5–4/4–8/8–12/12–30/30–80/80–100Hz; because our input is filtered to40Hz, upper bands contain little energy. This is a disclosed benchmark adaptation, not the proposed VMD regional hybrid. Source/licensing provenance is recorded rather than silently incorporating author code under a new licence.

### 5.7 REG-ICA and SGEYESUB extension

[REG-ICA](https://doi.org/10.1016/j.bspc.2011.02.001) applies extended Infomax ICA, identifies ocular-associated components and uses stabilized recursive least squares with EOG references to clean those components before reconstruction. Its paper tests artificially contaminated EEG and aims to retain cerebral content within ocular components. This differs from deleting an IC or directly regressing each electrode. [SGEYESUB](https://doi.org/10.1016/j.neuroimage.2020.117000) learns a sparse generalized eye-artifact subspace from labelled calibration, addressing corneo-retinal and eyelid effects in EEG/MEG. The associated OSF recordings provide native eye classes and resting references. A conceptual `cleaned = (I − ocular_operator) EEG` expresses its subtraction, but does not replace the full author algorithm.

The approved extension fits the spatial methods on the **full EEG montage** for better spatial information, then scores frontal outputs separately. Full-montage fitting is disclosed; it is not a region-only fit. Readiness passed40 contracts. Full version2 completed the paired Klados portion but failed a native p03 input-equality check involving one float32 ULP in one of8,000 values. Its partial outputs and failed notebook were retained. The repaired version3 (`frontal-spatial-full-002`, source `d87b2e43116c9a7d58dd3adf755ab8df6a0698e1`) subsequently completed: 42 contracts, 530 verified artifacts, three executed cells, eight figures, six tables and zero error outputs. The accepted results below supersede the failed run’s partial scores. REG-ICA freezes calibration ICA and resets SRLS per trial/window; SGEYESUB fits the original six-class calibration on 58 native EEG channels. Frontal-only scoring does not implement regional messaging. SGEYESUB has no Klados paired score because required six-class calibration labels are unavailable. No missing labels are fabricated.

### 5.8 Recent pretrained-only candidates

EEG-WNet and Diff-ADN were investigated as pretrained-only extensions. Neither had a compatible author-code/checkpoint pair verified within the inspected sources, so both are **not run—pretrained reproduction unavailable**. This does not prove weights do not exist anywhere. No substitute architectures, new training or invented results are included. The recovered closeout notebook passed36 contracts and contains two summary figures, not predictions from these candidates.

## 6. Frontal results and native validation

### Paired Klados: matched original comparison

Eight reused records, three windows each, seven mapped frontal channels. SNR is calculated from summed squared clean signal and reconstruction error within a window, then averaged within record and across records. Other metrics follow their saved record-balanced aggregation. Spectral RRMSE follows the implemented saved spectral representation; it must not be silently replaced by another paper's PSD/amplitude convention.

| Method | Output SNR dB | RMSE | Temporal RRMSE | Spectral RRMSE | Clean EEG CC |
|---|---:|---:|---:|---:|---:|
| Identity | −0.7934 | 11.3582 | 1.1429 | 4.3494 | 0.6985 |
| **Historical VMD–EOG** | **16.1346** | **1.7914** | **0.1639** | **0.2110** | **0.9846** |
| SVMD–FCM adaptation | 0.9529 | 9.1208 | 0.8961 | 1.9951 | 0.6986 |
| VME–GMETV adaptation | 3.8474 | 7.0831 | 0.6460 | 0.7839 | 0.7578 |
| EEGOAR-Net, fine-tuned | 9.0298 | 3.8953 | 0.3658 | 0.4525 | 0.9191 |
| GEVD–MWF | 6.2107 | 5.5100 | 0.5171 | 0.8736 | 0.8536 |
| BandRouteNet adaptation | 6.9388 | 4.8914 | 0.4628 | 0.6005 | 0.8827 |
| Original direct EOG ridge control | 15.2262 | 1.9537 | 0.1757 | 0.2250 | 0.9826 |

**Latest accepted extension:** calibration-frozen ridge scored17.488734dB, RMSE2.510300, CC0.952944; REG-ICA scored**5.665687dB**, RMSE**6.013513**, CC**0.849560**. These use calibration-based fitting and exact archived scoring windows. They are not interchangeable with the original score-window-fitted ridge control. The older failed run's REG-ICA5.597328dB is retained as failure history, not the accepted final result. Mean logarithmic SNR, averaged RMSE and CC can rank methods differently. Report all three rather than discarding a comparator that beats VMD on one metric. No corrective method qualifies.

### Native OSF: suppression versus preservation

| Method | Blink absolute VEOG CC after | Lateral absolute HEOG CC after | Mean resting relative RMS change |
|---|---:|---:|---:|
| Before / identity | 0.918516 | 0.358906 | approximately0% |
| Historical VMD–EOG | 0.328867 | 0.214818 | 63.63% |
| SVMD–FCM | 0.367851 | 0.137215 | 64.55% |
| VME–GMETV | 0.367017 | 0.045383 | 49.76% |
| EEGOAR-Net | 0.889118 | 0.377380 | 14.59% |
| GEVD–MWF | 0.559549 | 0.301306 | 61.43% |
| BandRouteNet | 0.375989 | 0.276730 | 29.13% |
| Direct EOG regression | 0.044528 | 0.063976 | 70.26% |
| SGEYESUB, accepted extension | 0.109069 | **0.033802** | 25.33% |
| REG-ICA, accepted extension | 0.337483 | 0.170647 | 41.25% |

All current corrective frontal methods fail native preservation. SGEYESUB has the lowest measured lateral association in the accepted table, but worst-participant resting modification is31.149812%, mean alpha error0.649467dB, beta error0.472962dB and relative covariance error0.084700. REG-ICA worst-participant resting modification is51.573949%. The exploratory SGEYESUB-minus-ridge lateral absolute-CC difference is−0.030174, four-participant bootstrap95%CI[−0.038909,−0.018779]; its blink difference is+0.064541,CI[0.043586,0.084723]. These suppression trade-offs do not prove clean neural recovery or external superiority. A63.63% input change is **not** a measured63.63% loss of brain activity, because resting input is not a known clean target. It is nevertheless too large to support the stated preservation requirement. Low EOG correlation alone cannot select the winner; genuine physiology, referencing and filtering can also contribute to associations. Klados lacks verified blink/lateral condition labels, so no invented Klados condition-specific ranking is reported.

### Metrics and gates

For paired reference s and estimate y: `MSE = mean((s−y)^2)`; `RMSE = sqrt(MSE)` per scored item; `RRMSE_t = ||s−y||/||s||`; `CC` is ordinary mean-centered Pearson correlation. Output SNR is `10 log10(sum(s^2)/sum((s−y)^2))`; SNR improvement is output minus input SNR. Averages of per-item RMSE need not equal the root of globally averaged MSE. Spectral definitions and weighting must follow saved configurations. Native EOG associations use unchanged references, retaining signed and absolute coefficients separately. Project engineering gates are separate from paper metrics: worst-record/participant mean clean modification≤1%, alpha/beta error≤0.5dB, relative covariance error≤0.02 and paired CC deterioration≤0.005 against the declared comparator. Native preservation is not passed merely because a paired gate passes.

## 7. Nonfrontal algorithms: central, temporal and posterior

The hypothesis is that **selective, conservative spatial correction may preserve nonfrontal EEG better than an unnecessarily strong per-electrode cleaner**. This is a hypothesis to test, not permission to assume all nonfrontal contamination is weak. Central, temporal and posterior outputs must be scored separately before proposing one shared correction policy. ICA is attractive because full-montage spatial mixtures may distinguish ocular fields from neural activity; it is not selected solely because electrodes are farther from the eyes. A rank-one MWF is lower-rank, not automatically safer, and a25% subtraction is a strength setting, not a different algorithm.

| Fixed central/temporal method | Central SNR dB | Central clean CC | Temporal SNR dB | Temporal clean CC |
|---|---:|---:|---:|---:|
| Identity | 10.5743 | 0.926343 | 12.3977 | 0.947137 |
| Ridge,25% correction | 12.6422 | 0.950379 | 13.9872 | 0.963254 |
| Ridge,100% correction | 18.3343 | 0.969361 | 16.7267 | 0.951142 |
| Rank1 MWF,25% correction | 10.4977 | 0.925898 | 12.2443 | 0.949004 |
| Rank1 MWF,100% correction | 9.0605 | 0.917523 | 9.8424 | 0.930023 |

All corrective candidates fail preservation. Central native mean resting change is3.55%/14.21% for25%/100% ridge and7.92%/31.70% for25%/100% MWF. Worst-record paired clean-input modification for full ridge reaches89.35% central and136.29% temporal. Record16/start3024 scores−1.07dB central and−4.11dB temporal after full ridge. These failures remain included; a high average does not erase them.

**Spatial challengers and what they mean:** ICA estimates independent source time-series and a mixing matrix `X=A S`; correction modifies selected components before back-projection. REG-ICA corrects EOG-related contributions inside those components. Wavelet ICA applies time-scale processing within selected components; selective wavelet ICA restricts correction to contaminated portions. [Selective wavelet ICA](https://doi.org/10.3390/brainsci9120355) is relevant because it explicitly addresses blinks and horizontal movements. SGEYESUB provides calibrated ocular-subspace subtraction. These are sensible candidates for matched full-montage fitting and region-specific scoring, not completed qualified central/posterior winners.

**Archived context experiment, not active evidence:** the retired LEMON–OSF study compared posterior-only ICA, common-frontal support, signed left/right/midline summaries, raw-frontal support and full-montage support, with posterior-only output correction. Recorded ICA mean scores were11.815/12.516/12.918/13.168/13.523dB respectively; direct posterior regression13.542dB and signed-summary MWF12.335dB. Its737recipes and320examples are historical construction-dependent results, not Klados posterior scores or native reconstruction SNR. They justify testing context, not claiming proven transfer. The active central and combined figures below replace any temptation to merge those datasets into one ranking.

**Candidate selection rule:** first require valid source/calibration contracts and preservation; then compare paired SNR/RRMSE/CC and native condition-specific suppression with uncertainty; finally compare runtime and failure rates. Among practically equivalent passing methods, choose the simpler one. If no correction passes, retain pass-through as the safety baseline rather than declare a winner from SNR alone. Full-montage fitting supplies spatial information but is not itself an explicit message from a frontal VMD estimate.

## 8. Actual frontal–posterior combination and next integration

The completed `frontal-posterior-combined-001` experiment uses the same24 archived windows and all19Klados rows. It combines seven frontal outputs with five posterior outputs; central3 and temporal4 remain unchanged. New ridge0.001 and instantaneous rank1MWF use the first2,000 separately processed calibration samples; historical frontal predictions and clean-input controls are verified saved parents. The run passed44 contracts and recovered84/84verified artifacts, three executed cells, eight figures and no error cells.

| Frontal correction | Posterior correction | Whole-cap SNR dB | Mean RMSE | Clean EEG CC |
|---|---|---:|---:|---:|
| Identity | Identity | 2.902146 | 5.600867 | 0.860895 |
| Identity | Calibrated ridge | 2.732457 | 5.771783 | 0.856711 |
| Identity | Rank1 MWF | 2.712865 | 5.871333 | 0.857546 |
| **Historical VMD–EOG** | **Identity** | **13.679530** | **2.076279** | **0.966309** |
| Historical VMD–EOG | Calibrated ridge | 13.065156 | 2.247195 | 0.962125 |
| Historical VMD–EOG | Rank1 MWF | 12.510022 | 2.346744 | 0.962960 |
| Calibrated ridge | Identity | 12.898252 | 2.341121 | 0.954630 |
| Calibrated ridge | Calibrated ridge | 12.903361 | 2.512038 | 0.950446 |
| Calibrated ridge | Rank1 MWF | 11.635197 | 2.611587 | 0.951281 |

**Measured decision:** among these combined arms, historical frontal VMD with posterior pass-through is best on whole-cap SNR and CC. Adding posterior ridge changes whole-cap SNR by−0.614374dB, record-bootstrap95%CI[−2.127571,0.557839]; adding MWF changes it by−1.169508dB,CI[−2.504891,−0.231968]. Eight unknown-participant records do not establish participant generalization. Posterior ridge's regional mean SNR18.163185dB exceeds identity16.294360dB, yet its mean posterior MSE31.696201 exceeds identity3.674021: poor cases/amplitudes and logarithmic averaging matter. **Regional dB values cannot be added or substituted for the reconstructed whole-cap score.**

Worst-record whole-cap clean modification is46.94% for VMD+ridge and25.25% for VMD+MWF. VMD+posterior-identity has zero scored clean modification but fails the declared worst-record CC-deterioration comparator gate (0.012449>0.005). All combined qualification rows are false. Native combined evaluation and a fresh whole-cap15dB target have not been established.

### How regional estimates are composed

For original EEG X and disjoint region masks M:

```text
frontal estimate    = frontal_VMD(original frontal EEG, available references)
posterior estimate  = posterior_method(original EEG, eligible calibration)
combined artifact   = M_frontal * frontal estimate + M_posterior * posterior estimate
cleaned EEG         = original EEG - combined artifact
```

This completed test is **disjoint composition, not explicit frontal-to-posterior message passing**. Proposed communication would supply separate blink/common and signed right-minus-left ocular summaries, estimated frontal artifact, event confidence and missing-context flags. Transfer coefficients must be fitted on unscored calibration and compared with no-message controls. Estimated artifact is not known pure EOG. The posterior branch must still estimate correction relative to original posterior EEG; subtracting a full spatial correction and then a second overlapping transfer would double-correct it. Central and temporal heads can use the same interface only after their own validation.

## 9. Planned EEG-only student–teacher development

The provisional shared student is **width32 TCN–BiGRU**, five dilated temporal blocks,32GRU units per direction, masked regional context and zero-initialized artifact heads. TCN convolutions capture patterns over several time scales; BiGRU adds past/future context, making the reference model offline. Its output is an artifact estimate and `cleaned = input − artifact`. Before accepting this complexity, compare a compact temporal CNN and TCN-only model under identical data/training budgets. This is the plan, not a completed qualified trained student.

Paired EEG-only learning should come first, using the input's normalization scale for both input and target. A teacher is used only if its accuracy and preservation qualify and cross-fitted teacher targets improve a paired-only baseline. Teacher fitting and selection must exclude target recipient/donor sources where those identities exist; Klados record grouping cannot be advertised as participant separation. Initial loss is paired MSE, with separately evaluated identity, spectral and covariance terms. An SNR-shortfall penalty does not guarantee15or20dB. The exported model should require EEG, masks and verified metadata only—no EOG or runtime VMD. Fresh CPU loading, whole-record overlap-add behavior, missing-channel handling,10–50channel scaling and export parity remain acceptance tests. Current teacher preservation failures mean a final teacher is not yet qualified.

## 10. Algorithm explanation reference

| Algorithm | Calculation | What it gives us | Main failure risk |
|---|---|---|---|
| Ridge regression | Fit regularized EOG-to-EEG coefficients | Reference-explained artifact estimate | Removes correlated neural activity; unstable transfer |
| VMD | Jointly fit K narrowband adaptive waveforms | Modes, centers, residual | Mixed neural/ocular content; K/alpha dependence |
| SVMD | Successively extract distinct concentrated modes | Mode sequence with stopping rule | Mode count/stopping errors; selection still required |
| FCM | Fit feature clusters with soft memberships | Similarity grouping, not physiological labels | Artifact/neural groups overlap |
| VME | Extract one targeted mode | Candidate artifact-dominated waveform | Neural activity shares the target region |
| GMETV | Penalized waveform estimation using enhanced TV | Structured contribution within the mode | Wrong morphology assumption or smoothing distortion |
| ICA | Estimate independent sources and spatial mixing | Component time-series and scalp contributions | Independence/mixing assumptions fail; mixed ICs |
| REG-ICA | EOG-referenced adaptive correction inside selected ICs | Partially cleaned components, then reconstructed EEG | Selection/reference leakage |
| SGEYESUB | Calibrated sparse ocular-subspace estimation | Multichannel ocular correction operator | Nonrepresentative or missing calibration |
| GEVD–MWF | Covariance contrast, low-rank artifact estimate, Wiener filtering | Spatial/optional temporal artifact estimate | Neural covariance changes mistaken for artifact |
| Selective wavelet ICA | Time-scale correction within selected IC intervals | Localized component correction | Threshold/interval mistakes; reconstruction distortion |
| EEGOAR-Net | Learn correction from teacher-derived examples | EEG-only prediction at inference | Teacher bias and montage/domain shift |
| BandRouteNet | Learned time-dependent band routing with full-band context | Frequency-selective neural correction | Benchmark/domain shift; no inherent preservation guarantee |
| TCN–BiGRU student | Shared temporal encoding, recurrent context, residual heads | Proposed compact EEG-only imitation/paired learner | Offline context; teacher error; unvalidated generalization |

ASR, CCA-based reference fields and rank-two reference models occurred in earlier challenger discussions/archives. They are not silently promoted to active frontal winners here. ASR is artifact-subspace reconstruction, not a synonym for SGEYESUB or GEVD–MWF; CCA maximizes association between two sets of variables, not independence as in ICA. VMD–SOBI and named gain/anchored variants remain historical experiments outside this active six-method selection, rather than extra validated paper methods.

## 11. Reproducibility, bottlenecks and remaining work

Source modules live in Git; Kaggle notebooks pin published commits and save configurations, identities, checkpoints, predictions, tables, modes and figures. Actual executed notebooks are retained under `notebooks/executed/`, including outputs, not merely source cells. Cached historical timing excludes VMD computation unless explicitly included; GPU neural and CPU decomposition timings are not a hardware-matched speed ranking. Fixed-K VMD cost grows with channels, windows, modes and solver iterations, typically with FFT-related per-iteration costs. Spatial methods additionally fit covariance/unmixing models whose cost grows with montage dimension. Actual10–50channel deployment measurements remain required; asymptotic reasoning is not a measured latency claim.

Known bottlenecks: stopped AutoVMD compute campaign; unavailable full ocular-paper settings; Klados label binding/participant uncertainty; short adapted neural training; native montage/domain shift; solver-cap warnings; a repaired native mode-reference diagnostic; failed spatial-run float-equality check; and repeated preservation failures. Original failed runs are preserved. The document does not repair them by changing thresholds or omitting bad examples.

Next scientific milestone: use the recovered full spatial evaluation to compare calibrated selective correction against preservation-aware pass-through, and test explicit frontal-context transfer only under matched controls. Freeze a teacher only after preservation and grouped evaluation support it. Then compare paired-first compact students. Reserved independent evaluation must not become another tuning set.

## 12. Source record and figure atlas

The attached atlas uses original Kaggle-produced PNGs, copied byte-for-byte. Its machine-readable manifest records each source path, SHA256, run, source commit and available retrieval receipt. The complete contemporaneous `RESULTS.md` is frozen beside this document so later repository changes do not rewrite this report's evidence snapshot. The obsolete native mode-correlation figure is excluded in favor of audit version2. Figures include paired accuracy, channel variation, waveforms, PSDs, training traces, mode evidence, native condition-specific EOG correlations, preservation and actual combined results.

The original experiment receipts document29contracts/922payloads for the frontal comparison,31/240 for native frontal,32/25 for the final mode audit,38/248 for central/temporal, and44/84 for the combined test, plus42contracts/530payloads for the accepted spatial extension. Contract counts establish implementation checks, not clinical validity or external generalization.

<!-- FIGURE_ATLAS -->

### Paired frontal comparison

Run `frontal-six-full-001`; source `c19637465e5a5c29991c2a988f25e6527a156f09`.

![Figure 1: 01 snr](assets/frontal-six-full-001__01_snr.png)

**Figure 1. 01 snr.** Reused paired Klados records; no fresh-confirmation claim. Original Kaggle output; no local replotting.

![Figure 2: 02 rmse](assets/frontal-six-full-001__02_rmse.png)

**Figure 2. 02 rmse.** Reused paired Klados records; no fresh-confirmation claim. Original Kaggle output; no local replotting.

![Figure 3: 03 rrmse](assets/frontal-six-full-001__03_rrmse.png)

**Figure 3. 03 rrmse.** Reused paired Klados records; no fresh-confirmation claim. Original Kaggle output; no local replotting.

![Figure 4: 04 pearson](assets/frontal-six-full-001__04_pearson.png)

**Figure 4. 04 pearson.** Reused paired Klados records; no fresh-confirmation claim. Original Kaggle output; no local replotting.

![Figure 5: 05 channel snr](assets/frontal-six-full-001__05_channel_snr.png)

**Figure 5. 05 channel snr.** Reused paired Klados records; no fresh-confirmation claim. Original Kaggle output; no local replotting.

![Figure 6: 06 preservation](assets/frontal-six-full-001__06_preservation.png)

**Figure 6. 06 preservation.** Reused paired Klados records; no fresh-confirmation claim. Original Kaggle output; no local replotting.

![Figure 7: 07 eog correlations](assets/frontal-six-full-001__07_eog_correlations.png)

**Figure 7. 07 eog correlations.** Reused paired Klados records; no fresh-confirmation claim. Original Kaggle output; no local replotting.

![Figure 8: 08 waveforms](assets/frontal-six-full-001__08_waveforms.png)

**Figure 8. 08 waveforms.** Reused paired Klados records; no fresh-confirmation claim. Original Kaggle output; no local replotting.

![Figure 9: 09 psd](assets/frontal-six-full-001__09_psd.png)

**Figure 9. 09 psd.** Reused paired Klados records; no fresh-confirmation claim. Original Kaggle output; no local replotting.

![Figure 10: 10 training](assets/frontal-six-full-001__10_training.png)

**Figure 10. 10 training.** Reused paired Klados records; no fresh-confirmation claim. Original Kaggle output; no local replotting.

![Figure 11: 11 mode reference evidence](assets/frontal-six-full-001__11_mode_reference_evidence.png)

**Figure 11. 11 mode reference evidence.** Reused paired Klados records; no fresh-confirmation claim. Original Kaggle output; no local replotting.

![Figure 12: 12 failures](assets/frontal-six-full-001__12_failures.png)

**Figure 12. 12 failures.** Reused paired Klados records; no fresh-confirmation claim. Original Kaggle output; no local replotting.

![Figure 13: 13 mode vectors](assets/frontal-six-full-001__13_mode_vectors.png)

**Figure 13. 13 mode vectors.** Reused paired Klados records; no fresh-confirmation claim. Original Kaggle output; no local replotting.

### Native frontal and reference correlations

Run `frontal-osf-correlations-001`; source `b9df0e65e0b7135aae45f9614439272bda88a8f3`.

![Figure 14: 01 klados clean cc](assets/frontal-osf-correlations-001__01_klados_clean_cc.png)

**Figure 14. 01 klados clean cc.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 15: 02 klados heog](assets/frontal-osf-correlations-001__02_klados_heog.png)

**Figure 15. 02 klados heog.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 16: 02 klados veog](assets/frontal-osf-correlations-001__02_klados_veog.png)

**Figure 16. 02 klados veog.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 17: 03 osf blink veog](assets/frontal-osf-correlations-001__03_osf_blink_veog.png)

**Figure 17. 03 osf blink veog.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 18: 03 osf lateral heog](assets/frontal-osf-correlations-001__03_osf_lateral_heog.png)

**Figure 18. 03 osf lateral heog.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 19: 03 osf vertical veog](assets/frontal-osf-correlations-001__03_osf_vertical_veog.png)

**Figure 19. 03 osf vertical veog.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 20: 04 rest change](assets/frontal-osf-correlations-001__04_rest_change.png)

**Figure 20. 04 rest change.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 21: 05 rest cc](assets/frontal-osf-correlations-001__05_rest_cc.png)

**Figure 21. 05 rest cc.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 22: 06 rest covariance](assets/frontal-osf-correlations-001__06_rest_covariance.png)

**Figure 22. 06 rest covariance.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 23: 07 rest spectral](assets/frontal-osf-correlations-001__07_rest_spectral.png)

**Figure 23. 07 rest spectral.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 24: 08 channel blink](assets/frontal-osf-correlations-001__08_channel_blink.png)

**Figure 24. 08 channel blink.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 25: 08 channel lateral](assets/frontal-osf-correlations-001__08_channel_lateral.png)

**Figure 25. 08 channel lateral.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 26: 09 waveforms blink](assets/frontal-osf-correlations-001__09_waveforms_blink.png)

**Figure 26. 09 waveforms blink.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 27: 09 waveforms lateral](assets/frontal-osf-correlations-001__09_waveforms_lateral.png)

**Figure 27. 09 waveforms lateral.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 28: 11 failures](assets/frontal-osf-correlations-001__11_failures.png)

**Figure 28. 11 failures.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

### Corrected mode/reference audit

Run `frontal-mode-clean-audit-002`; source `0876931804df1c4344b70f165f0fb907f3a28bc8`.

![Figure 29: 01 modes clean eeg and eog](assets/frontal-mode-clean-audit-002__01_modes_clean_eeg_and_eog.png)

**Figure 29. 01 modes clean eeg and eog.** Diagnostic associations do not establish that a mode is pure artifact. Original Kaggle output; no local replotting.

![Figure 30: 02 modes joint associations](assets/frontal-mode-clean-audit-002__02_modes_joint_associations.png)

**Figure 30. 02 modes joint associations.** Diagnostic associations do not establish that a mode is pure artifact. Original Kaggle output; no local replotting.

![Figure 31: 03 residual associations](assets/frontal-mode-clean-audit-002__03_residual_associations.png)

**Figure 31. 03 residual associations.** Diagnostic associations do not establish that a mode is pure artifact. Original Kaggle output; no local replotting.

### Central and temporal comparison

Run `central-temporal-001`; source `20584ab33028b43fada3912960e926c7948f70cc`.

![Figure 32: 01 paired pooled window snr db](assets/central-temporal-001__01_paired_pooled_window_snr_db.png)

**Figure 32. 01 paired pooled window snr db.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 33: 02 paired rmse](assets/central-temporal-001__02_paired_rmse.png)

**Figure 33. 02 paired rmse.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 34: 03 paired pearson cc](assets/central-temporal-001__03_paired_pearson_cc.png)

**Figure 34. 03 paired pearson cc.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 35: 04 native blink](assets/central-temporal-001__04_native_blink.png)

**Figure 35. 04 native blink.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 36: 05 native lateral](assets/central-temporal-001__05_native_lateral.png)

**Figure 36. 05 native lateral.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 37: 06 native vertical](assets/central-temporal-001__06_native_vertical.png)

**Figure 37. 06 native vertical.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 38: 07 preservation](assets/central-temporal-001__07_preservation.png)

**Figure 38. 07 preservation.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 39: 08 preservation](assets/central-temporal-001__08_preservation.png)

**Figure 39. 08 preservation.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

### Actual frontal/posterior composition

Run `frontal-posterior-combined-001`; source `f94436ab167d3019aacb4ceeadb64775b42d10c0`.

![Figure 40: alpha error db](assets/frontal-posterior-combined-001__alpha_error_db.png)

**Figure 40. alpha error db.** Whole-cap and regional aggregations are distinct; all combined qualification gates fail. Original Kaggle output; no local replotting.

![Figure 41: beta error db](assets/frontal-posterior-combined-001__beta_error_db.png)

**Figure 41. beta error db.** Whole-cap and regional aggregations are distinct; all combined qualification gates fail. Original Kaggle output; no local replotting.

![Figure 42: covariance relative error](assets/frontal-posterior-combined-001__covariance_relative_error.png)

**Figure 42. covariance relative error.** Whole-cap and regional aggregations are distinct; all combined qualification gates fail. Original Kaggle output; no local replotting.

![Figure 43: pearson cc](assets/frontal-posterior-combined-001__pearson_cc.png)

**Figure 43. pearson cc.** Whole-cap and regional aggregations are distinct; all combined qualification gates fail. Original Kaggle output; no local replotting.

![Figure 44: pooled window snr db](assets/frontal-posterior-combined-001__pooled_window_snr_db.png)

**Figure 44. pooled window snr db.** Whole-cap and regional aggregations are distinct; all combined qualification gates fail. Original Kaggle output; no local replotting.

![Figure 45: record snr](assets/frontal-posterior-combined-001__record_snr.png)

**Figure 45. record snr.** Whole-cap and regional aggregations are distinct; all combined qualification gates fail. Original Kaggle output; no local replotting.

![Figure 46: relative change](assets/frontal-posterior-combined-001__relative_change.png)

**Figure 46. relative change.** Whole-cap and regional aggregations are distinct; all combined qualification gates fail. Original Kaggle output; no local replotting.

![Figure 47: rmse](assets/frontal-posterior-combined-001__rmse.png)

**Figure 47. rmse.** Whole-cap and regional aggregations are distinct; all combined qualification gates fail. Original Kaggle output; no local replotting.

### Completed full-montage SGEYESUB and REG-ICA frontal extension

Run `frontal-spatial-full-002`; source `d87b2e43116c9a7d58dd3adf755ab8df6a0698e1`.

![Figure 48: klados pearson cc](assets/frontal-spatial-full-002__klados_pearson_cc.png)

**Figure 48. klados pearson cc.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 49: klados pooled window snr db](assets/frontal-spatial-full-002__klados_pooled_window_snr_db.png)

**Figure 49. klados pooled window snr db.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 50: klados rmse](assets/frontal-spatial-full-002__klados_rmse.png)

**Figure 50. klados rmse.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 51: osf blink](assets/frontal-spatial-full-002__osf_blink.png)

**Figure 51. osf blink.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 52: osf lateral](assets/frontal-spatial-full-002__osf_lateral.png)

**Figure 52. osf lateral.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 53: osf preservation](assets/frontal-spatial-full-002__osf_preservation.png)

**Figure 53. osf preservation.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.

![Figure 54: osf vertical](assets/frontal-spatial-full-002__osf_vertical.png)

**Figure 54. osf vertical.** Read axis labels for paired versus native scope; native EEG has no clean-target SNR. Original Kaggle output; no local replotting.
