# Region-aware EOG removal: research design and Kaggle execution contract

Updated: 2026-10-03. Status: research/design document; the proposed regional/streaming pipeline is **not implemented, trained or experimentally validated**. Local inventory and account observations below are dated 2026-10-02 unless stated otherwise.

## Decision in plain language

Test the user's two-method hypothesis: use a softly gated VMD/FCM expert primarily at the front, and a conservative ICA/ICA–VMD expert primarily at the back. ICA must estimate sources using the full available EEG montage, not just posterior electrodes. Treat ASR as an optional upstream burst-repair method and an independent comparator. Do not stack all methods unconditionally.

The proposed contribution is **blink-aware regional suppression–preservation control under an explicit streaming latency budget**, not merely combining VMD, FCM and ICA. Those combinations already exist. A regional approach is a hypothesis, not a proven anatomical rule. Compare it against a shared all-channel method under identical held-out evaluation; abandon the split if it does not help. Section 10 adds the real-time and blink research priorities without replacing the locked Klados/OSF validation.

All signal processing, model tests, training and evaluation are to run on Kaggle. The local computer only authors/version-controls source, inventories/packages files, invokes the official Kaggle CLI and retrieves outputs. No wavelets or state-space models are introduced.

## 1. Evidence survey: what the literature actually changes

The survey used publisher/PubMed pages, accessible primary manuscripts, official implementation documentation, alphaXiv and Consensus discovery. Some publisher full texts were inaccessible. Abstract-only findings below are explicitly restricted; no unavailable numerical table is reconstructed. Paper-resolution tools returned unrelated papers for two exact titles; those mismatches were discarded. This is a targeted survey, not a PRISMA systematic review or a confirmed Scopus-indexing audit.

| Reference | Evidence and limitation | Consequence for our system |
|---|---|---|
| [1] Sravanthi & Sharma, 2026, *Biomedical Signal Processing and Control*, SVMD–FCM | The publisher abstract describes successive VMD and FCM for blink removal. Full methods/results were inaccessible. | VMD-family + FCM is a direct competitor, not sufficient novelty. Keep standard VMD K=5/alpha=1000 as our comparable starting point; assess SVMD separately only with an inspectable implementation. |
| [2] He et al., 2026, *IEEE TBME*, GVICA | Fetched bibliographic abstract reports noise-grouped, GWO-optimized VMD plus ICA entropy denoising, evaluated on simulated and real EEG. Full protocol/code not verified. | VMD–ICA fusion is already published. Do not claim an exact replication or invent reported scores; add a reproducible simplified ICA–VMD comparator and label its differences. |
| [3] Srishyla et al., 2025, *Journal of Neuroscience Methods* | Comparison on 50 infants found a sensitivity/specificity trade-off: stronger ICA artifact correction versus less clean-segment distortion with Artifact Blocking. Infant-specific results cannot establish adult superiority. | Evaluate clean segments independently; artifact reduction alone cannot select our winner. |
| [4] Kim et al., 2025, *Journal of Neuroscience Methods*, Juggler's ASR | Explains reference selection, thresholding and reconstruction; tests extreme movement and rank-aware subsequent ICA. Juggling is not our ocular-only niche. | Audit calibration and rank. Evaluate conservative ASR and ASR→ICA, but do not assume motion-cleaning results transfer to EOG. |
| [5] You et al., 2025 early access / 2026 issue, *IEEE TETCI*, MASR | Multimodal features and channel significance inform adaptive ASR thresholds. Accessible author manuscript/metadata supports the method rationale. | Investigate burden-dependent intervention rather than one fixed correction strength everywhere. Auxiliary references may inform a teacher but must not secretly be required by an EEG-only student. |
| [6] Rico-Olarte et al., 2025, *MethodsX*, EEG-cleanse | ASR followed by ICA/component labeling forms a reproducible movement pipeline. Its proxies and brain-IC retention do not provide paired-clean neural ground truth. | Separate logging/reproducibility from proof of preservation. Do not equate retained brain labels or correlation with contaminated raw data to recovered clean EEG. |
| [7] Sattari et al., 2025 online / 2026 issue, *IEEE TBME*, contrast-based GED | Abstract reports generalized eigendecomposition, semi-simulated and walking/jogging evaluation; not a frontal-versus-posterior ocular comparison. | An EOG-informed contrast/subspace comparator is worth testing after the mandatory ICA/ASR baselines. It is not a mandatory extra neural branch. |
| [8] Shaikh et al., 2026, *Journal of Neural Engineering* | Multi-head contamination regression supports selective intervention. The published downstream denoiser uses wavelets. | Adapt the burden/gating principle only, not its prohibited wavelet stage or its benchmark numbers. |
| [9] Vishnu KN & Gupta, 2026-09-28, EMA-ASR preprint | Primary PDF explicitly reports stronger blink attenuation alongside larger reconstruction/spectral changes; evaluation against uncleaned input cannot isolate neural loss. | Do not automatically adopt the newest ASR variant. Put EMA-ASR in an exploratory row, not the locked primary method. |
| [10] Dora & Biswal, 2020, frontal VMD | Older, directly anatomical work estimates ocular artifacts from VMD modes then regresses the estimate rather than simply deleting low frequencies. | Frontal VMD is a defensible candidate, but not evidence that posterior ICA is always better. Prefer estimated residual subtraction to indiscriminate mode deletion. |

The literature does not establish the exact claim “VMD wins at the front and ICA wins at the back.” That is the question our regional ablation must answer. Spectral overlap makes low-frequency deletion unsafe. ICA components can mix neural and ocular activity; posterior visual responses time-locked to saccades are not automatically artifacts.

## 2. Verified local inputs and feasibility risks

The source bundle is `D:\Bunker\BrainComputerInterface\Lateral Eye Dataset\complete_dataset`: **375 files, 3,753,247,751 bytes**, verified by inventory on 2026-10-02, not re-inventoried on this update. Upload the whole folder privately, not just Klados. Bundling is not consent to publicly relicense all constituent files.

The user's repository is [saltypal/Lateral-eye-Artifact-removal](https://github.com/saltypal/Lateral-eye-Artifact-removal). Before this work, local and remote `main` both pointed to `11a9f7b95a1b204df8024619a9e95e193e4bc201`. The existing D: worktree contains unrelated deletions and untracked work; this design is being developed in an isolated clone, without resetting that worktree. Retain the reusable root package `eog_vmd_fcm_bgru`, avoiding unrelated legacy `src` imports.

| Input | Current evidence | Treatment |
|---|---|---|
| Klados four NPY arrays | Contaminated, pure, HEOG, VEOG are present. Local folder has no electrode/subject provenance file. Earlier array audit reported 53 records and 5401 samples, whereas published dataset descriptions describe 54 recordings. | Read actual headers on Kaggle; explain extraction/trimming/missing-record differences. Never assign standard channel order or pair record indices into subjects by guesswork. |
| `Dataset1_OSF` | 32 `*_prep.set` files: study01=5, study02=15, study03=10, study04=2. File discovery alone is not a usable-session count. | Read all four folders, verify embedded/external data, EOG, labels, subjects and channel positions. Report missing study04 files and exclusions. |
| `complete_eeg` derivatives | 59 metadata files; sampling rates 100 Hz (15), 200 Hz (30), 256 Hz (14). A study05 example explicitly lists 24 zero-padded channels and removed EOG/label channels. | Do not treat derived EEG-only files as independent sessions or paired clean targets. Padded electrodes are invalid observations. Use metadata for provenance, not automatic inclusion. |
| `EyeTrack` | Local description identifies an EEGEyeNet/OpenNeuro example, with only one participant visible in the current tree. | Upload it as requested; keep it outside the primary Klados/OSF protocol until provenance, completeness and labels are audited. Do not expand the cohort by counting duplicated derivatives. |

The earlier “45 sessions/39 participants” statement is not a substitute for inspecting authoritative upstream/session metadata. The upstream collection includes multiple studies; this local original-format subset has four. The Kaggle audit must write actual raw, derived, usable, excluded and unique-participant counts separately.

Published dataset links remain: [Klados data](https://data.mendeley.com/datasets/wb6yvr725d/1), [Klados paper](https://doi.org/10.1016/j.dib.2016.06.032), [OSF collection](https://doi.org/10.17605/OSF.IO/2QGRD), [OSF methodology](https://pubmed.ncbi.nlm.nih.gov/32497788/). No invented newer replacement dataset is introduced.

## 3. Signal and anatomical contracts

Use the existing evaluation band 0.5–40 Hz and resample to 200 Hz with anti-aliasing. For 100 Hz input, upsampling adds no original frequency information; log the native sampling rate. Preserve the existing reference unless a separately documented common-reference sensitivity analysis is required. Never re-reference input alone while keeping the target unchanged. For discontinuous epochs, filter/resample each continuous trial separately, not across concatenation boundaries.

Represent each example as `EEG[C,T]`, a valid-channel mask, electrode names/positions when verified, native sampling information, study/session/participant ID and optional event annotations. EOG and annotation channels are never mixed into EEG or classifier inputs accidentally. Units and filters are part of the saved inference contract.

Anatomical groups use verified coordinates or exact names: anterior/frontal, posterior/parietal-occipital, and central/temporal/unknown. Avoid `startswith('F')` alone because FC/FT are transitional. Record a versioned electrode dictionary and ensure each valid channel has exactly one group. Unknown locations use a region-agnostic fallback; they are not guessed to be frontal or posterior.

**Region-specific Klados acceptance is blocked until its channel mapping is recovered and verified.** Paired reconstruction and region-agnostic training can proceed, but anatomical claims cannot. The model must tolerate missing coordinates and masked/padded channels. Positions travel with their channels under permutations.

## 4. The two experts and ASR's precise role

### Expert A: anterior temporal-mode correction

1. VMD, starting at K=5 and alpha=1000; fit no dataset-level parameter on test records.
2. Retain reconstruction residual and original time alignment. VMD reconstruction need not be exact: do not lose the residual by summing “good” modes and pretending it equals the input.
3. Six mode descriptors: center frequency, low-band fraction, energy fraction, kurtosis, entropy and temporal concentration. Standardizer and two-cluster FCM are fitted on training records only.
4. Resolve cluster semantics using training-only ocular evidence. A cluster number has no inherent EOG meaning.
5. Estimate a time-local ocular residual with soft memberships and a gate. Do not remove an entire low-frequency mode everywhere in a record. Clean regions have an explicit identity constraint.

FCM is a probabilistic-looking membership heuristic, not a calibrated artifact probability. Membership sum and entropy are useful features; only the separate calibrated gate is evaluated as a probability.

### Expert B: posterior-preserving source correction

1. Fit rank-aware ICA on **all valid EEG channels** from the recording's declared unlabeled calibration portion. Start with Picard and compare extended Infomax for reproducibility.
2. Fit using a 1 Hz high-pass copy; apply the same spatial transform to the original evaluation-band signal. This follows [MNE's documented ICA contract](https://mne.tools/stable/generated/mne.preprocessing.ICA.html). Preserve reference, channel order and scale between fit/apply.
3. Identify candidate ocular sources using the named EOG channels, event timing, source morphology and verified topography. Determine thresholds only on development data. Provide a separate EEG-only identification comparator so EOG-assisted methods do not unfairly compete with an EEG-only deployment model.
4. Compare standard ocular-component rejection with **ICA-source VMD**: decompose only candidate ocular ICs and attenuate their event-local ocular portion before back-projection. This is an experimental extension, not a known guaranteed improvement.
5. Apply a conservative correction budget to posterior outputs; protect non-event alpha/beta and covariance. No deletion of all non-brain-labeled components.

ICLabel is an optional second opinion, not our sole selector. Its documented training contract is average-referenced, 1–100 Hz, extended Infomax data. Our 100/200 Hz and 40 Hz-limited data cannot fully satisfy that contract; upsampling cannot restore missing spectral content. Record this domain mismatch and validate probabilities before trusting them. [ICLabel documentation](https://mne.tools/mne-icalabel/dev/api/iclabel.html).

### ASR: independent baseline and optional pre-ICA repair

ASR detects unusually high variance relative to calibration; it does not know that a component is ocular. Test ASR alone and ASR→ICA with cutoff candidates 10/20/40 selected by preservation-constrained validation. These are engineering candidates, not universal optimal values. No automatic aggressive cutoff borrowed from an unrelated motion study.

Use an explicit sufficiently long, unlabeled calibration segment selected by a frozen input-only rule. Exclude that segment from calibration-excluded evaluation. If it is too short, contaminated or rank-deficient, mark ASR unavailable rather than using a held-out clean Klados target or secretly pooling other test subjects. Classical per-recording calibration is an operational requirement, reported separately from a fixed, calibration-free neural zero-shot method.

The new official [mne-denoise ASR documentation](https://mne.tools/mne-denoise/stable/asr.html) exposes standard and Juggler variants; its 0.0.3 release is recent. Prefer its inspectable pinned implementation as a candidate, but require shape/state/boundary regression checks before adopting it. Do not claim equivalence to EEGLAB without a numeric fixture comparison. ASRpy is an alternative whose current issue tracker includes shape bugs and an under-test Juggler fork, so “installed successfully” is insufficient validation.

### Fusion without double-cleaning

Both experts estimate a residual relative to the **same** input `x`; do not subtract sequential residuals derived from different inputs. For channel c and time t:

`y[c,t] = x[c,t] - g[c,t] * (w[c,t] * a_vmd[c,t] + (1-w[c,t]) * a_ica[c,t])`

`g,w` are bounded in [0,1]. Fixed regional weights are the first interpretable candidate; a learned router is a separate ablation. Unknown/central channels use a validated shared rule. Convex weights prevent coefficient double counting, but do **not** mathematically guarantee no neural damage: both artifact estimates can be wrong.

If ASR is enabled, both experts receive the same ASR output, and the total residual includes the ASR change. Report that change separately. Pure regional fusion is the default primary hypothesis; ASR is retained only if held-out preservation supports it.

## 5. Efficient deployable neural model

The expensive experts are training-time teachers/comparators, not compulsory inference dependencies. The deployable model accepts raw evaluation-band EEG, valid-channel masks and verified/unknown channel metadata. HEOG/VEOG are not required for its primary EEG-only interface.

Use a shared per-channel depthwise-separable encoder, two stride-2 stages, a permutation-equivariant masked spatial summary, one 48-unit-per-direction BiGRU and a 32-feature projection. Add two lightweight residual heads (anterior-mode-like and source-correction-like), a regional router, time gate, burden regression and optional artifact-type head. Trainable-parameter budget is <750,000, to be measured rather than asserted in advance.

Retain the **bounded-window offline/near-online BiGRU** as the quality reference and distillation teacher, preserving the user's preferred architecture. It is not causal: BiGRU, zero-phase filtering and window collection require future samples. In response to the 2026-10-03 real-time request, add a separately trained **causal or fixed-lookahead GRU student** as the proposed live deployment path (Section 10). This is an explicit architectural extension, not a claim that switching a trained BiGRU to one direction preserves its accuracy. Benchmark acquisition, filtering, buffering, transfers and compute, not GPU forward time alone. Quality and live latency are separate acceptance gates.

Train against paired Klados clean targets first. Teacher imitation is optional and subordinate to true paired supervision; reject teacher corrections that fail training/validation preservation. Never use held-out clean EEG to select teacher outputs or make pseudo-targets. OSF pseudo-label agreement is not clean-target truth; begin adaptation with gate/normalization heads only and genuine ocular annotations.

Losses: one waveform Huber term; multiresolution spectral and band-power losses; clean-window identity; cross-channel covariance; calibrated gate classification; burden regression; optional EOG specificity and genuinely annotated type supervision. Do not duplicate mathematically identical artifact/clean reconstruction losses. Ignore intermediate burdens for binary gate training: positive >−10 dB, clean ≤−20 dB. They still participate in reconstruction/regression. Klados HEOG-versus-VEOG amplitude alone cannot label a blink separately from vertical saccades.

## 6. Locked validation: same datasets and metrics, corrected interpretation

### Klados

Split records **before** windows, normalization, FCM fitting or teacher tuning. Recover subject identity if possible; otherwise keep record-held-out reporting and state that subject leakage cannot be ruled out. Hash-identical duplicates cannot cross partitions. Preserve the old 70/15/15 engineering split for comparison; the full protocol uses five grouped outer folds and three seeds, with validation confined to outer training records.

Historical values RMSE 2.848165, Pearson 0.952969 and SNR 12.278693 dB are descriptive references from the previous workflow. They are not paired significance evidence and may involve a validation-selected evaluation set. Its ~0.2753 value was normalized error; verify the exact denominator before treating it as temporal RRMSE. Retrain the old VMD–TCN–BiGRU on the new locked splits to make primary claims.

### OSF

Protocol Z: train/tune only on Klados, freeze weights/normalization/thresholds, evaluate every audited OSF original session once. Do not use aggregate OSF results to retune this version and still call it strict zero-shot.

Protocol A: hold out one of four available original studies; adapt gate/normalization on the other three; rotate. **Purge every training/validation session of participants appearing in the held-out study**, even if that session belongs to another study. Use participant-grouped development validation within the remaining studies. The local study04 subset has only two files: report low precision/coverage instead of promising well-powered cross-study significance.

Classical ICA/ASR session calibration is documented as input-only recording adaptation, not a calibration-free comparison. Report matched calibration-excluded segments for fairness and a separate full-session fixed-student result. Gaps/trials are not concatenated into fake continuous recordings.

### Metrics retained and added

| Family | Calculation and legitimate reference |
|---|---|
| Paired waveform | Pearson, Spearman, RMSE, MAE, NRMSE with its explicit historical denominator; temporal RRMSE=`||y-s||₂ / ||s||₂`; SNR=`10 log10(sum(s²)/sum((y-s)²))`; ΔSNR=cleaned SNR−contaminated SNR. Unit, demeaning and macro aggregation must match baselines. |
| Spectrum | Spectral RRMSE on **Welch linear PSD** using one locked FFT/window configuration; define it independently of time-domain RRMSE. Absolute delta/theta/alpha/beta log-power error in dB versus paired clean. Bands .5–4/4–8/8–13/13–30 Hz with non-overlapping bins. |
| Preservation | Clean-window RMS change, relative change, target error and fraction materially modified. Count actual `y-x` above locked numerical/relative amplitude tolerance, not `gate>0.1`. OSF non-event metrics describe stability relative to raw, not recovered clean accuracy. |
| Ocular suppression | HEOG/VEOG absolute-correlation reduction, EOG-band and lateral attenuation; blink/saccade event averages, fixed event windows and non-event preservation. No fabricated sample events from trial condition labels. |
| Detection | Gate AUPRC, prevalence/no-skill PR line, ECE using predeclared bins and valid binary labels. Window burden MAE/R² only against legitimate paired burden; test labels never set gate thresholds. Report missing/single-class outcomes explicitly. |
| Spatial | Paired covariance relative Frobenius error using the same valid channels/reference. OSF scalp maps and event-related ocular-loading change with measured positions. Without a clean spatial reference, call it **topographic change**, not neural topography error. |
| Legacy proxies | PSD distortion, energy retained, preservation correlation/RMSE, with reference and event status explicitly shown. Suppressed contaminated energy is not proof of improvement. |
| Utility/runtime | ERP/MI/BCI only with genuine non-ocular task labels and a separately isolated decoder. Otherwise N/A with reason. Parameters, serialized bytes, peak memory, batch-1 median/p95 latency, total preprocessing latency, real-time factor. |

Define clean and event windows without using a method's own output to decide which windows count. Align input, target, EOG and events consistently for any VMD 5400/5401 mismatch. Zero target energy yields an unavailable relative metric, not an artificial large “success” score. Never claim OSF paired SNR or RMSE without paired targets.

### Statistics and victory

Aggregate per channel→record/session; repeated seeds are not extra independent participants. Average repeated out-of-fold estimates within original record for primary inference. Bootstrap independent participants when known, otherwise records; report study-specific CIs and macro averages. With only four studies, study-level inference is weak. Use paired record/session Wilcoxon tests for declared superiority endpoints and correct the predeclared family with monotone Holm adjustment.

Preservation requires **non-inferiority bounds**, not p>0.05. Proposed development-only margins to lock before any test run: ≤0.5 dB additional alpha/beta log-power error, ≤0.02 additional normalized covariance error, ≤1% clean-window relative RMS modification, and Pearson deterioration ≤0.005. These are engineering proposals requiring sensitivity analysis, not accepted clinical thresholds. Use one-sided confidence bounds on paired differences and multiplicity-aware decision rules.

The full Kaggle profile tests ≥10% paired RMSE/temporal-RRMSE reduction and ≥1 dB SNR gain versus the **rerun** old model, non-inferior preservation and held-out adapted OSF benefit. Historical OSF reductions 0.0550/0.0210 remain descriptive targets. Require valid pairing, provenance and confidence bounds. A failure saves all evidence and does not rewrite the DOCX or claim superiority.

## 7. Comparisons that answer the user's question

Mandatory: raw/no-op; EOG ridge/regression (reference-assisted label); original hard/soft VMD–FCM; standard ICA; ICA-source VMD; ASR; ASR→ICA; shared all-channel VMD; shared all-channel ICA; fixed frontal-VMD/posterior-ICA fusion; learned fusion; old VMD–TCN–BiGRU; deployable EEG-only student.

Ablations: no VMD teacher, no ICA teacher, no ASR, no regional router, no gate, no EOG supervision, no spatial summary, BiGRU versus unidirectional GRU, and learned versus verified-name/position regional priors. Match channel availability, preprocessing, splits, supervision and tuning budget. Distinguish EOG-assisted and EEG-only columns.

Optional exploratory competitors: author-faithful SVMD–FCM/GVICA when protocols and implementations can be inspected; GED contrast correction; Juggler/EMA-ASR. Do not label a loosely reimplemented method “the published result.” These papers' different datasets/units/protocols cannot establish our numerical rank or Scopus-wide superiority.

## 8. Kaggle-only execution and incremental gates

| Phase | Inputs → outputs | Owner / acceptance |
|---|---|---|
| A. Upload + remote audit | Entire local tree → verified private Kaggle dataset, hash manifest, usable-session/channel/label inventory | Codex plumbing; account must authenticate; remote hashes must match every uploaded source file. No signal processing locally. |
| B. Classical benchmark | Frozen manifest + grouped splits → teacher outputs, paired/proxy metrics, event/preservation figures | Codex implementation with user review of algorithm logic; identity/alignment/rank tests and held-out preservation must pass. |
| C. Regional feasibility | Same splits + electrode provenance → fixed-region versus shared-method comparison | Joint design checkpoint; regional method must show incremental value. Unknown Klados mapping blocks anatomical claims. |
| D. Neural training | Safe teachers + paired targets → best/last weights and calibration | Kaggle GPU only; forward/gradient/mask/permutation/zero-gate tests, train-only fitting and resume equivalence. |
| E. Transfer + adaptation | Frozen Klados models + OSF participant/study manifests → strict zero-shot and isolated adapted outputs | Kaggle GPU/CPU; no participant overlap; every usable session counted and failures logged. |
| F. Deployment + decision | Out-of-fold evidence → deployment bundle, latency/QA, victory JSON | Fresh instance reload equality, preprocessing contract and CPU/GPU inference verified **on Kaggle**, no unsupported biological guarantee. |

One planned orchestrating notebook will be extended through these phases; core logic stays in source modules. Its first new regional-system milestone is **Phase A audit only**. No new regional audit notebook or controller has been delivered alongside this document; the pre-existing notebook is not evidence that the proposed pipeline is implemented.

Future modules under `eog_vmd_fcm_bgru`: `data_audit.py`, `datasets.py`, `regions.py`, `splits.py`, `ica_expert.py`, `asr_expert.py`, `regional_fusion.py`, `student.py`, `losses.py`, `evaluation.py`, `statistics.py`, `figures.py`, `deployment.py`, `experiment.py`. Add modules incrementally after the preceding gate, retaining existing VMD/FCM helpers.

Profiles become `kaggle_smoke` and `kaggle_full`, not `local_overnight`. First audit uses CPU. Benchmark wall time on Kaggle before scheduling all folds/seeds; do not promise a 3-hour full study. Shard the 5×3 full study by fold/seed when necessary, keeping one notebook interface and identical hashes/configuration. A single GPU is the initial tested path; do not assume two-T4 data parallelism or TPU compatibility.

Notebook setup makes a new temporary clone from the user's chosen branch or exact SHA, rather than deleting a user worktree. Freeze that SHA for a run/campaign. Install pinned dependencies, record hardware/package versions, write outputs to `/kaggle/working/results/<run_id>` and save best/last checkpoints atomically. Saved kernel-version outputs are the persistence unit; `/kaggle/working` alone does not survive a reset. Retrieve via CLI after completed runs, and attach completed outputs as a versioned private artifact dataset for resume. An abruptly aborted run cannot be promised durable checkpoints without an approved external store.

Required outputs: manifests/config/environment/SHA; per-record/per-session and per-region metrics; exclusions/N/A reasons; saved predictions; checkpoints and deployment bundle; training history; statistical comparisons/CI; diagnostic figures; victory decision. Bundle includes units, fs, filtering, reference, valid-channel and coordinate handling, normalization, architecture, weights and gate calibration. A teacher-assisted bundle also stores VMD/FCM/ICA/ASR contracts; the fast EEG-only bundle explicitly declares no such runtime dependencies.

Simple plots: shaded blink/saccade waveforms, removed artifact trace, paired before/after bars, clean-window no-harm page and front/back suppression-versus-preservation summary. Technical plots: VMD/FCM descriptors, IC spectra/topography and rank, ASR calibration/cutoff response, gate PR/calibration, burden scatter, band/covariance errors, regional Pareto frontier, study CIs, worst cases, ablations and latency. No unexplained composite “quality score” can hide clean-signal damage.

## 9. Current blockers and next authorized action

Last checked 2026-10-02: the installed official CLI was 2.2.4. The saved modern token was rejected by an account-scoped command; `datasets list --mine` returning empty was **not** proof of authentication. The requested GPT-6 Luna agent could not control Edge: no computer-control tool and no existing browser debugging endpoint. It stopped the waiting OAuth attempt without changing browser/account settings. Authentication has not been rechecked for this research-only update; uploads and training remain on hold.

When the user is present:

```powershell
kaggle auth login --force
kaggle kernels list --mine --page-size 1
```

After an explicit execution resume, implement the controller to prepare/upload the full folder privately, verify dataset status/files, submit the CPU audit notebook, check status and retrieve outputs. Until authentication succeeds, no upload, remote audit, expert benchmark, training, final model or superiority result is complete.

The original DOCX remains untouched. Its conditional rewrite remains subject to actual full-profile victory and document render/visual QA; no report of “amazing results” is written from this design.

## 10. Real-time and blink research addendum — 2026-10-03

### 10.1 New primary evidence and the novelty boundary

This addendum used alphaXiv/Consensus discovery followed by primary papers, author implementations and official documentation. Findings describe published methods, not results achieved by our system. Versions matter: alphaXiv returned nASR v1, whereas the verified August v2 changes the cohort, split and reported speedup. Use the versioned v2 reference below; do not mix its numbers with v1.

| New reference | What is already established or proposed | Design consequence and limitation |
|---|---|---|
| [11] ARMBR, JNE 2025 | Blink-reference backward regression estimates a lightweight scalp projection. It compares reconstruction, real non-blink preservation and ERP preservation with regression, SSP, ICA/ICLabel and ASR. | Add an explicit blink-specific spatial baseline. A trained projection can be cheap to apply; inspect calibration/detection before calling a reproduced implementation causal. Its results are not evidence on our Klados/OSF splits. |
| [12] EEGOAR-Net, BSPC 2025 | Uses SGEYESUB-corrected training targets and channel masking for calibration-free, montage-flexible ocular reduction; includes blink validation. The author repository now provides PyTorch code/weights as well as TensorFlow. | Teacher distillation, EEG-only inference and montage masking alone are not our novelty. Add a recent neural comparator with matched training data. The inspected code uses temporal SAME padding and default one-second inputs: fast inference alone does not establish sample-causal operation. |
| [13] Gated eyewear pipeline, Physiological Measurement 2026 | A lightweight detector selectively invokes artifact classification and U-Net correction; blinks and horizontal movements are represented. Accessible abstract describes participant-held-out testing. | Selective computation and a blink class already exist. Our hypothesis must test event-boundary coverage and preservation at a declared deadline, rather than claim the gate itself is new. Full runtime protocol remains unverified. |
| [14] AJDC neurofeedback, BIOSIGNALS 2025 | A calibrated frequency-domain BSS method is evaluated for blink attenuation and MI beta preservation in a pseudo-online setting. Its initial calibration becomes less effective over time. | Test a frozen spatial-transform alternative to ICA where reproducible. Online application of a fixed ICA/AJDC transform is different from repeatedly fitting it. Drift motivates monitoring, not automatic continual refitting. |
| [15] nASR v2, 2026 conference-accepted manuscript | Learns artifact/channel thresholds and selective spatial reconstruction with a downstream MI decoder. The inspected protocol uses 2.56-second windows, zero-phase preprocessing and reference statistics drawn across windows. | Learnable ASR thresholds and selective channel reconstruction are not new. Do not copy whole-record reference fitting or equate its reported inference acceleration with our end-to-end live deadline. MI classification is not paired blink-reconstruction evidence. |
| [16] BandRouteNet v2, 2026 preprint | Frequency-specific routing and a full-band conditioner yield a small denoiser on EEGDenoiseNet. Its band decomposition uses a segment-wide DFT. | Frequency routing and parameter efficiency alone are not novelty. Segment-wide spectral features are not automatically causal. Optional matched-data comparator, not a reason to add a third large branch. |
| [17] iPSD v2, September 2026 preprint | Learns noisy-signal partitions for self-supervised denoising, relying on independent noise and a shared underlying signal. Demonstrations include EMG and wearable EEG rather than our blink protocol. | Do not introduce unverified Noise2Noise-style OSF training: spatially coherent blinks can survive both partitions. Per-test-segment optimization is adaptation, not our frozen Klados-only zero-shot protocol. Defer this extension. |
| [18] CFo-CLEAN, EMBC 2025 | EEG-only adaptive ocular correction is described using 60/90-second estimation windows on driving recordings. Accessible institutional abstract only. | Calibration-free online ocular correction already exists. Long historical context need not equal future lookahead; the abstract alone does not establish its cold-start or sample-output latency. Include only if its implementation/protocol is inspectable. |

**Working contribution, not a confirmed first-in-literature claim:** a latency-bounded, blink-phase-aware student distilled from regional ocular experts, selected by joint suppression/preservation constraints and audited with continuous event-boundary replay. A broader prior-art search and successful matched ablations are still required before claiming scientific novelty. The audit itself can be valuable even if the anatomical split loses.

### 10.2 Ranked improvements and falsifiable experiments

Prioritize the first three as one coherent method. The remaining rows are independent efficiency/robustness ablations, not mandatory ingredients.

| Priority | Proposed improvement | Why it may help | Comparison that could disprove it |
|---|---|---|---|
| P1 | **Latency-conditioned regional distillation**: the BiGRU and VMD/FCM/full-montage ICA experts teach a small causal or fixed-lookahead GRU | Keeps costly decomposition out of the live path; makes the price of future context explicit | Same student trained only on paired clean targets; shared versus regional teachers; 0/40/80 ms lookahead. Reject distillation if it worsens paired error or preservation. |
| P1 | **Blink onset–peak–recovery compute control** with hysteresis and a bounded ring buffer | Prevents a window-average gate from missing the rising edge or turning off during the tail; saves refinement compute in verified clean periods | Always-on refinement versus ordinary burden gating versus phase-aware gating, at matched event recall and latency. Reject if saved average time causes missed blinks or burst deadline failures. |
| P1 | **Regional no-harm constraints plus teacher-disagreement weighting** | Prevents stronger frontal suppression from unnecessarily reducing posterior alpha/beta or reproducing unsafe teacher corrections | Shared unconstrained model, regional model alone, constraints alone, and the combination. Compare actual paired errors, not teacher agreement. Disagreement is a heuristic, not a calibrated uncertainty guarantee. |
| P2 | **Low-rank ocular residual decoder**, initially ranks 3 and 6 | A few learned scalp patterns may represent ocular spread more cheaply than independent full-size channel decoders | Dense residual head at matched width/quality. Reject if rank restriction leaks blink peaks or misrepresents asymmetric/mixed events. Low-rank ocular modeling is established, not independently novel. |
| P2 | **Channel-quality-aware routing** using verified masks and training-only dropout | Handles missing/noisy frontal electrodes and montage shifts without treating padded channels as observations | Remove frontal, EOG-adjacent and posterior channels separately; permute channels with their metadata. Report failure coverage, not just the average over easy montages. |
| P2 | **Backend-specific mixed precision/quantization and cached state** | Reduces memory traffic, repeated convolution work and allocation overhead | Batch-1 FP32 versus supported optimized kernels, including preprocessing and transfers. Retain only variants with non-inferior blink/preservation outcomes and better tail latency. |
| P3 | **Protected slow calibration updates** for an optional calibrated classical mode | May address ICA/AJDC/ASR drift while excluding confidently ocular periods | Frozen calibration versus clean-only bounded updates on chronological held-out sessions. High risk: a false clean decision can absorb blinks into the reference; never enable it in strict frozen zero-shot. |

No wavelets, state-space model, extra eye camera or unbounded test-time optimizer is required. ASR remains optional. A residual decoder with a small factorization is the first optimization candidate, not a reason to introduce a large attention stack.

### 10.3 Two explicit deployment contracts

**Quality/reference mode:** retain the current VMD/FCM and full-montage ICA research experts and the efficient BiGRU. This path establishes reconstruction quality using the unchanged offline evaluation. It may have seconds of context and must not be advertised as live merely because a GPU forward call is quick.

**Proposed streaming mode:** evaluation-band EEG → stateful causal preprocessing → shared cached temporal encoder → masked spatial summary → state-carrying unidirectional GRU → low-rank dual residual heads → calibrated ocular gate/regional router → cleaned chunk and quality flags. Primary inference needs EEG and metadata/mask only. VMD, FCM, ICA fitting and eigendecomposition are absent from this fast path; their training influence is stored in weights, not silently recomputed online.

Candidate dimensions: encoder widths 24/32; GRU hidden size 32/48; projection 24/32; residual ranks 3/6. Select on development data under the <750,000 parameter ceiling. Try a smaller <250,000-parameter deployment target, but do not assert that it is attainable at equal accuracy before measuring it. The rank-3 features are not automatically identifiable as blink/horizontal/vertical sources; semantic labels require supervision.

For valid channel c, retain the existing common-input subtraction contract:

`a_expert[c,t] = sum_q A_expert[c,q] * s_expert[q,t]`

`y[c,t] = x[c,t] - g[c,t] * (w[c,t]*a_anterior[c,t] + (1-w[c,t])*a_source[c,t])`

The small scalp coefficient network may use verified coordinates and channel summaries. Unknown positions invoke the shared fallback. Invalid channels are excluded from summaries/losses and returned with an invalid flag, not declared successfully cleaned. Gate zero must give exact identity relative to the **preprocessed** input; it does not undo bandpass filtering.

Make future-context budget an immutable deployment setting. A causal version has no future inputs; fixed-lookahead versions delay output by a declared bounded amount. Prefer separately trained/calibrated versions at first; a single latency-conditioned weight set is an experimental ablation, not a guarantee.

Initial engineering budgets, to lock before test access:

| Item | Proposed contract, not a measurement |
|---|---|
| Sampling/hop | 200 Hz, eight samples per hop = 40 ms. This aligns with the existing two stride-2 encoder stages; still carry convolution buffers and stride phase explicitly. |
| Lookahead | Compare 0, 8 and 16 samples = 0, 40 and 80 ms. Never retrospectively modify already emitted samples. |
| Compute | Aim for warm batch-1 p99 full-step compute <20 ms and measured deadline misses <0.1% under the selected hardware/load. All-blink/high-burden runs must satisfy the budget too. |
| Availability delay | Maximum nominal buffering + lookahead + proposed compute budget: 40 + 80 + 20 = 140 ms, **before** acquisition/transport/queue overhead. Target measured p99 total <150 ms only on a named configuration that actually passes. |
| Cold start | Emit a warm-up/unreliable flag until filter and state initialization are validated. Report warm-up length and worst startup behavior separately. |

The total sample-availability delay is not identical to filter phase delay. A causal 0.5–40 Hz filter can introduce frequency-dependent distortion even without waiting for future samples. Specify its response, state/reset rules and boundary handling; measure phase effects rather than pretending there is one universal group-delay constant. Anti-alias resampling must also have an explicit streaming state and delay.

**Keep two evaluation tracks, not two incompatible results in one table.** The original offline track retains its preprocessing, paired targets and all legacy metrics. The additional streaming track applies the same frozen causal preprocessing/reference/units to input and paired target, includes a filter-only no-op baseline, and scores at known timestamp alignment. Train streaming teachers/targets in that signal space; do not directly force zero-phase teacher waveforms onto phase-shifted causal targets. Do not optimize a time shift on held-out data to improve correlation. Causal-track improvements cannot be substituted for the historical offline victory threshold.

Cache recurrent state and causal convolution buffers; do not rerun overlapping history on every hop. Temporal normalization must use train-fitted constants or present-sample feature/channel statistics, never statistics over unseen future time. Forward, normalization, upsampling and buffering all have to satisfy the same causality contract. A learned spatial permutation-equivariant mixer is not automatically a safe anatomical model.

For an optional tiny classical deployment baseline, fit ICA/AJDC/ARMBR once on a declared calibration prefix and apply the frozen correction transform to subsequent samples. This can be cheap without neural distillation, but it is a separate calibrated operational mode. Do not run full ICA fitting per 40-ms chunk or imply that ICA inference itself must be slow.

### 10.4 Blink coverage is an explicit requirement

**Label contract.** First audit the original OSF annotation definitions and precision. Rest/blink trial labels identify conditions, not every sample onset; an epoch can contain both a blink and a saccade. Use genuine sample events when present, manually adjudicated annotations where authorized, or report trial-condition analysis. No invented timestamps. Missing annotations are unknown, not clean negatives. A VEOG peak alone is not a blink label.

Klados paired contamination supports waveform and continuous ocular-burden training, but does not establish blink-specific labels by itself. A blink head needs verified labels on Klados training records or separately documented weak supervision with expert checks. Without them, the frozen Klados-only model can be evaluated for removal on genuine OSF blink events, but its blink-versus-vertical-saccade classifier must be marked unsupported. OSF labels may supervise the adapted head only on non-held-out participants/studies; do not use them to tune strict zero-shot thresholds. External paper datasets/weights cannot be added to the Klados-only training protocol without relabeling the experiment.

Use an always-on ocular/burden head and, where supervision exists, **multi-label** blink/horizontal/vertical heads with an unknown mask. Independent labels allow overlaps; do not force mixed events into one exclusive class. Blink-state supervision (onset/rising, peak, recovery) is optional and requires onset/offset-quality labels, not just an estimated peak.

**Compute policy.** The cheap encoder, recurrent state and gate run on every chunk, including clean periods. Only an optional refinement head can be skipped. A lower entry threshold and higher-confidence clean exit rule plus hysteresis keep refinement active across the blink tail. The bounded buffer allows decisions about not-yet-emitted samples only; it cannot recover an onset that has already been sent downstream. Do not trigger cleaning only when a blink-type classifier agrees: the generic ocular gate must remain capable of removing unclassified ocular events.

Benchmark ordinary gating versus onset-sensitive gating with matched false-alarm/event-recall operating points. Training samples should include event interiors **and boundaries**, clean negatives, vertical saccades and overlapping events where available. Oversampling and boundary weighting use training data only. Train on the original pairs; optional augmentations must be explicitly labeled training augmentation, never a synthetic replacement for missing dataset files or test evidence.

Split blink evaluation into rising edge, peak and recovery using fixed annotation-derived intervals. Klados phase-wise waveform errors require genuine event annotation and paired clean targets. For OSF, event-aligned raw/clean peak-to-peak, EOG association and topographic change are suppression proxies—not proof that the remaining neural waveform is correct. Preserve non-event alpha/beta/covariance; event-locked neural activity can be real and must not be treated as artifact merely for being time-locked.

Required robustness slices: isolated/repeated blinks, spontaneous versus instructed when labeled, weak/large blinks, blink+saccade overlaps, events crossing every chunk boundary, missing frontal channels and abrupt session resets. Long closures and saturation are stress tests only where actually observed/annotated. Saturated acquisition can destroy information: return an unreliable flag rather than inventing guaranteed recovery. Preserve original EOG/event information as a separate output for multimodal use; removing its leakage from EEG must not erase the original eye signal or intentional blink commands in another modality.

Proposed development targets are >=95% annotated blink-event recall at <=1 clean false activation/minute, and p95 onset-decision latency <=200 ms, with annotation uncertainty and emission latency reported separately. These are configurable engineering goals, not published standards or achieved values. Lock definitions/thresholds on development data, then report per-session and participant-bootstrap bounds on test data. Insufficient event labels or insufficient precision means the blink-specific acceptance gate is unresolved, not passed.

### 10.5 Preservation, uncertainty and safe adaptation

Use paired reconstruction as the primary training anchor. Add teacher distillation only where training/validation evidence supports it. Down-weight residual targets when the experts disagree or a teacher violates clean-window/band/covariance constraints; report how many targets were excluded. Agreement can still mean both experts are wrong, especially in low frequencies. Do not use held-out clean targets to decide whether a teacher is trusted.

Express model selection as constrained optimization: minimize paired waveform error **subject to** the locked clean-window, alpha/beta and covariance non-inferiority requirements and the chosen runtime budget. A posterior penalty is not proof of posterior safety; require verified electrode mapping and regional held-out breakdowns. Protect frontal neural activity too. Never impose unconditional posterior identity, because blinks can contaminate posterior channels as well.

Save a separately calibrated quality flag for channel loss, clipping, unsupported montage and uncertain correction. Abstention can be conservative pass-through plus an unreliable flag; report both coverage and residual contamination on abstained intervals. It is not successfully cleaned data. Evaluate selective risk versus coverage and worst-participant failures; do not remove difficult segments from the headline denominator.

Strict zero-shot keeps normalization, weights, gate calibration and all thresholds frozen. Stateful filters/GRU memory are ordinary signal state, not test-time parameter fitting. Optional input-only session calibration, running-statistics adaptation and source-transform refitting are explicitly separate adapted modes with calibration-excluded scoring and participant-isolated tuning.

### 10.6 Extra measurements, plots and implementation gates

All existing metrics, grouped splits, zero-shot/adapted separation, seed accounting and victory requirements remain. Add these measurements; do not hide the trade-off in a composite score:

| Addition | Definition/interpretation |
|---|---|
| Blink-event recall/precision/F1 | One-to-one event matching using predeclared annotation tolerance/interval overlap; unmatched detections count as false positives. Freeze event-merging/refractory rules on development data. Trial-only labels do not permit event-level claims. |
| Onset/tail errors | Annotation-to-detection and annotation-to-output latency distributions; event-phase paired error where available; OSF event-phase suppression proxies otherwise. No detection-onset score when only peaks are labeled. |
| Clean false activations/minute | Measured on independently annotated clean/non-event intervals, with actual modification magnitude. Missing annotations cannot certify clean time. |
| Suppression/preservation/latency frontier | Plot each model and lookahead setting by paired error, clean harm and measured p99 latency, with participant/record uncertainty. |
| Streaming reliability | p50/p95/p99/max full-step and end-to-end latency, deadline misses, queue backlog, dropped samples, warm-up and reset behavior, all-event worst-case versus mixed natural streams. |
| Computational economy | Refinement duty cycle, actual runtime/MACs where instrumented, peak memory and serialized bytes. Duty cycle is not a measured energy/battery saving; no power claims without power instrumentation. |
| Chunk dependence | Compare continuous streaming outputs across multiple hop sizes/boundary offsets. Quantify max/RMS mismatch; no count of overlapping samples as independent observations. |
| Operational coverage | Fraction flagged unreliable/abstained; error or suppression proxies on those intervals and on accepted intervals, broken down by montage/study. |

New figures: a single blink showing onset/peak/tail, gate decisions and emitted-sample timestamps; chunk-boundary stress overlays; recall-versus-clean-false-alarms; latency-versus-blink-error curves; posterior alpha before/after during genuine non-events; teacher-disagreement versus paired error; low-rank/dense ablation; quality/coverage plots; runtime waterfall and p99/deadline timelines. Keep simple waveform/no-harm pages as well as these technical diagnostics.

New mandatory acceptance tests for future implementation:

1. **Future-perturbation test:** change samples strictly beyond each output's permitted lookahead; emitted output must not change, within a declared floating-point tolerance.
2. **State/chunk test:** streaming versus the reference stateful execution, hop/boundary offsets, carried stride phase, filter/resampler buffers, session resets and missing-data gaps. No resetting GRU state at every ordinary hop.
3. **Blink-boundary test:** replay genuine annotated blink boundaries at every hop offset; cover missed onset, premature gate exit, repeated events and already-emitted-sample immutability.
4. **Worst-case compute test:** force refinement always on and test noisy/channel-loss bursts. An average speedup is not a bounded deadline guarantee.
5. **Mask/topography test:** permute channels and metadata together; mask padded electrodes; test frontal dropout; disallow region-specific claims with guessed Klados mapping.
6. **Quantization test:** preserve gate calibration, paired metrics, blink recall and clean harm after conversion. Recalibration uses development data only. Fresh-load equivalence must include streaming state schema.
7. **Isolation test:** OSF annotation use only for evaluation in Protocol Z, adaptation-only supervision in Protocol A, and participant purging unchanged. Input-only calibration never reads clean test targets.
8. **Evidence gate:** missing event labels yield explicit N/A; unsupported hardware yields unavailable benchmarks; no DOCX victory merely because a real-time prototype runs.

PyTorch quantization is backend/version dependent. The current official docs direct development toward torchao; its linear-kernel support does not automatically quantize a fused GRU. Pin and test the available GRU/linear implementation rather than assuming that one conversion accelerates CPU, CUDA and TPU identically. Start with FP32 batch-1 CPU and supported CUDA FP32/FP16 comparisons, then CPU INT8 candidates. Keep precision-sensitive gate/correction accumulation in float if needed. CUDA gains on Kaggle do not certify latency on the user's laptop, mobile device or a wearable; under the Kaggle-only execution constraint, report only the tested Kaggle hardware configurations. [PyTorch quantization status](https://docs.pytorch.org/docs/stable/quantization.html), [torchao inference support](https://docs.pytorch.org/ao/stable/workflows/inference.html), [dynamic GRU API](https://docs.pytorch.org/docs/main/generated/torch.ao.nn.quantized.dynamic.modules.rnn.GRU.html).

**Incremental implementation order:** Phase A adds event/coordinate/participant provenance; B adds ARMBR and inspectable recent baselines; C decides whether regional teachers actually help; D trains the paired-only streaming student before distillation; E evaluates frozen and adapted variants separately; F adds compute control/quantization one at a time and runs continuous replay. Planned small modules: `streaming.py`, `blink_events.py`, `runtime_benchmark.py`, `quality_control.py`. One notebook orchestrates existing and new modules. Export separate offline and streaming PyTorch bundles with exact Git SHA, configuration, sampling/reference/filtering, lookahead, normalization, channel handling, gate calibration and state/reset contracts.

No training, signal-processing experiment, remote upload or model export was performed for this addendum. It advances the research design, not its validation status.

## References

1. Gandham Sai Sravanthi, Lakhan Dev Sharma. **EOG-free EEG using successive variational mode decomposition and fuzzy c-means clustering**. *Biomedical Signal Processing and Control* 115, 109434 (2026). [DOI](https://doi.org/10.1016/j.bspc.2025.109434). Publisher abstract inspected; full text unavailable.
2. He et al. **GVICA: A Multi-Channel EEG Hierarchical Noise Reduction Framework Based on GWO Dynamically Optimized VMD-ICA Fusion**. *IEEE Transactions on Biomedical Engineering*, 2026. [DOI](https://doi.org/10.1109/TBME.2026.3677897); [PubMed](https://pubmed.ncbi.nlm.nih.gov/41886329/). Fetched abstract via [Consensus](https://consensus.app/papers/gvica-a-multichannel-eeg-hierarchical-noise-reduction-he-liu/ba25e645c9b255b98990ab56bac703b9/?utm_source=chatgpt); full implementation unverified.
3. Diksha Srishyla et al. **Eye-movement artifact correction in infant EEG: A systematic comparison between ICA and Artifact Blocking**. *Journal of Neuroscience Methods* 418, 110405 (2025). [Primary manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC12753043/), [DOI](https://doi.org/10.1016/j.jneumeth.2025.110405).
4. Hyeonseok Kim et al. **Juggler's ASR: Unpacking the principles of artifact subspace reconstruction for revision toward extreme MoBI**. *Journal of Neuroscience Methods* 420, 110465 (2025). [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S0165027025001062), [DOI](https://doi.org/10.1016/j.jneumeth.2025.110465).
5. Wenlong You et al. **Adaptive Thresholding in EEG Artifact Removal Through Multimodal Fusion: A Multimodal Artifact Subspace Reconstruction Approach**. *IEEE TETCI*, early access 2025 / issue 2026. [Author repository](https://bura.brunel.ac.uk/handle/2438/32017), [DOI](https://doi.org/10.1109/TETCI.2025.3577504).
6. Carolina Rico-Olarte, Bjoern M. Eskofier, Diego M. Lopez. **EEG-cleanse: an automated pipeline for cleaning electroencephalography recordings during full-body movement**. *MethodsX* 15, 103702 (2025). [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12664388/).
7. Sahar Sattari, Naznin Virji-Babul, Lyndia C. Wu. **Contrast-Based Artifact Removal Enables Microstate Analysis in Ambulatory EEG**. *IEEE TBME* 73(7), 2351–2361 (2026; online 2025). [Publisher](https://ieeexplore.ieee.org/document/11231101/), [DOI](https://doi.org/10.1109/TBME.2025.3630112). Abstract-only method assessment.
8. Usman Qamar Shaikh et al. **Multi-head noise regression for single-channel EEG: estimating ocular and muscle contamination to guide artifact removal**. *Journal of Neural Engineering* 23(2) (2026). [Author manuscript](https://openrepository.aut.ac.nz/bitstreams/34ca287a-b70e-4f08-8dba-c67c387fba7a/download), [DOI](https://doi.org/10.1088/1741-2552/ae541d).
9. Vishnu KN, Cota Navin Gupta. **EMA-Based Subspace Tracking for Adaptive Artifact Subspace Reconstruction**. Preprint, submitted 2026-09-28. [arXiv](https://arxiv.org/abs/2609.35223). Full primary PDF inspected with alphaXiv; not peer-reviewed evidence.
10. Chinmayee Dora, Pradyut Kumar Biswal. **An improved algorithm for efficient ocular artifact suppression from frontal EEG electrodes using VMD**. *Biocybernetics and Biomedical Engineering* 40(1), 148–161 (2020). [Publisher](https://www.sciencedirect.com/science/article/pii/S0208521618303528), [DOI](https://doi.org/10.1016/j.bbe.2019.03.002). Foundational anatomical rationale, outside the new-paper window.
11. L. Alkhoury, G. Scanavini, S. Louviot, A. Radanovic, S. A. Shah, N. J. Hill. **Artifact-reference multivariate backward regression (ARMBR): a novel method for EEG blink artifact removal with minimal data requirements**. *Journal of Neural Engineering* 22(3), 036048 (2025). [DOI](https://doi.org/10.1088/1741-2552/ade566), [primary manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC12539671/), [author implementation](https://github.com/S-Shah-Lab/ARMBR). Primary text and PubMed abstract inspected; no replication performed.
12. Diego Marcos-Martínez et al. **Calibration-free Ocular artifact reduction in EEG signals using a montage-independent deep learning model**. *Biomedical Signal Processing and Control* 110, 108147 (2025). [DOI](https://doi.org/10.1016/j.bspc.2025.108147), [publisher abstract](https://www.sciencedirect.com/science/article/pii/S1746809425006585), [author repository](https://github.com/dmarcos97/EEGOAR-Net). Publisher abstract and native PyTorch source inspected; repository parity claims not independently tested. Check whether pretrained data overlap OSF before any evaluation; use matched Klados-only retraining for the primary comparison, not potentially overlapping pretrained weights.
13. Andrea Costanzo Palmisciano et al. **A Gated Artifact Management Pipeline for Low-Density Eyewear EEG**. *Physiological Measurement*, 2026-08-13 ahead of print. [DOI](https://doi.org/10.1088/1361-6579/ae99ab), [PubMed](https://pubmed.ncbi.nlm.nih.gov/42594937/). Abstract inspected; full causal/runtime protocol unverified.
14. Cassandra Dumas, Marie-Constance Corsi, Claire Dussard, Fanny Grosselin, Nathalie George. **Automatic Ocular Artifact Correction in Electroencephalography for Neurofeedback**. BIOSIGNALS/BIOSTEC 2025. [DOI](https://doi.org/10.5220/0013260900003911), [primary PDF](https://www.scitepress.org/Papers/2025/132609/132609.pdf), [author abstract](https://marieconstance-corsi.netlify.app/publication/dumas-automatic-2025/). Author abstract and indexed primary PDF excerpts inspected; numerical replication unverified.
15. Shantanu Sarkar, Jose L. Contreras-Vidal. **nASR: An End-to-End Trainable Neural Layer for Channel-Level EEG Artifact Subspace Reconstruction in Real-Time BCI**. [arXiv v2](https://arxiv.org/abs/2605.14941v2), [v2 primary text](https://arxiv.org/html/2605.14941v2), revised 2026-08-03; repository labels it IEEE SMC 2026 accepted/camera-ready. Deployment critique uses v2, not the older alphaXiv-returned v1.
16. Phat Lam. **BandRouteNet: An Adaptive Band Routing Neural Network for EEG Artifact Removal**. [arXiv v2](https://arxiv.org/abs/2604.24428v2), revised 2026-05-11. Primary PDF inspected via alphaXiv; preprint, not proof on our datasets.
17. Qiyu Rao, Haozhe Tian, Homayoun Hamedmoghadam, Danilo Mandic. **Enabling Unsupervised Training of Deep EEG Denoisers With Intelligent Partitioning**. [arXiv v2](https://arxiv.org/abs/2605.06724v2), [primary text](https://arxiv.org/html/2605.06724v2), revised 2026-09-26. Exploratory self-supervision evidence, not a verified blink-removal baseline.
18. Vincenzo Ronca et al. **A Novel Multi-Stage Algorithm for Real-Time Detection and Correction of Ocular Artifacts in EEG: A Calibration-Free Approach**. EMBC 2025. [DOI](https://doi.org/10.1109/EMBC58623.2025.11254864), [institutional abstract](https://iris.uniroma1.it/handle/11573/1767648), [PubMed](https://pubmed.ncbi.nlm.nih.gov/41335707/). Abstract-only assessment; causal timing and implementation not independently verified.

Official operational references: [Kaggle API authentication](https://www.kaggle.com/docs/api), [dataset commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md), [kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md), [kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), [MNE ICA](https://mne.tools/stable/generated/mne.preprocessing.ICA.html), [ICLabel contract](https://mne.tools/mne-icalabel/dev/api/iclabel.html), [mne-denoise ASR](https://mne.tools/mne-denoise/stable/asr.html).
