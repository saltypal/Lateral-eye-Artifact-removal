# Regional EOG Artifact Removal Research and Rehearsal Report

Research record through 16 October 2026

## 1 The project in one minute

We investigated removal of blink and lateral eye movement artifacts while preserving EEG. The working design uses fixed VMD on verified frontal channels and a separate nonfrontal estimator. We compared this design against simpler reference-regression controls, investigated unsuccessful corrections, and checked real OSF recordings as well as controlled mixtures.

**Current finding:** the latest EOG-assisted controlled development hybrid reaches 17.804727 dB whole-head and 18.261209 dB frontal channel-first output SNR. Matched direct regression reaches 17.785746 and 18.182199 dB. The small advantage is uncertain. The latest native pilot reduces ocular association but fails resting preservation. No teacher, EEG-only deployment model or production system is qualified.

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Historical 17.927210 dB | Replay of a guarded EOG-assisted hybrid on reused Klados data | VMD alone, fresh generalization or verified regional performance |
| Latest controlled comparison | Recovery relative to retained LEMON reference EEG under our constructed contamination | Artifact-free biological EEG or independent real-world superiority |
| Native OSF pilot | Recorded ocular association decreases; resting EEG changes too much | Reconstruction SNR or acceptable preservation |
| 341 tests in latest native run | Implemented numerical and source contracts passed | Scientific effectiveness or production readiness |

### Fast rehearsal route

With limited time, read sections 1–6 for the problem and representation, sections 14–16 for the latest evidence, and section 20 for answers to mentor questions. Use sections 7–13 and 17–19 when asked about alternatives, failed hypotheses, neural work or engineering decisions. References and the evidence register follow.

**Opening statement:** “Our project studies regional removal of blink and lateral eye artifacts. We achieved encouraging controlled reconstruction, but real-data evaluation exposed excessive resting-signal modification. Our contribution so far is a reproducible comparison and a clearer diagnosis of the remaining preservation problem, rather than a production-ready cleaner.”

**Reading rule:** experimental branches overlap in time. Section order follows the reasoning; it is not a claim that every job ran sequentially. Scores from different regions, corpora and aggregation conventions are not one leaderboard.

<!-- PAGE -->

## 2 Datasets and why we needed each one

| Source | Why we used it | Actual project scope and limitation |
|---|---|---|
| Klados and Bamidis [1] | Paired pre-contamination and contaminated EEG for legacy reconstruction tests | Local conversion: 53 records, 19 rows, 5,401 samples. Electrode ordering, participant mapping and physical units remain unverified. |
| Kobler OSF collection [2, 3] | Real blinks, horizontal/vertical movements, EOG and named electrodes | Publisher readme: 45 sessions, 39 participants, four studies. Development inventory: 36 records; latest frontal pilot: two records and 51 scoring trials. |
| LEMON [4] | Identifiable, named-channel low-ocular EEG recipients | Published multimodal cohort: 227 participants and a 62-channel resting EEG experiment. Our controlled development subset uses 32 recipients. |
| Project LEMON–OSF mixtures | Known pre-addition reference plus controlled ocular conditions | Latest corpus: 320 examples, 32 recipients, five donors. Constructed data, not 320 independent recordings. |

Klados allows comparison with EEG before artificial contamination. Its published collection has 54 recordings from 27 participants, 19 channels and 200 Hz sampling. These public characteristics do not establish identities in our 53-record conversion. We cannot guess which converted row is frontal or claim participant-independent evaluation from missing identities. Previously reused records are development-exposed.

OSF is the hosting service; Kobler and colleagues created the dataset. Its trial conditions are rest, lateral movement, vertical movement and blink. Sample labels additionally distinguish movement directions. HEOG means horizontal electrooculography; VEOG means vertical electrooculography. Both provide ocular evidence, but neither is automatically the exact artifact present at every EEG electrode. The local publisher readme, not an assumed session count, controls participant interpretation.

LEMON supplies the recipient waveform we retain before adding contamination. “Low-ocular” is deliberate: these segments are not proven biologically artifact-free. MNE-loaded LEMON EEG retains volts in this processing path. Native OSF amplitude units remain unverified; we do not transfer a microvolt label merely because another dataset uses it.

**Say:** “We need paired data to measure reconstruction and native data to test physiological transfer and preservation. No single source supplied all of these properties with verified metadata.”

**Sources:** Klados https://data.mendeley.com/datasets/wb6yvr725d/3 ; OSF https://osf.io/2qgrd/ ; LEMON https://doi.org/10.1038/sdata.2018.308 . The Klados deposit link is a public reference, not a verified version identity for our old conversion.

<!-- PAGE -->

## 3 How we constructed and tested the controlled examples

### The operation in plain language

1. Select a low-ocular EEG window from an identified LEMON recipient and retain it unchanged as reference A.
2. Use an OSF donor’s unscored EEG and HEOG/VEOG calibration to estimate the channel-specific ocular transfer.
3. Generate an estimated ocular contribution on verified common channels. This multichannel pattern is what we call the ocular field.
4. Scale and add the field to A to obtain contaminated input B. Supply the declared corresponding inference EOG, without exposing the generator’s true added field to the cleaner.
5. Run the denoiser on B to obtain C. Only the evaluator compares C with A.

**Conceptual flow:** LEMON reference A → add OSF-derived field → contaminated B → denoiser → output C → compare C with retained A. This is an explanatory diagram, not a measured result.

This is semi-synthetic: recorded EEG/EOG are ingredients, but their combination and amplitude are constructed. The ocular waveform did not occur naturally in that LEMON recipient’s recording. We did not overwrite original LEMON recordings. Nor did we simply paste an identical EOG trace into every channel.

### Why 320 does not mean 320 independent subjects

The latest broader corpus uses one 1,024-sample, 5.12-second recipient window for each of 32 participants. Each recipient/donor pair produces ten variants: one idealized clean control and blink, lateral and mixed contamination at nominal whole-montage input SNR −5, 0 and +5 dB. Thirty-two times ten gives 320 examples. Five donors are reused, limiting ocular diversity. Regional input SNR differs from the nominal whole-montage level.

### Bias and separation

Regression constructs the donor ocular field. A regression-based cleaner is therefore compatible with the generator’s assumptions; high controlled accuracy can favor that family even without access to true generator coefficients. Native and independent validation remain necessary. The zero-driver clean control means no added ocular field, not a guarantee that the original EEG contains no artifact.

Recipient and donor identities are frozen before mixture generation. Randomly separating related windows can leak the same background or ocular source across selection/evaluation. Global fitting and selection must respect both source identities. Per-record calibration is allowed only when declared and kept unscored. All current development evidence stays development-exposed; confirmation sources remain closed.

**Say:** “We know the signal before our addition, so we can measure whether the algorithm undoes that addition. We cannot infer perfectly clean biological EEG or real-data generalization from this constructed benchmark.”

**Evidence:** [E1], [E14].

<!-- PAGE -->

## 4 Why VMD was investigated and how it works

Imagine several sounds mixed into one microphone recording. VMD describes that recording using a chosen number of waveforms concentrated around different frequencies. It repeatedly adjusts the waveforms and their centre frequencies together until estimates change little or a solver limit is reached [5]. This is a representation step; it does not identify a pure eye source by itself.

| Question | VMD | ICA |
|---|---|---|
| Main evidence | Frequency structure within a waveform | Statistical structure across observed channels |
| Typical role here | Decompose one verified frontal channel | Estimate spatial components from multichannel calibration |
| Main limitation | Neural and ocular activity can occupy the same mode | Neural and ocular activity can occupy the same component |
| Cleaning decision | Additional reference/temporal evidence required | Additional component attribution required |

VMD suited our proposed independent frontal processing. ICA remained a reasonable spatial challenger for posterior/central EEG. These are design motivations, not physiological laws or demonstrated universal superiority. Independent frontal modes may not align across channels, so a mode index cannot be treated as a common source identity. Iterative VMD also has real computational cost.

### K and alpha

K is the requested number of modes. Too few may mix processes; too many can split activity redundantly. Alpha is the bandwidth penalty: under the same solver convention, larger values encourage tighter frequency concentration. It is not a cutoff in Hz, an amplitude threshold, or the EEG alpha band. Five physiological frequency bins do not justify K=5.

Our fixed recipe uses K=3, alpha=2000, input RMS normalization, uniform initialization, tolerance 1e−6 and a 2,000-iteration cap. It retains historical continuity after AutoVMD stopped. It is not an established optimum or a verified universal paper setting. K4/2000 was a fixed diagnostic comparator.

For 1,024 input samples, K3 produces three vectors of length 1,024 over the same interval. We retain the residual, defined as original minus the sum of modes. Estimated artifacts are subtracted from the original rather than reconstructing EEG only from “kept” modes. Strong HEOG/VEOG association is evidence of ocular contribution, not proof that the whole associated mode should be deleted.

**Say:** “VMD organizes a frontal waveform into adaptive components. Our correction mechanism must still decide what to subtract. Better-looking components do not automatically mean better preserved EEG.”

**Evidence:** [E2], [E3].

<!-- PAGE -->

## 5 Read the actual VMD mode figure

:::figure modes 6.65 6.35

**Figure 1.** Fixed K3/alpha2000 on one predeclared F7 controlled lateral example at nominal whole-input 0 dB. Horizontal axis: seconds; vertical axes: microvolts, with different scales between panels. The top blue curve is observed EEG; the dashed orange curve is the injected field available to evaluation only. Modes have measured centres approximately 0.86, 6.98 and 24.15 Hz. The residual is retained and is not presumed purely neural or ocular.

**Say:** “These are three simultaneous descriptions of the same time interval, not three time chunks or fixed delta/theta/beta bins. The slow mode resembles broad ocular activity, but resemblance does not establish purity.” Source: broader-frontal-review-001 [E15].

<!-- PAGE -->

## 6 AutoVMD and what the parameter plots establish

AutoVMD was a parameter-selection workflow around VMD, not a new decomposition solver. The declared grid had eight K values, 3–10, and five alphas, 250/500/1000/2000/4000: 40 configurations. The doubling alpha bank explored a broad numerical range within a bounded budget, without proving those bounds optimal.

:::figure centers 6.65 1.48

**Figure 2.** Earlier classical diagnostic on one Fz example. Each panel fixes alpha; horizontal axis is K, vertical axis is centre frequency in Hz, and each dot is a mode. Nearby centres motivate checking bandwidth/overlap; they do not automatically invalidate K. This completed diagnostic is distinct from unfinished AutoVMD neural selection. Open the embedded image at higher zoom for individual dots.

| Planned selection stage | Intended budget |
|---|---|
| Screen all 40 recipes | Five-epoch routing experiments on the same declared subset |
| Shortlist eight | Fifteen epochs over three inner folds |
| Shortlist two | Up to 80 epochs and two learning rates |
| Outer evaluation | Five outer folds; selected recipe fixed during each run |

Caches preserve vectors, residuals, centres, bandwidths, overlap, convergence and source/configuration identities. Reusing an exact decomposition avoids paying its cost for every correction experiment. K and alpha were to remain fixed during training and seed repeats, rather than changing each epoch or per window.

K3–K10 cache summaries record 1,600 entries per K across five alphas. These entries are cached decompositions, not trained models. The complete grouped neural screening/selection did not finish and was stopped. No AutoVMD winner is justified. Earlier documentation’s pending cache boxes are historical snapshots; available cache evidence does not imply completed selection.

**Say:** “The sweep helps inspect representation behavior. AutoVMD was supposed to select parameters by downstream recovery and preservation, but its complete selection did not finish. K3/2000 remains a bounded baseline.”

**Evidence:** vmd-001 classical diagnostic and [E4].

<!-- PAGE -->

## 7 Literature and the fixed frontal correction screen

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

**Say:** “Literature motivates mechanisms, but our adaptations must earn their own evidence. We did not reproduce every paper or find that more processing is automatically better.” Evidence: [E2], [E5].

<!-- PAGE -->

## 8 Historical replay and calibrated frontal transfer

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

**Say:** “We audited the old high score, found its full information requirements, and tested transfer. The new calibrated method was a different experiment, with its own stronger raw control and preservation failure.” Evidence: [E6], [E7].

<!-- PAGE -->

## 9 Temporal support and separate ocular axes

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

**Say:** “We tested when and how strongly to correct. Turning correction down improved preservation but often removed the useful suppression as well. None of the tested support recipes solved both requirements.” Evidence: [E8], [E9].

<!-- PAGE -->

## 10 The rejected mode shape restriction

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

**Say:** “A cleaner-looking restricted representation was not a better correction. It removed information needed to estimate the artifact. We rejected that branch and kept the evidence.”

**Evidence:** [E10]. The report tables retain every condition, not only the aggregate failure.

<!-- PAGE -->

## 11 Posterior ICA and GEVD MWF with frontal support

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

**Say:** “Frontal support helped some spatial recipes relative to posterior-only fitting, but direct regression remained highly competitive. We did not establish ICA or MWF as the posterior winner.”

**Evidence:** [E11].

<!-- PAGE -->

## 12 Regional integration exposed the missing middle

Initial integration combined fixed frontal VMD with posterior regression or ICA and passed central/shared channels through. Whole-cap output was 9.094213 dB with posterior regression and 9.089350 dB with posterior ICA; matched raw frontal plus posterior regression was 9.337592 dB. Regional scores cannot be added to obtain these values.

:::figure errors 6.65 2.83

**Figure 4.** Earlier frozen regional diagnostic. Vertical axis is mean per-example fraction of reconstruction-error energy, then source/condition balanced. Central/shared pass-through contributes 87.9066%; frontal 2.9326%; posterior 9.1608%. This is not a globally pooled energy ratio and does not prove that removing all central error is feasible without neural loss.

| Earlier full regional composition | Whole output SNR dB | Central/shared SNR dB |
|---|---:|---:|
| Central pass-through | 9.094213 | 1.954134 |
| VMD frontal artifact context for central transfer | 13.162704 | 10.903434 |
| Raw frontal artifact context for central transfer | 13.183201 | 10.949306 |

Both context arms retained the same scored VMD frontal and direct posterior outputs. “Raw context” did not mean replacing the frontal denoiser or an EEG-only method. The raw-minus-VMD context whole difference was 0.020497 dB with an interval crossing zero. Context coverage helped; unique benefit from VMD-derived context remained unproven.

**Say:** “The first whole-head score was low largely because we left the middle region untouched. Giving that region an estimator improved coverage. It did not by itself solve lateral contamination or native preservation.” Evidence: [E12], [E13].

<!-- PAGE -->

## 13 Nonfrontal challengers and the stronger simple control

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

**Say:** “More separation did not necessarily help. Complete reference regression was the strongest control in this comparison, so we retained it and investigated the frontal and data-generation limitations.” Evidence: [E14].

<!-- PAGE -->

## 14 Corpus repair and the anchored lateral correction

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

**Say:** “We repaired the experimental construction and tested whether raw-field information stabilizes an incomplete VMD-based gain. The pilot improved lateral behavior relative to old VMD, then we checked the same recipe on broader sources.” Evidence: [E15], [E16].

<!-- PAGE -->

## 15 Latest controlled results and the size of the VMD benefit

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

## 16 Native OSF validation is the decisive current checkpoint

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

**Say:** “The system suppresses ocular association, including lateral movement, but it intervenes too strongly during resting EEG. This is the main issue to solve before using it as a teacher or claiming deployment.” Evidence: [E18].

<!-- PAGE -->

## 17 Neural experiments and what deployment would require

### What was implemented and actually trained

The earlier band-routing adaptation includes a shared width-32 encoder, two scales, Inception kernels 3/7/15/31, group normalization, GRU bottleneck and component attention: 222,211 parameters. A separate shared TCN–BiGRU student has five dilated blocks, 32 units per direction and 47,427 parameters. Software fixtures exercised model arms, CUDA optimization and fresh CPU reload. Those are implementation checks, not denoising qualification.

| Five-epoch development pilot | Output SNR dB | Worst-source clean relative error |
|---|---:|---:|
| Fixed-band MSE | 7.772145 | approximately 0.479 |
| Student MSE | 7.364268 | 0.478943 |
| Fixed-band preservation loss | 7.458374 | 0.554103 |
| Student preservation loss | 7.619747 | 0.460719 |

All four pilots failed clean preservation. They used a bounded development fold/seed, not completed multi-seed grouped final training. The preservation loss added identity, log-PSD and covariance terms with project weights 1.0/0.1/0.05. The results do not establish that longer training, a larger model or a larger SNR penalty will solve preservation. A diagnosis dispatcher failed after 78 tests and was repaired before the replacement diagnosis; this was an integration failure, not a cleaning score.

### Why TCN and BiGRU were considered

A TCN supplies temporal context through dilated convolutions. A BiGRU combines information from both time directions, so this reference is offline, not causal streaming. Compact CNN and TCN-only alternatives may be cheaper; superiority requires matched training budgets. Band-routing work is a manuscript-inspired adaptation [14], not an exact reproduction or a separately proven “V2” architecture.

The intended deployment student accepts EEG, masks and verified metadata without EOG or runtime VMD. It is distinct from the current EOG-assisted classical system and the historical 2,499-parameter EOG-assisted gain model. Qualified teacher supervision would require cross-fitted targets excluding both recipients and donors from global fitting/selection. It should be retained only if it improves a paired-only student.

**Say:** “Neural implementations and short pilots exist, but no EEG-only student is qualified. The current failed teacher cannot be treated as clean ground truth.” Evidence: [E4], [E19].

<!-- PAGE -->

## 18 Metrics and the rules for interpreting them

| Quantity | Plain meaning | Main qualification |
|---|---|---|
| MSE | Average squared difference from paired reference | Lower is better; amplitude units must be known |
| Temporal RRMSE | Reconstruction error normalized by reference RMS | Lower is better; report aggregation |
| Ordinary Pearson CC | Waveform co-variation | High CC does not establish amplitude or spatial preservation |
| Output SNR | Reference energy relative to reconstruction-error energy, expressed in dB | Requires a paired reference; higher is better |
| SNR improvement | Output SNR minus input SNR under the same definition | Not the same number as output SNR |
| Native EEG–HEOG/VEOG correlation | Remaining association with ocular reference | Suppression proxy; low association can coexist with overcorrection |
| Rest RMSE and spectral/covariance changes | How much resting recordings changed | Rest is not proved clean; thresholds are project engineering limits |

Output SNR uses 10 times log10 of reference energy divided by error energy. PSNR depends on a justified peak/scaling convention and is not a substitute. A zero or nearly constant output is not acceptable merely because ocular correlation becomes small. Nonfinite/undefined metrics must remain visible rather than scored as successes.

### Why apparently conflicting SNRs can both be correct

Channel-first SNR averages individual channel log ratios before balanced summaries. Pooled-window SNR first combines channel energies within each window, then averages window scores. These weight heterogeneous errors differently. A method can have a slightly higher mean log-SNR but worse MSE or pooled score. Latest fallback whole-head figures are 17.804727 dB channel-first and 15.200363 dB mean pooled-window; neither can be substituted opportunistically. Regional dB values cannot be added.

### Paper metrics and project gates are different

MSE, temporal RRMSE and ordinary CC follow the prescribed paired literature. Native reference correlation/resting evaluation follows the declared EEGOAR-Net/OSF protocol [13], with implementation adaptations documented. Project limits are not universal paper standards: worst-source mean clean modification ≤1%, alpha/beta error ≤0.5 dB, relative covariance error ≤0.02 and CC deterioration ≤0.005 versus the declared comparator.

Supplementary controlled uncertainty uses recipient/donor-aware Poisson-multiplier draws, 2,000 repetitions and seed 42. It is a project approximation, especially limited by five donors. Native participant permutation is a separate paper-grounded procedure; two participants provide little inferential power. Intervals including zero do not establish an incremental benefit.

**Say:** “Every score needs a reference, population, region and aggregation. We report suppression and preservation together.” Evidence: [E20].

<!-- PAGE -->

## 19 Engineering evidence failures and next decisions

Reusable Python modules live in Git on research/vmd-eog-removal-notebooks. Numerical tests, signal processing, fitting, training, evaluation and measured plots run on Kaggle. Notebooks resolve one published SHA, create a fresh checkout, attach exact parent versions and save persistent run/configuration/environment/split/prediction artifacts. Local work authors source, handles Git/submission, checks bytes and reads saved reports. This report performs no new EEG analysis.

| Failure or limitation | What we did or must preserve |
|---|---|
| Missing study04 SET/FDT dependency | Restored the versioned source bundle and pinned checksums; preflight prerequisites |
| Neural diagnosis routed to legacy dispatcher | Repaired dispatch; retained original failed run and replacement version |
| Float32 mode reconstruction differences | Accumulated in float64 and checked a storage-rounding budget |
| Collapsed projected-axis identifiability fixture | Preserved failure; repaired/qualified contracts before comparison |
| Copied subgroup CC reporting | Recomputed true subgroup diagnostics before full CCA claims |
| Boolean False counted as non-null failure | Repaired to null-success/reason-string; cached metrics/intervals reproduced |
| Three controlled solver-cap rows | Added one failure-only raw fallback, preserving all other outputs |
| HTTP502, TLS and download timeouts | Bounded retrieval, hash-verified resume and byte bundles; API failure is not kernel failure |

The original broader run’s complete local recovery remained incomplete at the earlier transport checkpoint; this report does not promote it to full recovery. The latest native run has all 744 payloads verified. Watcher completion and current job status must be checked from receipts/events, not old “running” text. Local Python watchers provide status/retrieval; this is not a continuously reasoning LLM after a chat turn.

### What remains before qualification

First diagnose resting interventions using saved reference amplitude/slew, association, gain, selected modes and calibration baseline/variance. Compare with prior unsuccessful support recipes; do not repeat an unbounded threshold or K/alpha search. Any next bounded adaptation must retain matched raw/identity controls, separate conditions and unchanged preservation limits. Then expand eligible native development validation before model qualification.

Fresh reserved controlled evaluation, independent waveform datasets, final model freeze, cross-fitted distillation, CPU/ONNX parity and measured 10/20/30/40/50-channel quality/runtime scaling remain incomplete. EEGdenoiseNet, Magdeburg and additional BCI/ERP sources were considered/audited; no completed external waveform result is claimed here. Parallel per-channel VMD is possible in principle, but actual channel-scaling cost must be measured. Native stage runtimes include fitting/saving and cannot be called inference-only benchmarks.

**Say:** “Passing contracts makes the experiment interpretable; it does not certify a cleaner. The next milestone is preservation with retained suppression, followed by independent validation.”

<!-- PAGE -->

## 20 Rehearsal script and likely mentor questions

### A sixty second opening

“We are studying removal of blinks and lateral eye artifacts while preserving EEG. We used Klados for legacy paired testing, OSF for real ocular recordings and named electrodes, and LEMON as recipient EEG for controlled mixtures. Our design investigates frontal VMD with a separate nonfrontal estimator. We compared it with direct regression and spatial alternatives. The latest controlled VMD hybrid reaches about 17.80 dB whole-head output SNR, close to the matched direct control. Native OSF shows ocular suppression but excessive resting modification. Therefore our current system is a reproducible research candidate, with preservation as its main unresolved problem.”

### Questions and defensible answers

**Why VMD rather than ICA?** It supports per-channel frontal representation; ICA exploits multichannel structure. We investigated both roles and have not proved universal VMD superiority.

**Why K3 and alpha2000?** Historical continuity and a fixed bounded baseline after AutoVMD stopped. The historical high score included EOG, a learned gain and gating. K3/2000 is not a measured optimum.

**Did the 40-setting search finish?** Earlier classical diagnostics and caches exist. The full AutoVMD grouped neural-selection campaign did not finish. Cache completion is not model-selection completion.

**Did we make the artifacts artificially?** We combined recorded LEMON EEG with an OSF-derived, regression-estimated ocular field. This gives known pre-addition reference EEG but can favor regression and does not replace native validation.

**Must we retrain for every dataset?** Frozen neural external testing should retain weights. A declared classical calibration procedure may fit new per-record transfer/covariance/ICA parameters without redesigning the method. Fine-tuning is a separate adaptation experiment.

**What does each mode represent?** An adaptive full-length waveform with measured centre/bandwidth and ocular evidence. A mode index is not a physiological label or guaranteed pure source.

**Why are some scores 2, 9 or 18 dB?** They describe rejected branches, different coverage, cohorts or aggregation. The 2.52 dB shape restriction was rejected; a regional pass-through bottleneck produced about 9.09 dB. Compare only matched protocols.

**Can regression be the final method?** It is a strong valid comparator, but requires reliable EOG/calibration and currently fails native preservation here. The synthetic generator is also regression-derived. An EEG-only device needs a separately validated solution.

**Are we better than the latest published methods?** Not established. We have adaptations, limited exposed cohorts and no complete independent matched benchmark. Our data do not support a state-of-the-art claim.

**What is the main contribution so far?** A versioned regional comparison, source/error attribution, preserved negative results and identification of the remaining suppression/preservation trade-off. Final novelty must be argued after independent validation.

**What will we do next?** Diagnose excessive resting correction, test a bounded observed-data support mechanism against strong controls, and require preservation before promoting a teacher or deployment claim.

<!-- PAGE -->

## 21 References for datasets and algorithm foundations

[1] Klados, M. A., and Bamidis, P. D. (2016). A semi-simulated EEG/EOG dataset for the comparison of EOG artifact rejection techniques. Data in Brief, 8, 1004–1006. https://doi.org/10.1016/j.dib.2016.06.032 . Public deposit: https://data.mendeley.com/datasets/wb6yvr725d/3 . Deposit version is not verified for our old converted arrays.

[2] Kobler, R. J., Sburlea, A. I., Lopes-Dias, C., Schwarz, A., Mondini, V., and Müller-Putz, G. R. EEG eye artifact dataset. Open Science Framework. https://doi.org/10.17605/OSF.IO/2QGRD . Counts and annotations follow the publisher readme distributed with our copy: 45 sessions and 39 participants. Author repository: https://github.com/rkobler/eyeartifactcorrection .

[3] Kobler, R. J., et al. (2020). Corneo-retinal-dipole and eyelid-related eye artifacts can be corrected offline and online in electroencephalographic and magnetoencephalographic signals. NeuroImage, 218, 117000. https://doi.org/10.1016/j.neuroimage.2020.117000 .

[4] Babayan, A., et al. (2019). A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults. Scientific Data, 6, 180308. https://doi.org/10.1038/sdata.2018.308 . EEG download source: http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON/downloads/download_EEG.html .

[5] Dragomiretskiy, K., and Zosso, D. (2014). Variational mode decomposition. IEEE Transactions on Signal Processing, 62(3), 531–544. https://doi.org/10.1109/TSP.2013.2288675 . Author resources: https://math.montana.edu/dzosso/code/ . Python reference: https://github.com/vrcarva/vmdpy .

[6] Dora, C., and Biswal, P. K. (2020). An improved algorithm for efficient ocular artifact suppression from frontal EEG electrodes using VMD. Biocybernetics and Biomedical Engineering, 40(1), 148–161. https://doi.org/10.1016/j.bbe.2019.03.002 . Full numerical settings were not recovered for exact reproduction.

[7] Liu and Zhang (2022). Remove artifacts from a single-channel EEG based on VMD and SOBI. Sensors, 22, 6698. https://doi.org/10.3390/s22176698 . Author wrapper audited through https://github.com/Readyou123/V-sobi . Required helper functions and solver parity remain limitations.

### Citation discipline

These papers motivate mechanisms and evaluation. Our project settings, engineering thresholds, fixed blend and calibration/gain choices are not automatically prescribed by these sources. Study-specific reported scores cannot be converted into a claim that our method is superior without matched datasets and definitions.

<!-- PAGE -->

## 22 References for correction and neural comparisons

[8] Xiong et al. (2024). Research on ocular artifacts removal from single-channel electroencephalogram signals in obstructive sleep apnea patients based on support vector machine, improved variational mode decomposition, and second-order blind identification. Sensors, 24, 1642. https://doi.org/10.3390/s24051642 . Our fixed adaptation defers GA and mRMR; reported parameter inconsistencies remain disclosed.

[9] Schlögl, A., et al. (2007). A fully automated correction method of EOG artifacts in EEG recordings. Clinical Neurophysiology, 118(1), 98–104. https://doi.org/10.1016/j.clinph.2006.09.003 .

[10] Plöchl, M., Ossandón, J. P., and König, P. (2012). Combining EEG and eye tracking: identification, characterization, and correction of eye movement artifacts in electroencephalographic data. Frontiers in Human Neuroscience, 6, 278. https://doi.org/10.3389/fnhum.2012.00278 .

[11] Issa and Juhasz (2019). Improved EOG artifact removal using wavelet enhanced independent component analysis. Brain Sciences, 9(12), 355. https://doi.org/10.3390/brainsci9120355 . Our wavelet/component adaptations are not author-pipeline parity.

[12] MWF author implementation and documentation. https://github.com/exporl/mwf-artifact-removal . Our GEVD–MWF changes covariance regularization, support and output restrictions; the reference toolbox’s manual/automatic mask requirements must not be hidden.

[13] Marcos-Martínez, D., et al. (2025). Calibration-free ocular artifact reduction in EEG signals using a montage-independent deep learning model. Biomedical Signal Processing and Control, 110, 108147. https://doi.org/10.1016/j.bspc.2025.108147 . Author implementation: https://github.com/dmarcos97/EEGOAR-Net . This is a method/evaluation reference, not the origin of Kobler’s OSF data.

[14] BandRouteNet manuscript revision v2, the recorded inspiration for our routing adaptation. https://arxiv.org/html/2604.24428v2 . A manuscript revision is not evidence of an independently qualified “V2” architecture. Our model, data and budgets differ; exact reproduction is not asserted.

[15] Picard author implementation. https://github.com/pierreablin/picard . Extended ICA is an implementation component; component selection and regional output restrictions are project adaptations.

### Further methods reviewed or awaiting qualification

Frontal multivariate VMD, VME–DWT/VME–GMETV, published SVMD/fuzzy-clustering and diffusion-based refinement were reviewed as alternatives. SGEYESUB calibration/source eligibility was audited, but no qualified reproduction is claimed. An older repeated-K1 approximation is not the published SVMD optimizer. Independent benchmark eligibility audits do not establish waveform validation. See the dated alternatives and source-audit documents in [E21].

<!-- PAGE -->

## 23 Evidence register and complete record access

This report covers every major experimental branch located in the project record, with selected exact results and representative figures. Every candidate/channel/record table is not duplicated into these pages. The append-only Work So Far.md and its evidence index preserve the detailed historical specifications, failures and available per-record outputs. Historical status text is dated; it does not supersede completed-run receipts.

Paths below are relative to the VMD_EOG_Removal project root. All numerical results and plot values were read from saved Kaggle reports; no new EEG fitting, inference, statistics or plotting was run locally to prepare this report.

| ID | Project source |
|---|---|
| E1 | docs/mentor/sections/01_datasets.md; complete_dataset OSF publisher readme in parent workspace |
| E2 | docs/FRONTAL_PAPER_AUDIT.md; docs/ALGORITHM_REFERENCES.md |
| E3 | docs/FRONTAL_METHOD_GUIDE.md; src/vmd_eog/frontal_methods.py |
| E4 | docs/AUTOVMD_IMPLEMENTATION.md; configs/bandroute.json |
| E5 | docs/FRONTAL_RESULTS.md |
| E6 | docs/GUARDED_K3_ALPHA2000_REPLAY.md; docs/GUARDED_REPLAY_RESULTS_20261011.md |
| E7 | Work So Far.md sections 6 and 7; docs/SIX_HOUR_DELIVERY.md |
| E8 | docs/NATIVE_SEPARATE_AXIS_RESULTS.md; docs/FRONTAL_BOUNDED_FEASIBILITY_RESULTS.md |
| E9 | docs/NATIVE_TRIAL_SUPPORT_RESULTS.md |
| E10 | docs/FRONTAL_ARTIFACT_SHAPE_RESULTS.md; docs/FRONTAL_SHAPE_DIAGNOSTIC_RESULTS.md |
| E11 | posterior results.md, read-only other-chat report; docs/POSTERIOR_CONTEXT_PLAN.md |
| E12 | docs/REGIONAL_RESULTS.md; docs/REGIONAL_DIAGNOSTIC_RESULTS.md |
| E13 | docs/CENTRAL_COVERAGE_RESULTS.md; docs/MATCHED_CONTEXT_RESULTS.md |
| E14 | docs/NONFRONTAL_CHALLENGERS_RESULTS.md; docs/NONFRONTAL_CCA_PLAN.md |
| E15 | docs/BROADER_ANCHORED_VALIDATION_PLAN.md; docs/FRONTAL_REFERENCE_COUPLING_REPAIR_PLAN.md |
| E16 | docs/COUPLED_FRONTAL_PILOT_RESULTS.md; docs/LATERAL_GAIN_RESULTS_20261015.md |
| E17 | docs/BROADER_FRONTAL_RESULTS_20261015.md; broader-cached-review-001 outputs |
| E18 | docs/NATIVE_FROZEN_FRONTAL_RESULTS_20261015.md; native-frozen-frontal-001 outputs |
| E19 | docs/NEURAL_PRESERVATION_DIAGNOSIS.md; four saved pilot training_summary.json files |
| E20 | docs/EVALUATION_PROTOCOL.md; docs/REPAIRS.md |
| E21 | docs/RESEARCH_ALTERNATIVES_20261010.md; docs/LATERAL_REPAIR_REVIEW_20261015.md; docs/FRONTAL_BENCHMARK_ELIGIBILITY_RESULTS.md |

**Full ledger:** Work So Far.md. **Evidence index:** docs/work_so_far_evidence_index.json. **Current checklist:** TASKS.md. The ledger’s large historical annex is supporting evidence, not a short rehearsal script.

<!-- PAGE -->

## 24 Selected run identities and presentation guardrails

| Completed run or branch | Source identity | Evidence and scope |
|---|---|---|
| vmd-001 | befa09f | Earlier 40-setting classical diagnostic; not completed neural AutoVMD |
| frontal-screen-001 | 638d496 | 708 correction recipes, 20 examples |
| posterior-context-full-001 | a25acb9 | 737 candidates; posterior development search |
| regional-diagnostic-full-001 | 220e83d | 134 tests; frozen 320-example error diagnosis |
| native-bounded-feasibility-001 | 4719e52 | 215 tests; 48 active settings all fail preservation |
| native-trial-support-001 | 0e33d87 | 224 tests; timing-boundary adaptation fails qualification |
| frontal-shape-full-001 | 1444b4e | 236 tests; rejected 2.521295 dB shape correction |
| nonfrontal-full-001 | 46171ac | 281 tests; 21-arm full comparison |
| nonfrontal-cca-full-001 | b0c8613 | 299 tests; complete ridge stronger than CCA |
| lateral-gain-review-003 | 5b59c6d | 325 tests; two-pair anchored candidate |
| broader-frontal-review-001 | f126300 | 330 tests; fixed broader comparison |
| broader-cached-review-001 | 5ad2c35 | 332 tests; original metrics reproduced, failure-only fallback |
| native-frozen-frontal-001 | c48af6a | 341 tests; all 744 payloads verified; preservation failed |

The latest native source SHA is c48af6a0cda4d6bd905856613a3cf0c01f5bdcd2. Its manifest SHA256 is ccc9253ee7b71f676bfacf805c5d7aabf3b4fed3cae147abad707c04ff071111. Exact full identities and immutable parents for other rows remain in run_spec.json, source reports and the full ledger. Short hashes above are navigation aids, not replacements for reproduction pins.

### Statements we can defend

We investigated regional correction, preserved negative results and compared simpler controls. K3/2000 is a fixed baseline. Controlled development reconstruction is encouraging but biased by construction. Native ocular association decreases, while resting preservation fails. Final teacher/student and independent qualification are incomplete.

### Statements the evidence does not support

We have not shown complete removal of all EOG, universal VMD superiority, an optimal K/alpha from AutoVMD, fresh 17.93 dB generalization, native reconstruction SNR, journal acceptance, production readiness or verified 50-channel real-time deployment. We do not add regional decibels, call software tests scientific success, or interpret relative RMS modification as a percentage of neural activity proved lost.

**Closing statement:** “Our next experiment is driven by the native failure: reduce inappropriate resting intervention without losing lateral/blink suppression. Only after that evidence passes should the system be promoted to a teacher and distilled or replaced by a deployment model.”
