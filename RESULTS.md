# EEG ocular-removal results

Latest completed stage: central/temporal comparison,38 contracts and248 verified
artifacts. Central ridge100% output18.3343dB on reused paired Klados records,
but all corrective preservation checks fail. See the central/temporal section
below; frontal results remain a separate benchmark.

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

## Pretrained-only extension: release eligibility

EEG-WNet and Diff-ADN are **not run—pretrained reproduction unavailable** within
the inspected sources as of17October2026. No compatible author-code/checkpoint
pair was verified. Empty searches and failed publisher access are recorded
separately; they are not proof that weights do not exist anywhere. No new
training, fine-tuning, substitution or denoising predictions occurred.

The literature survey and release queries are in docs/PAPERS.md and
docs/provenance/frontal_pretrained_discovery.json. A separate numbered CPU
notebook verifies source/hash-pinned completed parents and presents this
explicit non-execution decision beside the existing metrics. Its completion
will not change any method score or resolve the native preservation failure.

The extension is now complete: sourcea8c7b7171ad15b373422e233357f670159b0dee8,
Kaggle kernel `satyapaladugu/frontal-pretrained-closeout/2`,
run `frontal-pretrained-closeout-002`. **36 contracts passed**; all22 payloads
were retrieved and SHA256/length verified. The actual executed notebook has
3/3 executed code cells,2 inline figures,4 HTML tables and zero error outputs.
Both generated figures were visually inspected for labels, bars and baselines.

Open [the executed pretrained closeout](notebooks/executed/frontal-pretrained-closeout-002.ipynb).
The notebook shows existing paired reconstruction and native suppression/
preservation side by side. It does not contain EEG-WNet/Diff-ADN predictions.
The first version passed35 contracts but failed because Git newline conversion
changed the raw source-JSON checksum; its failure notebook is preserved. The
repair uses an explicitly defined canonical-JSON identity for that text snapshot
only; dataset and scientific artifact checks remain byte-exact.

**Frontal closeout:** the six existing methods and correlation audits are done;
two new pretrained candidates were investigated but not run. Best paired
frontal output remains historical16.134624dB/CC0.984644 on reused Klados
records. Native preservation remains failed, so there is no qualified frontal
winner. Regional/shared-message implementation is outside this completed
extension. No new training or numerical improvement is claimed.

# Central and temporal comparison — 17 October 2026

Run `central-temporal-001`, source `20584ab33028b43fada3912960e926c7948f70cc`,
Kaggle CPU kernel `satyapaladugu/central-temporal-eog-comparison/1`.
Central = FC/C/CP, temporal = FT/T/TP with separate fits/results. Native OSF
contains21 central and6 temporal channels per examined session; Klados central
rows are C3/C4/Cz, temporal T3/T4/T5/T6 using the disclosed paper-list mapping.
Eight archived Klados test records,24 windows per group; four OSF participants,
32 scored trials per group. All correction fits use calibration only.

| Fixed method | Central output SNR (dB) | Central clean EEG CC | Temporal output SNR (dB) | Temporal clean EEG CC |
|---|---:|---:|---:|---:|
| Identity |10.5743|0.926343|12.3977|0.947137|
| Ridge,25% strength |12.6422|0.950379|13.9872|0.963254|
| Ridge,100% strength |18.3343|0.969361|16.7267|0.951142|
| GEVD–MWF,25% strength |10.4977|0.925898|12.2443|0.949004|
| GEVD–MWF,100% strength |9.0605|0.917523|9.8424|0.930023|

SNR is pooled energy per window, then equal window means within records and
equal record means in dB. CC is ordinary Pearson. Ridge penalty0.001 and MWF
rank1 are fixed; strengths0.25/1 are predefined ablations, not selected recipes.
These results do not establish superiority over other ranks or MWF algorithms.
The separately filtered calibration/scoring protocol differs from archived
frontal caches: no matched cross-region superiority or summed regional dB claim.

Native OSF central results (unchanged H/V references, equal participant weight):

| Method | Blink abs VEOG CC | Lateral abs HEOG CC | Mean resting RMS modification |
|---|---:|---:|---:|
| Before / identity |0.783635|0.187934|0%|
| Ridge25% |0.708279|0.126766|3.55%|
| Ridge100% |0.109971|0.144744|14.21%|
| MWF25% |0.725631|0.189268|7.92%|
| MWF100% |0.219712|0.190806|31.70%|

**No corrective method passes all declared preservation checks.** Worst
participant mean central rest modification is4.06% for ridge25% and16.25% for
ridge100%, versus1% engineering threshold. Paired clean-input controls also
fail: ridge100% worst-record central modification89.35%, temporal136.29%.
These are signal-modification ratios, not percentages of neural activity lost.
The strong mean SNR therefore cannot qualify ridge as a safe central method.

Mean scores conceal poor examples: record16/start3024 central ridge100% output
SNR is−1.073532dB; its temporal output is−4.110142dB. Temporal record45 also
has negative-output windows. All are included, with no fit failures/exclusions.
Preservation/comparator gates and individual rows remain available in artifacts.
Identity is the pass-through control, not a successful artifact-removal model.
Native EEG has no paired clean target, so no native reconstruction SNR.

The experiment passed38 contracts in5.02s; zero calibration failures. It saved
eight figures, named prediction arrays, original references, fitted operators,
per-channel/window metrics, source identities and completion state.
Open [the executed central/temporal notebook](notebooks/executed/central-temporal-001.ipynb).
Recovery verification is recorded in results/central-temporal-20261017/central-temporal-001/retrieval_receipt.json.

The next research decision is reliable detection/transfer of ocular activity
without modifying clean intervals. Frontal information passing is not yet
implemented or qualified. VMD, neural training, LEMON and confirmation data
were not used in this central experiment.

Recovery confirmed:248/248 payloads verified,3/3 executed code cells,eight
inline figures,four HTML tables and zero error outputs. Kaggle reports COMPLETE.
Scientific execution SHA20584ab is frozen; subsequent documentation commits do
not alter the experiment.

## Frontal SGEYESUB/REG-ICA extension in progress

Readiness passed40 contracts and original author-code runtime checks;52
payloads and actual3/3-cell notebook recovered. Full version2/sourcee8d0021
passed41 contracts and completed paired Klados but failed native p03 input
equality:onefloat32ULP in1/8000 values.359 payloads and actual failed notebook
were verified. Version3/source d87b2e43116c9a7d58dd3adf755ab8df6a0698e1,
runfrontal-spatial-full-002, passed42 contracts including the bounded rounding
repair. It recomputes after Kaggle did not expose the requested failed parent.
Native evaluation and full report retrieval are pending at this entry.

Version2's completed paired part: REGICA pooled-window record-balanced
SNR5.597328dB, PearsonCC0.848019; new calibrated ridge17.488734dB, CC0.952944.
The historical VMD+EOG comparison on the exact same scoring windows remains
16.134624dB, CC0.984644. Calibrated and score-window-fitted ridge are distinct.
These are reused development records and do not qualify any method without
the paired clean-input/native preservation review. SGEYESUB has no Klados
paired score because verified six-class calibration annotations are unavailable.
Full-montage fit, frontal-only output scoring; other regions remain deferred.

## Combined frontal/posterior results requested during frontal evaluation

Completed CPU runfrontal-posterior-combined-001, exact source
f94436ab167d3019aacb4ceeadb64775b42d10c0, Kaggle
frontal-posterior-combined-comparison/1.44 contracts passed, no method failures;
84/84 artifacts verified. Actual3/3 executed cells, eight figures, three tables,
zero errors. SNR, per-record and clean-modification figures visually inspected.

Eight reused Klados records,24 identical archived windows, full19-channel
raw/clean targets. Frontal7:FP1/FP2/F3/F4/F7/F8/Fz. Posterior5:P3/P4/O1/O2/Pz.
Central3 and temporal4 remain exactly unchanged. Paper-list row binding is
still an assumption. Fit new ridge0.001 and instantaneous GEVD-MWF rank1,
strength1 on first2000 separately preprocessed samples. MWF uses calibration
EOG activity bottom/top30%, not known-clean labels. No clean-target fitting.
Historical frontal outputs and clean-input controls are verified saved parents.
This is disjoint composition, not explicit frontal-to-posterior message passing.

Whole-cap SNR is calculated from the combined19-channel reconstruction and
target, averaged within record and then equally across records. **No regional
dB values are added.** RMSE below is the declared channel/window/record mean
in source amplitude units, not a separately pooled root of mean MSE.

| Frontal method | Posterior method | Whole-cap SNR dB | RMSE | Clean EEG CC |
|---|---|---:|---:|---:|
| Identity | Identity |2.902146|5.600867|0.860895|
| Identity | Calibrated ridge |2.732457|5.771783|0.856711|
| Identity | Rank1 MWF |2.712865|5.871333|0.857546|
| Historical VMD+EOG | Identity |**13.679530**|**2.076279**|**0.966309**|
| Historical VMD+EOG | Calibrated ridge |13.065156|2.247195|0.962125|
| Historical VMD+EOG | Rank1 MWF |12.510022|2.346744|0.962960|
| Calibrated ridge | Identity |12.898252|2.341121|0.954630|
| Calibrated ridge | Calibrated ridge |12.903361|2.512038|0.950446|
| Calibrated ridge | Rank1 MWF |11.635197|2.611587|0.951281|

Frontal VMD output remains16.134624dB/CC0.984644 under every posterior option.
Posterior identity16.294360dB/CC0.979999; ridge18.163185dB/CC0.964101;
MWF12.701463dB/CC0.967272. Despite ridge's higher posterior mean dB, its
mean posterior MSE31.696201 exceeds identity3.674021 and whole-cap accuracy
worsens. Regional mean logs cannot be combined to infer joint performance;
record/amplitude weighting and poor individual cases matter. Every record is
retained in per_record_region.csv and the record plot.

Paired record-cluster bootstrap against VMD frontal/posterior pass-through:
ridge mean whole-cap difference−0.614374dB,95%CI[−2.127571,0.557839];
MWF−1.169508dB,CI[−2.504891,−0.231968]. These exploratory eight-record intervals
do not establish participant-level generalization; participant identities are
unknown. No claim that either posterior correction improves the system.

Worst-record mean whole-cap clean-input modification: VMD+ridge46.9381%,
VMD+MWF25.2517%, VMD+posterior-identity0%. The VMD+identity arm still fails
the worst-record CC deterioration gate against the declared fully calibrated
ridge comparator(0.012449>0.005). All whole-cap gate rows are false. No
qualified corrective winner, fresh15dB target or native combined evaluation is
established. Native OSF has no reconstruction SNR. These findings support
testing better selective posterior methods, not deploying the current hybrids.

Open [the actual executed combined notebook](notebooks/executed/frontal-posterior-combined-001.ipynb).
All predictions, H/V correlations, fits, controls, tables and figures are under
results/frontal-posterior-20261017/frontal-posterior-combined-001/.

## Accepted frontal SGEYESUB / REG-ICA closeout

**The frontal extension is complete; no corrective method is qualified.**
Accepted run `frontal-spatial-full-002`, source
`d87b2e43116c9a7d58dd3adf755ab8df6a0698e1`, Kaggle
`satyapaladugu/frontal-sgeyesub-regica/3`. It passed 42 contracts and completed
eight Klados records and four OSF participants / 32 native scoring trials.
All 530 artifacts were checksum-verified, including the actual 3/3-cell
executed notebook with eight figures, six tables and zero error outputs.
Native suppression and preservation plots were visually inspected.

SGEYESUB uses original six-class native calibration and a full 58-channel EEG
fit, scored on F3/F4/F7/F8/Fz. It has no paired Klados score: the export lacks
verified calibration sample classes. REG-ICA uses the original extended
Infomax, fractal selector and SRLS routines, with calibration-frozen ICA and
per-trial/window SRLS reset as disclosed evaluation adaptations. It fits 19
Klados channels under the paper-list mapping assumption and 58 native EEG
channels. Only frontal outputs are evaluated in this notebook. This does not
implement inter-region messaging or a separately fitted posterior algorithm.

### Paired frontal comparison on identical archived windows

| Method | Output SNR dB | RMSE | Pearson CC with clean EEG |
|---|---:|---:|---:|
| Historical VMD + EOG model |16.134624|1.791442|0.984644|
| Score-window EOG ridge |15.226231|1.953725|0.982620|
| Calibration-frozen EOG ridge |17.488734|2.510300|0.952944|
| REG-ICA author-routine adaptation |5.665687|6.013513|0.849560|
| SGEYESUB |Not eligible|Not available|Not available|

These are reused development records, not fresh confirmation or a paper SOTA
comparison. Primary SNR pools channel energy within a window before record
balancing. It differs from the mean per-channel dB column. Calibrated ridge's
higher mean dB but worse MSE/CC is retained, not used to force a winner.
Every per-record result, clean-input control and H/V correlation is saved.

### Native OSF frontal comparison

Values are participant-balanced mean absolute Pearson correlations with the
same original HEOG/VEOG references. Resting modification is mean relative RMS;
it is an unpaired preservation proxy, not a neural-loss percentage or SNR.

| Method | Blink VEOG CC | Lateral HEOG CC | Mean rest modification % |
|---|---:|---:|---:|
| Before / identity |0.918516|0.358906|0.00|
| SGEYESUB |0.109069|**0.033802**|25.33|
| REG-ICA |0.337483|0.170647|41.25|
| Historical VMD + EOG model |0.328867|0.214818|63.63|
| Score-window EOG ridge |**0.044528**|0.063976|70.26|
| GEVD-MWF comparator |0.559549|0.301306|61.43|
| BandRouteNet adaptation |0.375989|0.276730|29.13|
| EEGOAR-Net custom-montage fine-tuning |0.889118|0.377380|14.59|
| SVMD-FCM adaptation |0.367851|0.137215|64.55|
| VME-GMETV adaptation |0.367017|0.045383|49.76|

SGEYESUB has the lowest measured lateral EOG association in this comparison.
It does not qualify: worst-participant mean resting modification is
31.149812% against the fixed 1% engineering limit, and covariance preservation
also fails. Mean alpha error is 0.649467 dB; mean beta error 0.472962 dB;
mean relative covariance error 0.084700. The low EOG correlation cannot by
itself establish clean neural recovery. REG-ICA worst-participant resting
modification is 51.573949%. Native EEG has no paired reconstruction SNR.

Exploratory four-participant bootstrap for SGEYESUB versus score-window ridge:
blink absolute-CC difference +0.064541, 95% CI [0.043586,0.084723]; lateral
difference -0.030174, CI [-0.038909,-0.018779]. This describes the recorded
suppression trade-off only; four participants and reused development sessions
do not establish external superiority. Vertical/rest H/V results and all
individual channels/trials remain in the saved tables.

### Failures, recovery and numerical limits

There were no method exceptions in the accepted run. One of 12 ICA calibration
fits reached its 512-step cap (Klados record 14); its scores remain included
and it is not described as converged. The earlier failed run's paired REG-ICA
score was 5.597328 dB, while the accepted fresh recomputation is 5.665687 dB.
Both are preserved; no best-run picking or averaging is applied. Fixed seeds
did not establish bitwise solver reproducibility across the two workers.

Version 2 stopped at a one-float32-ULP archived-input mismatch. The repair
verifies the narrow rounding condition, logs each discrepancy, and supplies
the exact archived frontal bytes to scoring. References remain unchanged and
scientific thresholds were not relaxed. Attempted failed-parent mounting was
unavailable, so version 3 recomputed all fits and predictions. Its generated
RESULTS.md contains a generic recovery-reuse sentence that does not describe
this execution; this curated closeout corrects it. The archived original is
left unchanged for checksum provenance. Future reporting code now checks
whether a recovery parent was actually restored.

Open [the actual executed frontal notebook](notebooks/executed/frontal-spatial-full-002.ipynb).
Saved evidence is under results/frontal-spatial-20261017/frontal-spatial-full-002/.
The separate combined run is documented above. Neither run uses LEMON,
AutoVMD search, new neural training or reserved confirmation sources.

Next research shortlist, not selected winners: targeted wavelet ICA from the
published 2025 RELAX work; original author GEVD-MWF with documented delay
embedding; and separately scored central/posterior outputs of the verified
full-montage SG/REG fits. See docs/NONFRONTAL_CANDIDATES_20261017.md for primary
papers, author code, dependency limitations and the preservation-first rules.
