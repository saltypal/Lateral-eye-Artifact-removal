# Posterior and central component correction: pilots and full development results

14 October 2026 IST. New numerical evidence, not a final winner or qualification.

## Execution and recovery

Run nonfrontal-pilot-002, Kaggle notebook
https://www.kaggle.com/code/satyapaladugu/vmd-nonfrontal-challengers-pilot (version2),
source472b21744e8f92d12241cd9f1b08fab9221e68fa. Campaign
vmd-nonfrontal-challengers-20261014. Fixed parents corpus002 and matched-context
full001; exact configuration/source/split and each consumed array verified.

Completed50examples, five recipient participants, five donors, five calibrations,
21arms. All281tests passed (14warnings,22.45s),900frozen-frontal parity checks,
30saved-fit replay checks and85,800archived primary-metric checks passed. All
186manifestpayloads recovered and SHA256verified, including50prediction bundles,
per-example decisions/metric shards and allfivefit bundles. Manifest:
081dfc70e5b5716d25e245fa050cc711184f8f944accca51d1d388761bc3f971.

Numerical-stage runtime288.771624s; runner443.513476s, peak630416KiB. Bootstrap,
Kaggle publication and local retrieval are additional. No newVMD solver,
AutoVMD, modeltraining, targetfitting or reserved-source access occurred.
Allfivefit records report empty failure dictionaries. Inherited failure-labelled
rows remain scored (56whole rows per main candidate); these are not newfit failures.

## Matched whole-cap pilot means

All scores below use the identical five-recipient cohort, channel-first and
condition/input-balanced recipient means. Do not compare them directly with the
32-recipient frontal17.112814dB or historicalKlados17.92721dB.

| Arm | Output SNR dB | Temporal RRMSE | Pearson CC |
| --- | ---: | ---: | ---: |
| Identity | 3.476403 | 1.074185 | 0.749180 |
| Archived raw-frontal-context hybrid | 13.938976 | 0.295134 | 0.945204 |
| Archived VMD-frontal-context hybrid | 13.994230 | 0.294122 | 0.945402 |
| ICA rejection: posterior | 14.160985 | 0.290109 | 0.946424 |
| ICA rejection: central | 13.840497 | 0.280794 | 0.949574 |
| ICA rejection: both | 14.007252 | 0.276780 | 0.950596 |
| ICA component ridge: posterior | 14.336394 | 0.287379 | 0.946765 |
| ICA component ridge: central | 16.152036 | 0.228012 | 0.962958 |
| ICA component ridge: both | 16.494200 | 0.221268 | 0.964321 |
| Wavelet ICA: posterior | 14.098067 | 0.293110 | 0.945475 |
| Wavelet ICA: central | 13.821718 | 0.290206 | 0.946944 |
| Wavelet ICA: both | 13.925556 | 0.289194 | 0.947016 |
| Selective wavelet ICA: posterior | 14.101217 | 0.293052 | 0.945483 |
| Selective wavelet ICA: central | 13.733664 | 0.293118 | 0.946410 |
| Selective wavelet ICA: both | 13.840651 | 0.292048 | 0.946491 |
| Reference rank1: posterior | 13.727321 | 0.312662 | 0.938824 |
| Reference rank1: central | 13.834942 | 0.342751 | 0.930908 |
| Reference rank1: both | 13.568034 | 0.361290 | 0.924330 |
| Reference rank2: posterior | 14.345661 | 0.286020 | 0.947429 |
| Reference rank2: central | 16.376599 | 0.208992 | 0.968808 |
| Reference rank2: both | 16.728031 | 0.200889 | 0.970836 |

Frontal output is unchanged in all18newcompositions. Both frozen context
baselines also retain the same frontal VMD output; the word raw describes the
central context representation, not an EEG-only or raw-frontal-output system.
The retained frontal score on this smaller cohort is20.176007dB; it is not a
new frontal method or improvement over the larger cohort17.112814dB.

## Regional outcomes and mechanisms

| Arm | Central/shared output SNR dB | Posterior output SNR dB |
| --- | ---: | ---: |
| Archived VMD context | 11.121655 | 13.653156 |
| ICA component ridge: both | 15.846273 | 14.768334 |
| Reference rank2: both | 16.333011 | 14.788694 |

The largest gain is central/shared correction. Replacing central output alone
produced16.376599dB whole-cap for reference-rank2; changing posterior as well
produced16.728031. These are composed-waveform scores, not added regional dB.
Rank2 denotes retained reference-field spatial directions after rest whitening;
it is unrelated to VMDK and does not prove one direction is blink and the other
lateral. Component regression retains the full projected reference contribution
within selected ICs instead of discarding the entire selected IC signal.

Wavelet, selectivewavelet and rank1 candidates did not improve the whole-cap
mean over archived VMDcontext in this pilot. Preserve these negatives. They are
declared adaptations, not proof that the original published methods fail.

## Rank2 both: whole-cap condition breakdown

| Condition | Nominal input dB | Output SNR dB | CC |
| --- | ---: | ---: | ---: |
| Blink | -5 | 17.308777 | 0.977078 |
| Blink | 0 | 18.952548 | 0.984134 |
| Blink | 5 | 20.869103 | 0.982391 |
| Lateral | -5 | 8.986898 | 0.913527 |
| Lateral | 0 | 12.794951 | 0.957762 |
| Lateral | 5 | 15.936228 | 0.976332 |
| Mixed | -5 | 17.133986 | 0.981485 |
| Mixed | 0 | 18.246822 | 0.982890 |
| Mixed | 5 | 20.322966 | 0.981923 |

Lateral remains weaker despite overall gain. Nominal input labels describe
mixture construction, not every channel's measured input SNR. Output SNR and
improvement remain separate in the full saved tables.

## Uncertainty and gates

Supplemental recipient/donor Poisson multiplier approximation,500draws,
456validdraws: rank2both minus archivedVMDcontext mean+2.733801dB,95percent
interval[0.134641,4.007387]. Against rawcontext:+2.789055,
[0.140679,4.202778]. Componentridgeboth versusVMDcontext:+2.499970,
[-0.428054,3.898477]. These are small-pilot project intervals, not a replicated
paper-native significance test or an independent final estimate.

Rank2both andcomponentridgeboth pass all reported whole-cap andcentral pilot
engineering gates against both context comparators. Controlled clean change,
alpha/beta and covariance error are exactlyzero. This does not establish native
rest preservation. Their posterior mean remainsbelow15dB. The inheritedfrontal
CC deterioration against strongest rawfrontal is0.005534757, above0.005; it still
fails. Prior nativefrontal rest failures and field-generator bias remain unresolved.
Qualification=false; no teacher/model promotion.

## Failures, repair and presentation checks

pilot001 atf681322:278tests passed/1failed,21.28s; no scoring. contracts002
atd7001c8:279passed/1samefailure,21.26s. The pointwise wavelet zero-sample check
was replaced by a1e-10 vector-relative unchanged-reconstruction check; deliberate
1percent corruption still fails. Example saved calibration-scaled source
reconstruction relative errors1.3460444e-12 and1.2111307e-12 explain why that
check is appropriate. Scientific thresholds were not changed.

pilot002 executed top-to-bottom and saved plots/tables. Whole-cap PNG inspected
locally: labels/axis/15dB target and bars correspond to saved means. Full notebook
HTML has not been visually inspected; stored CSV/JSON/PNG and execution state
were checked. No local numerical processing was performed.

## Persistent evidence and next step

Artifacts are under results/vmd-nonfrontal-challengers-20261014/nonfrontal-pilot-002/
results/vmd-nonfrontal-challengers-20261014/nonfrontal-pilot-002/ . Primary files:
nonfrontal_method_means.csv, nonfrontal_condition_means.csv,
nonfrontal_channel_scores.csv.gz, nonfrontal_controlled_gates.json,
coverage_uncertainty_intervals.csv, nonfrontal_predictions/, nonfrontal_fits/,
nonfrontal_fit_records.jsonl, nonfrontal_summary.json andartifact_manifest.json.
Allper-example/channel metrics and all21arms persist, includingnegative results.

Next: run allsix fixedmechanisms unchanged on320examples/32developmentrecipients,
then groupedshortlist review, original-label eligible native evaluation and
centralGEVD-MWF/SGEYESUB/CCA challengers. Do not select a final system from this pilot.

## Parallel fixed GEVD-MWF pilot: completed, negative result

Run `nonfrontal-mwf-pilot-001`, notebook59,
https://www.kaggle.com/code/satyapaladugu/vmd-middle-posterior-mwf-pilot (version1),
source `6c1a4bf4eefc05df2609246aaaaf2f46093416c4`. Same50examples/five recipients/
five donors and exact input parents. Two fixed ranks, instantaneous full-montage
support, strength1, same0.4reference-field association gate. Three controls plus
six disjoint regional compositions. No grid repetition or scoring-target fitting.

All282tests passed (14warnings,24.42s),300frozen-frontal checks,10saved-fit replay
checks and85,800baseline metric checks. All186manifestpayloads verified, including
all50prediction bundles and fivefits. Manifest
`af46b1de8474cf46f909b3d0a0e1d4a073f59f9556ca5dd67ce669e293c9fb99`.
Stage71.287043s, runner280.045952s, peak453780KiB. Whole-cap PNG inspected;
saved tables/JSON and completion verified. NotebookHTML visual inspection remains
unperformed. Separate pre-run review found the submitted family/config correct;
a future guard against mistyped family labels was authored separately.

| Fixed MWF composition | Whole output SNR dB | Temporal RRMSE | CC |
| --- | ---: | ---: | ---: |
| Rank1: posterior only | 13.696121 | 0.316873 | 0.937612 |
| Rank1: central only | 12.868710 | 0.398675 | 0.917513 |
| Rank1: both | 12.570601 | 0.421426 | 0.909723 |
| Rank2: posterior only | 13.706579 | 0.316188 | 0.936757 |
| Rank2: central only | 12.907576 | 0.383476 | 0.918386 |
| Rank2: both | 12.619925 | 0.405541 | 0.909741 |

Matched frozen baseline remains13.994230dB/CC0.945402. MWFboth rank1/rank2
central scores8.609306/8.692977 and posterior12.725037/12.761755dB. Synthetic
clean preservation iszero, but whole CC deterioration against the VMDcontext
comparator is0.035678871/0.035660532, above0.005. Both fail SNR15 and pairedCC
gates. Stop these two fixed recipes before full replication or teacher promotion.
This does not establish that every MWF configuration or the published method
fails. It establishes that these matched controls are inferior on this pilot.

Reference-fieldrank2 transfer16.728031dB and GEVD-MWFrank2 control12.619925dB are
different algorithms. Rank alone is not a method identity. The new stronger
reference method estimates a calibrated EOG field then rest-whitens/truncates
it and fits output-region gain; the MWF applies an ocular/rest covariance-derived
operator to original scoring EEG. Full reporting retains this information access.

MWF evidence is under `results/vmd-nonfrontal-mwf-20261014/nonfrontal-mwf-pilot-001/`.
The same file names as the first pilot store complete per-channel/condition
metrics, all9arms, decisions and fitted matrices. Negative results remain archived.


## Full nonfrontal comparison — 14 October 2026

`nonfrontal-full-001/1` completed on Kaggle at source
`46171ac7d14715b853e722013edfde2260a86fb6`. The exact matched development cohort
contains320examples,32recipient participants and5ocular donors. There are21arms:
identity and two saved context controls, followed by six fixed mechanisms with
posterior-only, central-only and joint compositions. No frontal predictions changed,
no posterior/central VMD ran, no clean target entered fitting, and no model trained.

281tests passed;5760frozen-frontal parity checks,192saved-fit replay checks and
549000baseline paper-metric checks passed. Stage runtime1103.517074s, runner
1216.063582s, peak2091892KiB. Exact manifest:
`5c0776e2f99374cb5c1fd13aa1076eae4aada28d6551a20ee293a5d6474ef0b2`.
Core tables/status/identity are checksum verified. Full bulk prediction recovery hit
interrupted downloads; a byte-verified resume was added. Complete recovery must
be checked from the receipt, separately from successful Kaggle execution.

| Joint nonfrontal method | Whole mean output SNR dB | Temporal RRMSE | Pearson CC |
| --- | ---: | ---: | ---: |
| Identity | 3.175566 | 1.148284 | 0.733878 |
| Saved VMD frontal-context control | 13.162704 | 0.338343 | 0.934641 |
| Saved raw frontal-context control | 13.183201 | 0.339202 | 0.934836 |
| ICA component ridge | 14.735326 | 0.270078 | 0.952350 |
| ICA rejection | 12.541144 | 0.326524 | 0.937555 |
| Reference-field rank1 | 12.453789 | 0.394229 | 0.918336 |
| Reference-field rank2 | **15.155692** | **0.242700** | **0.960545** |
| Selective wavelet–ICA | 12.333200 | 0.343143 | 0.933246 |
| Wavelet–ICA | 12.396655 | 0.340632 | 0.933751 |

The two context controls above both retain VMD frontal output; their labels refer
to context representation. They are not a raw-regression frontal-output ablation.
The stronger method uses an EOG-calibrated ocular field, rest-covariance whitening,
two retained reference-derived components and scoring-window region gains
estimated from observed EEG versus the calibrated field (never the clean target).
It is a project adaptation, not an exact SGEYESUB/iCanClean reproduction or MWF.

With rank2joint correction, unchanged frontal SNR is17.112814dB; central/shared
rises10.903434→14.905802dB, posterior13.541703→14.095698dB. Regional scores
cannot be added. These summaries average per-channel dB within source-balanced
strata. A frozen-output follow-up explicitly compares pooled whole-window energy
SNR against channel-first averages and splits verified middle from temporal rows.

Whole output SNR for rank2 by condition/input level:

| Condition | −5dB input | 0dB input | +5dB input |
| --- | ---: | ---: | ---: |
| Blink | 16.124917 | 18.250146 | 17.562110 |
| Lateral | 8.050232 | 11.869812 | 14.796459 |
| Mixed | 15.118827 | 17.300671 | 17.328050 |

Weak blink+5 and mixed+5 worsen relative to19.821757 and18.501700dB saved
VMD-context controls. Severe lateral remains8.05dB. A high aggregate cannot hide
these failures. Next correction-confidence/shrinkage work must infer contamination
from observable EEG/EOG and unscored calibration, never from true input-SNR labels.

Supplemental recipient/donor-aware multiplier uncertainty for rank2minus VMD-context
is+1.992988dB,95% interval[0.679036,3.333935],498/500valid draws. This is not the
OSF paper's native participant permutation procedure, independent validation or
proof of superiority over published methods. Full source-aware results remain saved.

Whole controlled preservation/CC gates pass versus the two context controls.
Frontal paired CC deterioration is0.007441477 versus strongest calibrated raw
frontal, above0.005. New native validation is absent; previous frontal native gates
remain failed. Overall qualification remainsfalse. Controlled references are
low-ocular EEG, not proven clean EEG; ocular mixtures are regression-derived.
This is development-exposed exploratory evidence, not a fresh15dB acceptance pass.

Whole-cap PNG inspected. Notebook HTML layout has not been visually inspected.
Complete per-channel tables, all21arms, predictions, fits, failures and conditions
are in the versioned Kaggle outputs and indexed recovery directory. No broad
AutoVMD search, neural stage or reserved confirmation access was restarted.


## Frozen frontend audit launch and failure-lineage correction

`nonfrontal-review-001/1` submitted at exact published source
`f86d37499a2da96a54b073289ea3861588478b37`; full profile with only
nonfrontal-full001/1 and matched-context-full001/1. Notebook60 is copied from
that actual launch and pins the correct full configuration and parent versions.
It adds no fitting and seven matched output arms. Core paired metrics and pooled
SNR audit run on Kaggle, with both middle and temporal subgroup diagnostics.

Static review found a reporting distinction: a raw-frontal composition must not
call the replaced VMD parent's faults active branch faults. Version1's gate
`failure_rows_scored` means conservative historical parent flags, not active
composed-output failures. Metric calculations and arrays are unaffected. A newer
source fix keeps historical flags separate from decision-JSON nonfrontal faults,
retains raw parent's whole-pipeline flags conservatively and declares active raw
frontend failure counts unavailable from these parents. The new lineage regression
fixture requires Kaggle validation before the revised module is used again.
No parent flag was silently dropped, and no qualification follows from this audit.


### Full result recovery confirmed — 14 October2026,00:43:37UTC

The resumed bounded download recovered all1050manifest payloads of
nonfrontal-full001/1; exact hashes and completion are verified in its receipt.
Downloaded outputs1052 include state and manifest. Previous interrupted attempts
remain known; the completed receipt supersedes the partial recovery status.
Predictions, per-record shards, all32calibration fits, full metrics, diagnostics,
configuration and environment are available locally and in immutable Kaggle output.


Latest dataset-free contracts004/1 source6dedb47142dd9d4f37e8b4f743be4497d8f9f7ee:
290passed,14warnings,18.98seconds; all8manifest payloads and completion verified.
Runner22.903515s, peak105988KiB. Includes reconstruction/composition, SNR aggregation,
exact arm lineage and historical replaced-VMD fault fixtures. Numerical tests run
on Kaggle only. This validates fixtures, not new native qualification or a second
full review execution at the revised source.


## Frozen frontend and SNR aggregation audit — complete

Run nonfrontal-review001/1 atf86d37499a2da96a54b073289ea3861588478b37 completed
320examples/32recipients/5donors with7arms,1600exact parent-array checks and
549000archived paper-metric checks.289tests passed(14warnings,19.74s). No fitting,
AutoVMD, neural weights, reserved access or new decompositions. Stage534.824925s,
runner632.100575s,peak1088332KiB. Core tables and identity checksum verified;
665payload full recovery is separately tracked in its receipt.

| Frozen system | Mean channel SNR dB | Pooled-window energy SNR dB | CC |
| --- | ---: | ---: | ---: |
| Old VMD-context regional output | 13.162704 | 11.268085 | .934641 |
| VMD frontal+ICA component ridge elsewhere | 14.735326 | 12.838938 | .952350 |
| VMD frontal+reference rank2 elsewhere | **15.155692** | **13.480444** | .960545 |
| Raw frontal+same ICA component ridge | 14.978704 | 13.035402 | .954039 |
| Raw frontal+same reference rank2 | **15.399070** | **13.686979** | .962234 |

Mean channel SNR averages per-channel dB before source-balanced aggregation.
Pooled energy sums signal/error across available scope channels/time within each
window, then uses the same source-balanced condition/input averaging. Both use
10log10(signal energy/reconstruction-error energy). They are different
aggregations, not invented alternative equations. Regional dB cannot be added.
The pooled metric remains below15dB and is a supplemental aggregation audit,
not a newly imposed engineering gate. The declared channel-first development
criterion is exceeded; fresh-source and preservation qualification remain unmet.

Frontal VMD mean17.112814/pooled16.628262 versus raw18.193247/17.939047dB.
Unchanged nonfrontal rank2 mean/pooled:posterior14.095698/13.279070,
centralshared14.905802/15.124574. Actual20middle channels14.141899/14.341411;
6temporal17.452148/16.610706dB. Therefore the pooled centralshared bucket
obscures a weaker true middle subgroup. Grouping did not change correction.

Whole VMD-minus-raw frontend meanSNRdelta−.243378dB with supplemental95%interval
[−.516659,+.043339]; wholeCCdelta−.001689[−.002925,−.000494]. FrontalCCdelta
−.007441 exceeds preservation comparator tolerance. These contrasts do not
support VMD superiority. Preference for VMD cannot change the reported winner.

The v1 failure_rows_scored counter contains conservative historical parent
flags, not active branch failures after frontend replacement. Revised source
separates historical and nonfrontal fault provenance; its fixtures passed among
290contracts004 tests. This affects interpretation, not calculated waveforms,
SNR,CC or baseline parity. No active failure count is fabricated.

Whole PNG visually inspected/readable. NotebookHTML visual inspection remains
unperformed. The next bounded strength/blend experiment uses saved fields only;
its target-fitted scalar/blend bounds are evaluator-only and cannot drive inference.


### Fixed nonfrontal gain/blend pilot launched

nonfrontal-gain-pilot001/1 atsourcec049d8dca726adb032f9ee45b32353d9164dd272,
Kagglevmd-nonfrontal-fixed-gain-pilot, exact nonfrontal-full001/1 parent.
Ten arms:foursavedcontrols plus three fixedstrengths and three fixedblends.
Pilotfirst50developmentexamples/fiverecipients, frontalVMD unchanged. No
AutoVMD, newcalibration, target-guided inference, reserved access or neuraltraining.
Correct submitted notebook copied to numbered61 with exactSHA/spec. Numerical
suite runs before scoring. Completion and recovery have not yet been established.
Primaryagent monitors directly after monitoring subagent became unavailable from
its usage limit; continuous background monitoring is not asserted.

Frozen review001 manifestSHA256:
90cb1a71ec0a4cbf42b7d619c689fadb5c32df168c761ade4c2cbffd98426861.
Core13payloads verified; full665payload recovery remains inprogress. WholePNG
visually inspected. The full21armparent retains all1050payloads verified locally.


### Frontend audit recovery confirmed

At01:26:21UTC14October2026 the receipt verified all665manifest payloads and
completion for nonfrontal-review001/1. Core and bulk recovery are now complete:
all320raw-frontal prediction bundles,metric shards,paper scores,subgroup tables,
condition tables,oracle-free comparison,source-aware intervals,figures and lineage
limitations are preserved. Manifest90cb1a71ec0a4cbf42b7d619c689fadb5c32df168c761ade4c2cbffd98426861.
Earlier partial-download statements are historical and superseded by this receipt.


## Fixed strength/blend pilot completed

nonfrontal-gain-pilot001/1 atc049d8dca726adb032f9ee45b32353d9164dd272 completed
50examples/5sortedrecipient participants/4ocular donors. This is a deterministic
subset of the full cohort; it differs from the earlier source-balanced50example
component pilot. Compare within this run only, not its baseline20.086786frontal
against earlier20.176007 as if methods changed.292tests passed,14warnings,
18.63s;85200archived paper-metric checks. Stage86.230018s,runner159.251149s,
peak832820KiB. No calibration/weight fitting,AutoVMD,models or reserved sources.

| Fixed nonfrontal arm | Whole mean channel SNR dB | Pooled-window SNR dB | CC |
| --- | ---: | ---: | ---: |
| Unchanged rank2joint | 15.906784 | 13.699583 | .965093 |
| Strength0.25 | 10.268276 | 7.263809 | .883922 |
| Strength0.50 | 11.886580 | 9.650133 | .918388 |
| Strength0.75 | 14.347441 | 12.426323 | .950711 |
| Old/rank2blend0.25 | 14.491577 | 12.355973 | .949559 |
| Old/rank2blend0.50 | 15.268996 | 13.298144 | .960385 |
| Old/rank2blend0.75 | 15.944701 | 13.861136 | .966507 |

The largest fixedblend pilotgain isonly+.037918dB channelmean; supplemental95%
interval[−.679458,+.858902],454/500validdraws. No evidence of superiority or
selected production weight follows. Simple uniform attenuation is inferior.
Middleblend0.75mean14.569999 versus14.246635; temporal17.185124 versus18.197961.
This illustrates subgroup tradeoffs hidden in the wholeaverage.

Optimistic scalar/blend reports use scoring targets after fixedarms exist and
are explicitly diagnostic_target_fit=true/deployable=false. They do not change
any prediction or constitute achievable generalization. Full comparison adds
participant-balanced bound summaries to make gain-versus-shape headroom clear.
Primary scores use unchanged paper equations; clean and CC gates are unchanged.
Core output recovery verified; full123payload recovery tracked independently.
