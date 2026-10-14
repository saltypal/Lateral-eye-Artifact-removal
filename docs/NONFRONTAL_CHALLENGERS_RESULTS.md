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


### Gain pilot fully recovered; full comparison launched

All123manifest payloads for gain-pilot001/1 verified at01:31:05UTC14October2026.
Full320example comparison nonfrontal-gain-full001/1 submitted to
https://www.kaggle.com/code/satyapaladugu/vmd-nonfrontal-fixed-gain-full
at248a5070131d8049eddc01eede8ebee8465a77bb. Parent full001/1 remains fixed;
no data/recipe change from the pilot. Added evaluator-only participant-balanced
bound tables at this source do not alter inference. Ten fixedarms, no fitted
weights, no frontal change, no new VMD or neural/reserved work. Completion is
pending. Primary agent monitors during the active chat; no unattended heartbeat
or agent availability is promised.


Pilot condition audit:blend0.75 weak blink+5 improves19.272949→19.972946dB,
weak mixed+5 improves18.491068→19.159358dB, while severe lateral−5 worsens
9.376985→8.879075dB. These are matchedpilot-only results. They motivate testing
observable contamination-dependent correction rather than one fixed strength,
but no such rule is yet implemented or qualified. Full fixedcomparison pending.


## Full fixed gain/blend comparison — complete14October2026

nonfrontal-gain-full001/1 completed320examples/32recipients/5ocular donors at
248a5070131d8049eddc01eede8ebee8465a77bb.292tests passed(14warnings,18.12s),
549000archived paper-metric checks. Stage493.397756s,runner557.734671s,
peak1332200KiB. No fitting,target-guided fixedprediction,VMDsolve,models or
reserved access. Manifesta95e33513549efc97379831b05c5a6ccaef4807f1105e546bce4d3b99b7a672e.
Core14payloads and completion verified; full665payload recovery separately pending.
WholePNGvisuallyinspected/readable; notebookHTMLvisualQA not performed.

| Fixed joint nonfrontal correction | Whole mean channel SNR dB | Pooled-window SNR dB | CC |
| --- | ---: | ---: | ---: |
| Original rank2 | 15.155692 | 13.480444 | .960545 |
| Strength0.25 | 9.540135 | 7.214790 | .879123 |
| Strength0.50 | 11.196361 | 9.569044 | .913169 |
| Strength0.75 | 13.629778 | 12.301465 | .945571 |
| Old/rank2blend0.25 | 14.014078 | 12.417302 | .947241 |
| Old/rank2blend0.50 | 14.753609 | 13.316326 | .957174 |
| Old/rank2blend0.75 | **15.370676** | **13.826832** | **.962884** |

The strongest blend has point gain+.214985dB versus rank2, supplemental95%interval
[−.059848,+.502170],498/500validdraws. CCgain+.002339 interval[−.000182,+.005434].
Neither interval supports a robust superiority claim. It passes whole controlled
engineering gates against rank2, but native/fresh qualification remainsfalse.
Frontal remains17.112814dB/CC.973700, with its known.007441CCloss against strongest
raw frontal. A matched raw-frontal+sameblend ablation was not run; old raw+rank2
cannot substitute for that matched comparator. No VMDbenefit or deployment claim.

Middleblend0.75mean14.588935/pooled14.575971 versus14.141899/14.341411; posterior
14.371630/13.671148 versus14.095698/13.279070. Both subgroupmean15gates fail.
Weakblink+5 rises17.562110→18.492344 and weakmixed+5 rises17.328050→18.194064dB.
Severelateral−5 falls8.050232→7.758159dB,CC.893543→.888087. Lateral0 falls
11.869812→11.702260; lateral+5 rises14.796459→14.819895. The wholeaverage hides
this tradeoff. Retain originalrank2 as reference and blend as a candidate, not an
approved final replacement. No broader strength grid follows these failed arms.

### What the evaluator-only bounds establish

Optimistic target-fitted middle scalar correction changes pooledmean14.341411→
16.144333, and oracleblend bound16.629263. These use clean scoring targets and
are explicitly nondeployable. Their numeric values are not achieved inference
accuracy, newmodel results or evidence of generalization. Averaging oracleweights
and applying that average is invalid; the per-window oracle is optimistic.

For severe lateral−5 in true middle channels,current pooled5.444037dB rises only
to5.570377 with optimistic scalar and6.152358 with optimistic old/rank2blend.
Posterior severe lateral pooled9.012374→9.380995 scalar,9.288298blend. Thus scalar
strength cannot rescue these fields in the hardest condition. Residual energy
accounting closes; centered/time-varying error dominates mean-bias energy in the
saved tables. This supports prioritizing artifact-field/coverage diagnosis rather
than another DCcorrection or broadgainsearch. It does not identify the biological
source of every error, prove an optimal newmethod, or eliminate reference-coupling
and regression-generated-mixture bias.

Next priorities: separately versioned field/reference coupling contracts first;
then distinct lowrank referenceCCA with matched ridge/degeneracy controls and
eligible authorSGEYESUB parity. Gate-missed and unavailable-field cases remain
included. Confidence-dependent gains require observableinputs and grouped source
selection, never condition labels or oracle coefficients. Native suppression/
resting preservation and independent datasets remain mandatory before models.


## Next approach authored: posterior/middle CCA adaptation

Currentuser requested nextapproach. Authored src/vmd_eog/nonfrontal_cca.py,
calibration-onlyregularized canonicaldirectionfit,reference1/2 andEEG1/2arms,
matchedjointdirectridge andoldrank2controls. Reuses21armregioncomposition/reporting
engine. Addedfullreference/ridgeidentity,permutation,units,boundaries,constantrefs,
mask andsavedreplay fixtures. StaticAST/notebookinspection passed locally;
numericalvalidation notyetperformed. Confignonfrontal_cca,numberednotebook62.

Auditconfirms oldsynthetic EEGfield/referenceamplitudemismatch. RC01-RC05 remain
pending; the current nextapproach runs only a matchedlegacyengineeringpilot with
explicitcoupling/coverageaudit. This is an executiondecision under the user's
newdirection, not a claim that repair/nativequalificationpassed. No thresholds
relaxed or reserved/modelwork opened. No authorMATLABparity/iCanCleanreplication
claim. FrontalK3/2000outputs unchanged; allnewfit/outputonlynonfrontal.

Previousgain-full001 resumedrecovery succeeded:all665manifestpayloads andcompletion
verified locally, after interruptedSSL/chunkeddownloads. Failedretrievalattempts
are transportfaults, not numericalkernelfailures. Immutablegainrun preserved.


### CCA pilot submitted

nonfrontal-cca-pilot001/1 submitted toKaggle vmd-nonfrontal-cca-pilot at exact
publishedSHA1a2283151a9c840522ea4df25a239519fefcda81. Pilot50source-balanced
examples/21arms; sixfixedmethods with posterior/central/both outputs. Attachonly
corpus002/2 andmatched-context-full001/1. Notebook62copiedfromactualsubmission
withcorrectSHA/config/parents. Tests precede scoring; completionpending. Existing
Lunamonitor askedtofollowthisrun; primaryagentretainsdiagnosis/fixresponsibility.
NoCCAaccuracy,SNRimprovement,authorparity,repaircompletion orqualifiedteacher
claim beforeverifiedoutputs. Legacyreferenceamplitudemismatch remains explicit.


### CCA subgroup preservation correction authored before claims

Staticreview found firstsource1a22831 derives middle/temporal rows by copying
central score rows. Perchannelpaper metrics retain correctchannelrows, but alpha/
beta/covariance engineeringvalues inherit fullcentral group rather than subgroup.
These are not valid subgroup-preservation estimates if clean correction occurs.
A fix resolves exactsubgroupchannelrows beforepaired/spectral/covariance scoring,
with explicit unknown-anatomy fixture. Sourcefix is authored/notyetKagglevalidated.
Version1 continues for preservedfailure/evidence; retrieve beforeversion2push.
No subgroupgate,VMDbenefit,teacher ormodelqualification claimed fromversion1.
Matchedprimarywhole/frontal/posterior/centralpaper metrics are unaffected by this
reportingfix. Fullcorpus/algorithm/thresholds remainunchanged.


### First CCA pilot complete, primary metric evidence

nonfrontal-cca-pilot001/1 at1a2283151a9c840522ea4df25a239519fefcda81 completed
50examples/5recipients/5donors/5calibrations/21arms.297tests passed,14warnings,
13.41s;900frozen-frontalchecks,30savedfitreplay,85800baselinepapermetricchecks.
Stage87.033244s. Core14payloads/completion verified; full188payload recovery
pending separately. No source/datachange,scoringtargetfit,newVMD,models orreserved.

| Joint nonfrontal method | Whole mean channel output SNR dB | CC |
| --- | ---: | ---: |
| Direct calibrated lagridge | **19.601765** | .983597 |
| ReferenceCCA rank2 | **17.549658** | .976053 |
| Previousreference-rank2control | 16.728031 | .970836 |
| EEGCCA rank2 | 15.163465 | .966923 |
| ReferenceCCA rank1 | 14.345758 | .938454 |
| EEGCCA rank1 | 13.716166 | .932696 |

The17.55CCAresult improves the matchedoldrank2pointmean, but direct19.60 is
stronger and cannot be omitted. CCA does not beat the simplerfullfieldcontrol.
This is the same source-balancedpilot as componentpilot002, distinct from the
sortedrecipientgainpilot. Do not compare pilotfrontal20.176007 with full17.112814
as if a method changed. Direct centralshared21.755970 andposterior16.097518;
CCAref2central17.992342/post15.010884. These are channelmeans, not yet pooledSNR.
Pilotv1 lacks pooledtables; correctedfullsource adds them without replacingmetrics.

Middle/temporalspectral-covariance gates fromv1 remainunqualified because of the
reported subgroupcopybug. Primaryperchannel/wholepaper scores andpairedbaseline
parity are unaffected. New source resolves subgroup rows before preservation
computation, reportsactualcanonicalretainedrank and adds strongest directridge
comparator plus reference-rank2control paireduncertainty. WholeCCA CCloss versus
direct is larger than .005 on pilotmeans; use corrected pairedtable before any
qualificationdecision, not weakcontextcomparator passes. Native/coupled-source
validation remains absent. Numericalfixtures for the revisedreporting run on
Kaggle before fullreplication; no author-iCanClean parity orsuperiorityclaim.


### CCA pilot fully recovered; reporting contracts launched

All188manifestpayloads/completion verified forcca-pilot001/1 afterboundedresume.
Manifest56e6e9349cc605382a4c249ba0b7f6feee617e28143fe02ad82032d674aa7952.
Runner219.862469s,peak657744KiB. Primaryplots/tablemetrics recovered. v1subgroup
engineeringcopylimitation retained; no subgroupqualificationclaim.

Dataset-free contractscca-contracts001/1 submitted atpublished
a9d882c3fc7956c9edfa2f0adceaf7f73367792d to vmd-nonfrontal-cca-contracts.
Tests include exactsubgroupchannelresolution and stronger-calibrateddirect CCgate,
plus297earlierfixtures. Pending numericalresult. Revisedfull320scorewilluse
ownsubgroupspectral/covariance calculation,actualretainedrank,pooledSNR andpaired
CCA-versus-direct/reference-rank2uncertainty. Fixedrecipes/sourcepartitions unchanged.
No duplicatedAutoVMDsearch,weights orconfirmationaccess.


### Revised CCA reporting contracts verified

cca-contracts001/1 at a9d882c3fc7956c9edfa2f0adceaf7f73367792d completed:
299passed,14warnings,17.72s; all8manifestpayloads/completion verified. Runner
21.360177s,peak106072KiB. Fixedsubgroupchannelresolution and strongestdirect
comparator fixture passed; earlier CCAreconstruction,lag/scaling/permutation/
serialization fixtures pass. Fullcomparison mayproceed with unchangedrecipes.
This is a reporting/numerical checkpoint, not native/fresh qualification.


### Full CCA/direct comparison launched after corrected contracts

nonfrontal-cca-full001/1 submitted atlatestpublished
b0c861373a8bc66c4538b78269a3b9b988ce3171 to vmd-nonfrontal-cca-full.
Full320examples/32recipients/21fixedarms; samecorpus002/2 andmatched-context-full001/1.
Numericalcontractsuite precedes scoring. Correctsubgroupspectral/covariancemetrics,
pooledSNR,actualcanonicalranks,strongestdirectridgeCCgate andmatchedCCA/direct/
reference-rank2uncertainty. No algorithmparameter/source split changes frompilot.
Lunamonitor follows status during activecampaign; primaryagent handlesdiagnosis.
The19.601765direct and17.549658CCApilotmeans are not extrapolated as fullresults.
Fullcompletion/recoverypending; native/coupledsourcevalidation unresolved.

### Full corrected CCA/direct comparison completed — 14 October 2026

Run `nonfrontal-cca-full-001/1`, campaign `vmd-nonfrontal-cca-20261014`,
published source `b0c861373a8bc66c4538b78269a3b9b988ce3171`.
[Kaggle notebook](https://www.kaggle.com/code/satyapaladugu/vmd-nonfrontal-cca-full).
Immutable parents: `satyapaladugu/vmd-eog-corpus/2` and
`satyapaladugu/vmd-matched-context-full/1`. The same 320 examples cover
32 recipients, five donors and 32 calibration bundles. All 21 arms completed.
Kaggle tests: **299 passed**, 14 warnings, 20.68 seconds. Verified checks:
5,760 frozen-frontal parity comparisons, 192 saved-fit replays and 549,000
baseline paper-metric comparisons. Stage runtime 986.118150 seconds; total
runner runtime 1,193.665894 seconds; peak memory 2,451,336 KiB.

The table reports output SNR, not SNR improvement. Channel-first means and
within-window channel-pooled energy SNR are separately balanced over source/
condition/input-level groups; the latter is not a new acceptance gate and is
not a single concatenated global-energy calculation.

| Frozen frontal VMD + nonfrontal method | Mean output SNR, dB | Pooled-window SNR, dB | MSE, V² | Temporal RRMSE | Ordinary Pearson CC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Complete calibrated lagged reference ridge | **17.533717** | **14.780705** | 3.546698e-12 | .191564 | .972400 |
| Reference-side CCA rank 2 | 15.833574 | 13.923972 | 4.280153e-12 | .222509 | .965985 |
| Previous rest-whitened reference rank 2 | 15.155692 | 13.480444 | 4.913399e-12 | .242700 | .960545 |
| EEG-side CCA rank 2 | 13.597170 | 12.168657 | 6.092703e-12 | .272894 | .952032 |
| Reference-side CCA rank 1 | 12.981316 | 10.798309 | 1.511775e-11 | .354372 | .928033 |
| EEG-side CCA rank 1 | 12.519167 | 10.576723 | 1.604671e-11 | .369297 | .922790 |

All these joint arms retain identical K3/alpha2000 frontal output. The
17.533717 dB result is an **EOG-assisted regional hybrid on development-exposed
regression-generated mixtures**, not VMD alone, EEG-only deployment, a fresh
confirmation score or evidence of superiority over published methods.
The simpler complete reference field is the strongest of these six fixed
nonfrontal recipes. Removing directions through CCA did not improve it.

| Verified output region | Direct ridge mean / pooled, dB | CCA reference rank 2 mean / pooled, dB | Previous rank 2 mean / pooled, dB |
| --- | ---: | ---: | ---: |
| Frontal, unchanged | 17.112814 / 16.628262 | 17.112814 / 16.628262 | 17.112814 / 16.628262 |
| Middle, 20 channels | **18.652706 / 18.614822** | 15.447017 / 15.613187 | 14.141899 / 14.341411 |
| Temporal/shared, six channels | **21.298006 / 20.412816** | 18.559419 / 17.726674 | 17.452148 / 16.610706 |
| Central/shared, all 26 channels | **19.263160 / 19.193922** | 16.165264 / 16.352900 | 14.905802 / 15.124574 |
| Posterior | **15.350217 / 14.584468** | 14.432034 / 13.602651 | 14.095698 / 13.279070 |

Middle and temporal rows are now scored on their own channel sets before
spectral/covariance calculation. They are not duplicated central/shared
preservation rows. Their larger dB scores cannot be added to frontal/posterior
scores to obtain a cap score.

| Condition | Input SNR, dB | Direct joint ridge output, dB | CCA reference rank 2 output, dB | Previous rank 2 output, dB |
| --- | ---: | ---: | ---: | ---: |
| Blink | -5 | 18.369567 | 16.916620 | 16.124917 |
| Blink | 0 | 19.684750 | 18.792334 | 18.250146 |
| Blink | 5 | 17.914413 | 17.773645 | 17.562110 |
| Lateral | -5 | **13.278714** | 9.212597 | 8.050232 |
| Lateral | 0 | **16.219934** | 12.921279 | 11.869812 |
| Lateral | 5 | **17.940872** | 15.474983 | 14.796459 |
| Mixed | -5 | 17.621888 | 15.960562 | 15.118827 |
| Mixed | 0 | 18.920192 | 17.880395 | 17.300671 |
| Mixed | 5 | 17.853123 | 17.569747 | 17.328050 |

The largest remaining problem is severe lateral movement: even the stronger
hybrid is below 15 dB in that stratum. Weak blink/mixed cases must remain visible;
their improvement does not establish native false-positive preservation.

Matched supplemental uncertainty was computed on Kaggle using 500 paired
recipient/donor Poisson multiplier draws, 498 valid. This is a project
approximation, not the paper's native participant permutation procedure:
CCA reference-rank2 minus previous rank2 whole-cap SNR = +0.677882 dB,
95% interval [.315406, 1.059387]; CCA reference-rank2 minus complete ridge =
**-1.700143 dB**, interval **[-2.548602, -.943181]**. Its CC difference versus
complete ridge = -.006414969, interval [-.011541962, -.002287459].

The CCA rank2 arm fails the strongest new direct-comparator whole-cap CC gate:
deterioration **.006414969 > .005**; the verified middle deterioration is
**.012577343 > .005**. Posterior CCA mean 14.432034 also misses the 15 dB gate.
Passes against weaker context controls must not replace these comparisons.
Complete ridge passes the controlled clean-preservation checks against the
old context controls; zero reported clean modification/alpha/beta/covariance
error concerns idealized synthetic clean inputs only. A self-comparator row
for ridge has zero CC loss by construction and is not independent qualification.

The unchanged frontal output still loses .007441477 CC against its archived
stronger calibrated raw-frontal control, exceeding .005. This experiment does
not provide the new complete-ridge nonfrontal system paired with a raw-frontal
output; that matched frontend ablation remains to be evaluated. Previous native
frontal resting-preservation failures also remain unresolved. Qualification,
teacher promotion, neural training and reserved confirmation access remain false.

Gate tables retain 170 whole-cap failure-flagged channel rows (39 frontal,
53 posterior and 78 central/shared). These include inherited historical flags;
they are not 170 new CCA solver failures or 170 failed records. All eligible
rows remain scored. The summary distinguishes fixed output parity from active
algorithm faults. Coverage/true-artifact/generation-multiplier tables are
evaluator-only after predictions and never select components or gain.

Core outputs, tests, plots, parent/source identities, gate tables and matched
uncertainty were checksum verified. Manifest SHA256:
`c8418eb6acc9977aab73fa7606bfd5326dcdf0f12b056922fdabed1816bc91eb`.
Bulk recovery completed: all **1,058 manifest payloads**, completion and source
identity are verified locally in the retrieval receipt. The saved run directory
contains 3,575,310,609 bytes including logs/receipt. The whole-cap PNG was visually
inspected and is readable. HTML visual QA was not performed.

Decision: retain complete calibrated nonfrontal reference ridge as the stronger
development candidate, preserve the negative CCA/rank-one/EEG-side results, and
stop this fixed CCA recipe rather than launch a blind rank/threshold search.
Next checkpoint is RC01-RC05 reference-coupling/provenance repair followed by
matched fixed frontal controls and native qualification. An eligible author-code
SGEYESUB challenger still requires the six calibrated movement/rest labels;
they cannot be guessed from HEOG signs. No claim that coupling repair guarantees
an SNR gain. No AutoVMD search, neural weights, new dataset or reserved source
was used in this completed comparison.

## New complete-ridge hybrid: matched frontal contribution review

14 October 2026. Run `new-hybrid-frontend-review-002/2`, published source
`0b74d48a20ceeaef17af1b2d7eadd3894f1ff87a`, campaign
`vmd-new-hybrid-review-20261014`.
[Kaggle](https://www.kaggle.com/code/satyapaladugu/vmd-new-hybrid-frontend-review).
Immutable completed parents CCAfull001/1 and matched-context-full001/1.
The same320examples/32recipients/five donors compare nine frozen-output arms.
No fit, VMD experiment, neuralweights, parameter search or confirmation access.
313tests passed,14warnings,18.02s. Parent-array checks1,600; baseline paper
metric checks549,000. Stage657.393049s; runner729.466371s; peak1,424,484KiB.

| Whole-cap arm | Channel-first output SNR,dB | Pooled-window SNR,dB | MSE,V² | TemporalRRMSE | OrdinaryCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| VMD frontal + complete nonfrontal ridge | **17.533717** | **14.780705** | 3.546698e-12 | .191564 | .972400 |
| Raw calibrated frontal + identical complete nonfrontal ridge | **17.777095** | **15.126152** | 3.251705e-12 | .184337 | .974089 |
| VMD frontal + CCA reference rank2 | 15.833574 | 13.923972 | 4.280153e-12 | .222509 | .965985 |
| Raw calibrated frontal + identical CCA rank2 | 16.076952 | 14.157095 | 3.985161e-12 | .215282 | .967674 |
| VMD frontal + previous reference rank2 | 15.155692 | 13.480444 | 4.913399e-12 | .242700 | .960545 |
| Raw calibrated frontal + identical previous rank2 | 15.399070 | 13.686979 | 4.618407e-12 | .235473 | .962234 |
| VMD frontal/context control | 13.162704 | 11.268085 | 1.323511e-11 | .338343 | .934641 |
| Raw frontal-context representation control, VMD output retained | 13.183201 | 11.289971 | 1.359897e-11 | .339202 | .934836 |
| Identity | 3.175566 | .000000285 | 1.016074e-10 | 1.148284 | .733878 |

Pooled energy is computed over available channels/time within each window and
then source/condition/input-level balanced. It is not one global concatenated
energy ratio and is not a newly imposed qualification gate. The identity
channel-first mean is not nominal global mixture SNR because spatial ocular
energy is uneven across channels. All figures are output SNR, not improvement.

Posterior/middle output is held identical within each frontend pair. Therefore
these contrasts isolate frontend contribution rather than altered nonfrontal
fitting. The complete-ridge raw control crosses15dB in both summaries on this
legacy development corpus. It is **not a VMD result or fresh qualification**.
The requested VMD research branch remains17.533717/14.780705.

| Whole-cap condition | Nominal input,dB | VMD frontal + ridge | Raw frontal + same ridge |
| --- | ---: | ---: | ---: |
| Blink | -5 | 18.369567 | 18.466502 |
| Blink | 0 | 19.684750 | 19.727035 |
| Blink | 5 | 17.914413 | 17.963001 |
| Lateral | -5 | **13.278714** | **13.752310** |
| Lateral | 0 | 16.219934 | 16.805857 |
| Lateral | 5 | 17.940872 | 18.562202 |
| Mixed | -5 | 17.621888 | 17.739386 |
| Mixed | 0 | 18.920192 | 19.021829 |
| Mixed | 5 | 17.853123 | 17.955736 |

VMD-minus-matched-raw whole-cap SNR delta=-.243378dB, supplemental95% interval
[-.516659,.043339], crossingzero; this does not establish an SNR winner or VMD
benefit. Whole-cap CC delta=-.001689061, interval[-.002925135,-.000493789].
Intervals use the declared paired recipient/donor Poisson approximation, not the
paper-native participant permutation method. Do not equate a higher raw point
mean with proven SNR superiority.

The VMD hybrid passes the whole-cap controlled gates against the stronger raw
control, but the **frontal CC deterioration .007441477>.005 fails**. Whole-cap
averaging cannot overrule that regional failure. Zero synthetic clean changes
remain idealized controls, not proof of native neural preservation. All stages
retain qualification=false and native_validation_required=true. Both systems
still have severe-lateral mean below15dB. No model/teacher promotion follows.

The repaired experiment changes only parity-reporting scope handling. Failed
version1 retained all320savedexamples,306passedtests and a strict cohort error;
all651manifest payloads are checksumverified locally, state remainsfailed.
The focused repair run passed313tests,all8payloads/completion verified before
version2. Version2 consumes the original completed parents, not the failed run;
its repair-parent field is lineage only. No gate/array/split/recipe was relaxed.

Version2 manifest:
`67b36773a42d80ea71e8d3159f5814a4063edbc1a60756f3c9c9e4153cb72d23`.
All669manifestpayloads, completion/source identity and tables verified locally.
The run directory contains554,272,516bytes including logs/receipt. The whole-cap
PNG was inspected: labels/axes are readable, but the legend overlaps upperbars;
publication figure layout still needs adjustment. CSVtables are authoritative.
No HTML visual-QA claim.

Next: generate the separately versioned two-pair coupled corpus from the
verified original sources and tested homogeneous field, then recompute fixed
frontal-only modes on new input hashes. Keep complete nonfrontal ridge matched
in both frontal controls and evaluate native preservation before model work.
If a challenger is needed, test at most one VMD-informed temporal confidence/
selective reference-field correction justified by lateral and clean-error
diagnostics; no condition/clean-truth oracle may choose the inference branch.
