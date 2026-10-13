# Posterior and central component correction: verified development pilot

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
