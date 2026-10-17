# Independent frontal benchmark source audit: completed, no eligible candidate

14 October 2026 IST. IB01's documentary audit completed. None of the eight audited
candidates currently satisfies the complete independent frontal paired-benchmark
contract. IB02-IB04 remain unlaunched. No new EEG accuracy or SNR was measured.
This finding concerns the audited register; it does not establish that no suitable
dataset or model exists anywhere.

## Execution and recovery

- Failed run frontal-source-eligibility-001/1 at36b21a12f8f2c1d46368ca7aa72b8d1acbdc5c5e:
  263tests passed in19.20s, then the EuropePMC HTTP body timed out. Allseven
  failed-run manifest payloads recovered/verified; completion false. Error retained.
- Repaired run frontal-source-eligibility-002/2 at27eeefe3f64417c9c8c8c4294c4ae148cbddd345:
  266tests passed,14dependency warnings,21.41s. All17outputs recovered and all15
  manifest-listed payloads checksum-verified; completion true.
- Repaired result manifest: dfe024b9ab0b83558983276c4e648dfcde41ed7bfe9a1d2e9d3509dc4c2f1299.
- Register hash:8ad9902c689721f06a72feaf82b0a9341ed61b3a01e5c94bf9117d3e4b1b2db7.
- Exact source document:0f7ee5133c6d3b6ac8d39af51ca831c267831410b8793b8e37f615061bcbedc8,
  25540bytes. The unchanged reviewed document is attributed and packaged in the
  published Git checkout; no live publisher HTTP was attempted by the repair.
- Repaired stage documentary runtime0.014528037s; runner wall26.828647139s;
  peak resident memory190996KiB. CPU, no datasets or waveform parents attached.
- First repaired RUNNING21:43:45UTC,COMPLETE21:45:20UTC13October2026;
  recovery verified21:45:28UTC (14OctoberIST).

Tests establish strict requirement states, unknown/absent/closed blocking,
source hash rejection, unchanged grouped table text, malformed document rejection
and pinned path/byte identity. **Other registry states are documentary assessments.**
This job validates their contract structure and the exact supplement, rather than
empirically revalidating every dataset or proving waveform quality.

## Source decisions

| id | paired_target | frontal_mapping | independent_generator | amplitude_coupling | selection_access | eligible |
| --- | --- | --- | --- | --- | --- | --- |
| existing_LEMON_OSF_controlled | verified | verified | absent | absent | verified | False |
| Klados_publisher | verified | unresolved | unresolved | unresolved | verified | False |
| EEGdenoiseNet | verified | absent | verified | absent | unresolved | False |
| native_OSF | absent | verified | absent | absent | verified | False |
| AFFiNE_paired_simulation | verified | unresolved | verified | unresolved | closed | False |
| Ear_EEG_published_sensitivities | absent | unresolved | unresolved | unresolved | verified | False |
| Magdeburg_reserved | absent | unresolved | absent | absent | closed | False |
| EEGEyeNet_local_BIDS_excerpt | absent | unresolved | absent | absent | unresolved | False |

The full eleven-requirement matrix, citations, limitations and blocking reasons
are saved in benchmark_eligibility.csv and benchmark_source_assessments.json.
These states are eligibility decisions, not new evaluation metrics.

Klados remains a legacy paired comparison. The previous Kaggle provenance audit
matched53cropped exports exactly to publisher originals but did not recover
channel row names, physical units or participant groups. Read the recovered audit
in ../EOG_Artifact_Framework/region_aware_research/docs/Recovered_Klados_Provenance.md;
the publication's electrode list does not establish MATLAB row order.

Native OSF supplies actual blink/lateral signals and preservation/suppression
proxies, with no simultaneous clean reconstruction target. The local EEGEyeNet
BIDS excerpt has identified EGI sensors and one subject but no clean paired target
or verified conversion to our frontal region contract. Magdeburg remains reserved.
EEGdenoiseNet retains its separate published-benchmark role and cannot establish
frontal/lateral anatomy from unavailable annotations.

AFFiNE describes independent simulated blink truth, but the publisher reports
that its data are not approved for public release. Its methods inform reference
quality and preservation; they do not give us an accessible paired frontal/lateral
waveform benchmark. [Publisher](https://www.mdpi.com/2306-5354/11/7/707)

The Ear-EEG supplement's TableS3 was checked on Kaggle with its original checksum.
It has Fz/Cz/Oz and grouped C3/4,T3/4,F7/8 rows for blink,vertical andhorizontal
sensitivities. It does not establish an individual signed frontal map and EOG
subtraction convention for our generator. Published linked-mastoid results are
not yet reconciled with the retained LEMON reference. Model directions alone
cannot supply missing geometry or fields. [Primary article and supplement](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.997377/full)

## Consequences and bounded next engineering step

Do not infer signed coefficients, mirrored channels, participant groups or clean
native targets. No independent mixture generator or benchmark is launched. The
original17.112814dB frontal VMD development result and18.193247dB calibrated-raw
comparison are unchanged; neither is reclassified as fresh independent performance.
AutoVMD remains stopped. No model, posterior integration or reserved-source
promotion follows from this audit.

The existing corpus producer separately scales its injected ocular field while
retaining unscaled calibration/scoring reference drivers. This was already
reported and is confirmed by source inspection. A separately versioned reference-
coupling repair is the next practical engineering checkpoint; its protocol is in
FRONTAL_REFERENCE_COUPLING_REPAIR_PLAN.md. Preserve baseline/affine offsets, units,
lag masks and the same multiplier in generated EEG and reference excursions.
Reacquire/check unscored source inputs where missing; do not infer mixed-prefix
truth. This repair retains the regression-generated family and does not resolve
independent-benchmark validity. It may leave existing gain-adjusted output unchanged.

RC01-RC05 are declared, not implemented or launched. No numerical EEG, decomposition,
fitting or training work ran in this source audit. Local activity was source/document
inspection, authoring, hash checks, Git and output retrieval; software tests and
final documentary execution ran on Kaggle. Performance metrics/gates remain unchanged.

## Notebook presentation validation boundary

Kaggle version2 completed top-to-bottom and its saved result tables/document identities were inspected. Kaggle CLI pull returned source cells without execution outputs. The private rendered notebook UI was not visually inspected in this session; open https://www.kaggle.com/code/satyapaladugu/vmd-frontal-source-eligibility/2 while logged in to check that presentation. The verified CSV/JSON results and run completion support this report; no claim of rendered-notebook visual QA is made.
