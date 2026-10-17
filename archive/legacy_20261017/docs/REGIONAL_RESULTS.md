# Regional integration results — 13 October 2026

Verified full controlled development comparison; not independent or production qualification.

Run `regional-full-001`, notebook `satyapaladugu/vmd-regional-composition-full/1`, source `eadff90f6411ba48e6f8cc133b08c211da95d157`. Completed320examples/32recipients/53arms/320posterior calibration fits, zero posterior fit failures.126tests and19,200 metric parity checks passed. Numerical runtime2736.146906s (approximately45.6minutes).15downloaded summary/test/metric payloads checked against artifact manifest; full waveform retrieval is not asserted.

| Arm | Whole-cap output SNR dB | SNR improvement dB | Pearson CC |
|---|---:|---:|---:|
| K3_calibration_event_gain+ica:full | 8.857765 | 5.682199 | 0.855250 |
| K3_calibration_mode_gain+direct:posterior | 9.094213 | 5.918647 | 0.859318 |
| K3_calibration_mode_gain+ica:full | 9.089350 | 5.913784 | 0.858279 |
| K3_calibration_mode_gain+mwf:signed_context | 8.709587 | 5.534021 | 0.852230 |
| K3_calibration_mode_gain+pass | 8.314914 | 5.139348 | 0.847655 |
| calibration_raw_gain+direct:posterior | 9.337592 | 6.162026 | 0.861007 |
| identity | 3.175566 | 0.000000 | 0.733878 |

The VMD+full-ICA arm retains frontal17.112814dB and posterior13.522890dB but whole-cap9.089350dB. Both region estimates use original EEG and disjoint outputs. Central/shared/unknown rows are passed through; their separate error contribution was not scored as its own scope. Whole-cap scoring includes those available rows. The lower whole-cap score is not evidence that ICA reduced the frontal result; regional SNR cannot be added. A Kaggle-only error/coverage diagnostic is needed before attributing the entire gap to unchanged channels.

Best whole-cap point estimate is calibrated raw frontal+direct posterior9.337592dB. No whole-cap arm reaches15dB. VMD+direct and VMD+fullICA pass the controlled CC deterioration criterion, but fail output SNR. Temporal VMD+ICA and signed MWF additionally fail CC deterioration. Synthetic clean preservation remains zero modification; native preservation is unresolved. Three inherited VMD failed example/channel cases remain scored pass-through. The170failure-labelled whole-cap rows reflect example-wide propagation of those failure labels, not170distinct decomposition failures.

Full source-excluded posterior review gives direct13.541703dB, ICAfull13.522890dB, ICAsigned12.862756dB and MWFsigned12.335080dB. Only direct/fullICA meet the added mean CC deterioration screen among these posterior methods. These are inherited partial-gate recipes without retuning; the review does not establish native or teacher qualification.

Monitoring interruption: pilot finished and launched full automatically. A nonzero Kaggle status API call stopped the local chain while full continued. On13October live status confirmed bothCOMPLETE; full outputs were then retrieved and checksum-verified. This was an observability failure, not kernel numerical failure. No models or confirmation evaluation launched.

Next decision: run a focused Kaggle breakdown of original/corrected errors for frontal, posterior and remaining channel names/masks; also report channel-first versus pooled-energy aggregation. Preserve all input/target/split identities. Select a declared central/shared non-VMD policy only after this evidence and native preservation review. Do not silently relabel unknown anatomy or call this a17dB whole-cap system.

Saved results: `results/vmd-regional-integration-20261012/regional-full-001/results/vmd-regional-integration-20261012/regional-full-001`.
