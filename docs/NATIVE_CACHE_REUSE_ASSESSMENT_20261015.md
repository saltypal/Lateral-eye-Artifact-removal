# Native OSF input-cache assessment

This is a source-metadata assessment while broader controlled validation runs.
It is not new native evaluation, waveform validation or teacher qualification.

## Located historical source

- Published parent: `satyapaladugu/vmd-band-native-protocol/1`.
- Run: `native-repaired-002`, campaign `vmd-bandroute-20261010`.
- Source: `4533b143a46e2718ae0b7652678ef8a8a866a20d`.
- Completed native report:36 recordings,846 scoring trials,31 declared global
  participant IDs across the existing five development buckets.
- Source ledger:only development roles; no calibration/scoring trial-ID overlap
  in any record. These are annotation/identity checks, not waveform tests.
- Archived source preflight:36 records passed;375 original files plus one
  supplement were verified. Preserve those original source identities.
- The artifact manifest lists846 `native_trial_predictions/*.npz` files.
  Original EEG, inference references and unchanged published evaluation
  references were saved alongside historical predictions by the pinned code.
- The historical report contains326 method failures. Those are historical
  evidence; input availability does not turn those methods into qualified
  baselines. Do not reuse their cleaned outputs as new algorithm inputs.

## Compatibility evidence and remaining checks

Static Git comparison found no change to `signal.py` between that source and
the current campaign. Main sampling/window/hop definitions remain200Hz,
1024 samples and512 samples. Physical units of native EEG remain unverified;
do not label it volts or microvolts by analogy with MNE-loaded LEMON EEG.

Only selected metadata payloads have been downloaded and verified in this
assessment. The full2213-payload native parent has not been recovered locally.
Do not represent the partial metadata receipt as complete waveform verification.

Before consuming any cached trial on Kaggle:

1. Pin the exact manifest and source/run/configuration identities.
2. Match the complete source ledger to original corpus calibration records,
   partitions, source hashes, references and trial IDs.
3. Verify checksums of every consumed trial and calibration payload.
4. Require every eligible scoring trial, including rest; reject omissions,
   duplicate trials, calibration trials or reserved roles.
5. Check shapes, finiteness, channel order, metadata and unchanged evaluation
   reference arrays. Confirm the archived identity output equals original EEG.
6. Feed only original EEG and allowed inference/calibration references to the
   newly frozen method. No historical correction, condition label or paired
   reference may drive denoising.
7. Preserve full-trial overlap-add and independent trial boundaries. Compare
   native ocular associations and resting preservation; never reconstruction
   SNR against the contaminated original EEG.

## Execution decision

Finish and assess the broader fixed-frontal comparison first. Then implement
an explicit input-cache adapter and a small native pilot under the same frozen
correction parameters. Reuse of archived raw inputs is an input-pipeline
adaptation, not a replication of historical cleaning results. Full native
coverage and participant-level reporting remain subsequent qualification work.
No native fitting, new VMD, student training or reserved access occurred in this
metadata assessment.
