# Frozen frontal native OSF diagnostic: completed, preservation failed

## Provenance and numerical validation

- Kernel: `satyapaladugu/vmd-native-frozen-frontal/1`.
- Run: `native-frozen-frontal-001`, campaign `vmd-native-frozen-20261015`.
- Exact source: `c48af6a0cda4d6bd905856613a3cf0c01f5bdcd2`.
- Manifest SHA256: `ccc9253ee7b71f676bfacf805c5d7aabf3b4fed3cae147abad707c04ff071111`.
- All744 manifest payloads recovered and independently rechecked locally;
  completed execution and source identity verified. Local work checked bytes
  and read saved reports; all numerical processing/evaluation ran on Kaggle.
- 341 tests passed,14 warnings,24.78s. Stage runtime319.748067258s,
  separate from setup, parent verification and report rendering.
- Two development recordings:study01_p02_prep andstudy01_p03_prep;
  all51 scoring trials,306 overlap-add windows,4284 frontal decompositions.
- Four solver-flagged channel windows retained in the failure ledger and
  handled by the declared raw fallback arm. No AutoVMD, parameter selection,
  posterior changes, neural training or reserved confirmation access.

## Primary native ocular-association results

These are saved participant means of absolute ordinary Pearson correlation
against unchanged publisher references over complete annotated conditions.
HEOG is used for lateral; VEOG for blink and vertical. Lower correlation is
a suppression proxy, not proof that neural EEG was preserved.

| Participant/condition | Original | Raw shared | VMD shared | VMD anchored | Anchored plus raw failure fallback |
|---|---:|---:|---:|---:|---:|
| p02 lateral |0.273208|0.058827|0.024696|0.047653|0.047653|
| p03 lateral |0.422304|0.070063|0.045239|0.055387|0.055387|
| p02 blink |0.913125|0.048049|0.051474|0.043426|0.043421|
| p03 blink |0.958671|0.044100|0.105602|0.071581|0.047265|
| p02 vertical |0.455667|0.065002|0.070345|0.060398|0.060398|
| p03 vertical |0.685271|0.164394|0.144703|0.157586|0.157586|

The anchored hybrid shows lower lateral association than raw shared in both
records. This two-participant development diagnostic does not establish
population superiority. Shared VMD has lower lateral association still, but
also fails preservation; association alone cannot choose a qualified winner.
The solver fallback substantially repairs the p03 blink result, not lateral.

## Resting preservation: decisive failure

Reported engineering values are means over resting trials per record, then
the worst record for each statistic. Different statistics may have different
worst records. Rest is a preservation reference, not proven artifact-free EEG.

| Method | Worst relative RMS modification | Worst alpha error,dB | Worst beta error,dB | Worst relative covariance change |
|---|---:|---:|---:|---:|
| Identity |0|0|0|0|
| Raw shared |0.569527|2.097657|0.967177|0.433347|
| VMD shared |0.456386|1.934405|0.944334|0.392097|
| VMD anchored |0.541158|2.076375|0.967189|0.425106|
| Anchored plus raw fallback |0.541158|2.076375|0.967189|0.425106|
| Unchanged project limit |0.01|0.5|0.5|0.02|

Anchored/fallback relative RMS modification is0.365531 forp02 and0.541158
forp03. These are approximately36.55% and54.12% relative RMS change, not
percentages of neural activity proven lost. Resting RMSE remains in unverified
original array units:3.045708 and3.078259 respectively. Do not call these
values microvolts without source-unit verification.

Every active correction arm fails all four preservation criteria. Identity
passes by construction and removes no artifact. Large resting changes are
consistent with excessive correction outside clearly ocular intervals, but
the present reports alone do not prove the causal mechanism or distinguish
ocular activity remaining in rest from neural contamination of the fit.

## Decision and next diagnostic

`teacher_qualified=false` remains mandatory. The controlled17.804727dB result
does not overrule native preservation failure. No native reconstruction SNR
can be reported because these trials have no paired clean target.

Do not proceed to a deployment or distillation claim. Keep K3/alpha2000 and
the current predictions frozen. Before another correction experiment, inspect
saved rest interventions, raw HEOG/VEOG amplitude and slew, field association,
gain, mode selection, and calibration reference baseline/variance. Reuse the
existing native intervention/event-support work to avoid repeating old failed
gates. A subsequent bounded support adaptation must use calibration/inference
observations only; condition labels remain evaluation-only. It must preserve
lateral/blink suppression while passing unchanged resting limits, with matched
raw regression and identity controls. Neither blind reduction of correction
strength nor lower EEG-EOG correlation alone establishes an improvement.

The notebook and numerical reports completed successfully. This pilot does not
yet contain a dedicated rendered native comparison figure; no new waveform
visual inspection is claimed in this checkpoint.
