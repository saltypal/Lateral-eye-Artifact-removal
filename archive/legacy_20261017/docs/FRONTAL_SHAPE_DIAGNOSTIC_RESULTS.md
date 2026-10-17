# Frontal shape failure: frozen-output diagnostic

13 October 2026. The diagnostic completed, and the tested error identity explains
the rejected fixed projection's reconstruction penalty. Retain the original
calibrated full artifact estimate. No correction or recipe changed in this run.

## Exact evidence

- Run frontal-shape-error-001/1, kernel satyapaladugu/vmd-frontal-shape-error/1.
- Source 5d655c7929e6b5417caa2c29c134eb8d3f37ea96.
- Result manifest d0589e2ce9036b0a7b9bc02b595202babb8a8a19aca903d8a5232493e58e4ecf.
- Exact parent manifest 7543c9799eb0911beee1c2c3c5e4c22d1b4fb245c088cbbcb7e5b33c7194586f.
- All17outputs retrieved; all15manifest-listed payloads checksum-verified.
- 243 tests passed,14dependency warnings,19.95seconds.
- 320windows,4160frontal channel rows,32recipients,five ocular donors.
- 640 saved prediction identities,8320archived channel MSE checks and4160DC
  channel checks passed; invalid lag edges, nonfrontal pass-through and saved
  selected-rank bounds also checked. Three inherited failure rows stay scored.
- Maximum absolute channel MSE closure error 2.0679515313825692e-25.
- Numerical runtime18.384022752seconds; runner wall62.067941197seconds;
  peak resident memory258320KiB,CPU. FirstRUNNING16:17:59UTC,COMPLETE16:20:21UTC,
  recovery verified16:20:28UTC. Runtime and API elapsed time differ.

## What the diagnostic measures

Let A be the original full estimated artifact, B the shaped artifact and D=A-B.
The full corrected error is e=x-A-s for fixed input x and fixed paired reference s.
The shaped error is e+D. Therefore

    MSE(shape) - MSE(full) = mean(D squared) + 2 mean(e times D).

The first term measures variation removed from the correction. The second reports
its interaction with the original reconstruction error; it can be positive or
negative, as covered by fixtures. Both are retrospective evaluation quantities.
They cannot fit an inference parameter or establish the neural/ocular content of
D. Sources are low-ocular recipient references with regression-generated ocular
contamination, and the cohort was development-exposed. Unit labels remain the
stored squared signal unit.

## Saved participant-balanced terms

| Term | Kaggle result |
| --- | --- |
| Full original correction MSE | 4.298972132532952e-12 |
| Shaped correction MSE | 3.839487712200015e-11 |
| MSE increase | 3.40959049894672e-11 |
| Discarded-field mean square | 3.032235725690054e-11 |
| Error/field cross term | 3.773547732566658e-12 |
| Signed mean closure residual | 1.3813830962536827e-28 |

These terms are averaged on Kaggle using the same recipient/condition/input
balancing as the frozen parent. Clean controls are excluded from the overall
accuracy aggregate and kept as a separate diagnostic condition. The largest
positive term is discarded-field mean square; the positive cross term further
increases error. All nine contaminated condition/input cells have positive
mean MSE change and positive cross term. This is an exposed-cohort diagnostic,
not a new independent uncertainty or biological-preservation analysis.

## Every condition/input cell

| condition | input_snr_db | full_mse | shape_mse | mse_delta | discarded_field_mse | cross_term | closure_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blink | -5.0 | 4.724716277086463e-12 | 4.55430929315622e-11 | 4.081837665447573e-11 | 3.94711208612688e-11 | 1.3472557932069291e-12 | 3.4127336392022842e-28 |
| blink | 0.0 | 1.0113751008487918e-12 | 3.3100531672951065e-11 | 3.208915657210227e-11 | 3.041557985055203e-11 | 1.6735767215502384e-12 | 4.111785764444289e-28 |
| blink | 5.0 | 7.612124392312467e-13 | 2.2540855084416437e-11 | 2.177964264518519e-11 | 2.010561817357059e-11 | 1.6740244716145993e-12 | 2.8107720869105586e-28 |
| clean |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| lateral | -5.0 | 1.693696247013856e-11 | 7.727100685711834e-11 | 6.03340443869798e-11 | 4.968979532838029e-11 | 1.0644249058599499e-11 | -7.96142698192283e-29 |
| lateral | 0.0 | 5.952642255223183e-12 | 4.2584094272944937e-11 | 3.663145201772175e-11 | 2.8572416177634423e-11 | 8.05903584008733e-12 | -1.5049038807293149e-28 |
| lateral | 5.0 | 4.336525126740917e-12 | 2.2901732303817472e-11 | 1.856520717707656e-11 | 1.3552725227794216e-11 | 5.012481949282343e-12 | -1.4223768937215783e-28 |
| mixed | -5.0 | 2.830513191105037e-12 | 4.485134253012957e-11 | 4.202082933902453e-11 | 4.054105181136152e-11 | 1.4797775276630134e-12 | 3.0098077614586223e-29 |
| mixed | 0.0 | 1.234536576890084e-12 | 3.388958313252934e-11 | 3.265504655563925e-11 | 3.0605770701752954e-11 | 2.049275853886302e-12 | 6.393414228775831e-28 |
| mixed | 5.0 | 9.022657555322897e-13 | 2.2871655312532016e-11 | 2.1969389556999726e-11 | 1.9947137179790066e-11 | 2.022252377209659e-12 | -8.738151565525051e-29 |

## Decision and scope

The projection omitted a contribution that was useful for reconstruction in this
controlled experiment. Its reduction in correction energy did not improve
accuracy. Production-case subtraction/DC/edge checks and paper-MSE parity passed;
there is no evidence here for a sign, unit, channel-axis or baseline-loading error.
This does not exclude every defect, establish that VMD generally fails, or prove
biological neural loss. The rejected operator restricted a calibrated artifact to
a contaminated EEG-derived selected-mode span; mode membership is not a verified
ocular subspace label.

Stop native extraction for this rejected shape variant and retain the stronger
original full-field VMD correction. A further experiment must preserve that
baseline and investigate a distinct frontal/lateral mechanism or independent
contamination benchmark. The original lateral and native preservation deficits
remain unresolved; posterior integration and neural promotion are not qualified.
No AutoVMD, parameter search, field/gain fit, new decomposition, prediction change,
model training or reserved confirmation access occurred. All numerical work ran
on Kaggle. Local work was authoring, static review, hashes and formatting saved
results. See FRONTAL_ARTIFACT_SHAPE_RESULTS.md for the17.112814/18.193247/2.521295dB
comparison, primary paper metrics and uncertainty.

Reproduction from the project root with an unused run ID:

```powershell
python tools/kaggle_campaign.py submit --stage frontal-shape-diagnostic --run-id frontal-shape-error-reproduction-001 --source-sha 5d655c7929e6b5417caa2c29c134eb8d3f37ea96 --profile full --config configs/frontal_replication.json --parent-run frontal-shape-full-001 --kernel-slug vmd-frontal-error-reproduction --kernel-source satyapaladugu/vmd-frontal-artifact-shape/1 --kernel-source satyapaladugu/vmd-eog-research-ready/1 --kernel-source satyapaladugu/vmd-frontal-review-contracts/1
```
