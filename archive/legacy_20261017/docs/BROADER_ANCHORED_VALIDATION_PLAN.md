# Broader validation of the fixed anchored frontal VMD candidate

## Frozen question and constraints

Does the fixed 75% raw/25% VMD frontal gain retain its lateral improvement and
EEG preservation beyond the two development pairs used to design it? Frontal
VMD stays K3/alpha2000, uniform initialization, RMS normalization, tolerance1e-6,
cap2000. Five arms: identity, shared direct ridge, separate-axis direct ridge,
old VMD shared gain, anchored VMD. The same nonfrontal joint ridge method is
used in every corrected arm. No correction/weight/K/alpha search, neural work,
condition-label routing or reserved access.

## Source cohort and exposure

Use the source-only cohort from the immutable completed corpus002/2 and
CCAfull001/1 metadata. Select one original chronological scoring start per
recipient using the archived source ordering, never prediction values. There
are 32 recipients, five donors and five frozen recipient/donor buckets. Keep
their existing role/fold identities. All sources are development-exposed;
the anchored recipe was developed on two of these pairs. Distinguish those
pilot recipients and donors in the report; do not call the other records fresh
reserved confirmation or independent participants and donors interchangeably.

## Module A: original-source recovery

- Inputs: exact corpus, completed source-selection parent, verified two-pair
  recovery and a source-only cohort identity frozen before processing signals.
- Reuse the existing recovery routine after extracting a shared per-pair helper;
  preserve the original bounded audit interface and its count guard.
- Two deterministic interleaved shards, 16 recipients each. No duplicate source
  across shards; union must equal the frozen 32-source cohort.
- Reuse original recovered prefixes/windows only if source, donor, scoring start
  and every consumed byte hash match. Otherwise recover original archive/header
  bytes and preserve preprocessing, units, EOG axes and unscored/scoring splits.
- Save each completed pair and failures. An availability audit can complete with
  `all_sources_recovered=false`; this explicitly blocks corpus construction.
  A missing or changed source cannot be silently excluded.
- Kaggle tests must pass before source work. Local activity is authoring and
  source-metadata/Git/byte inspection only.

## Module B: repaired coupled development corpus

After both recovery outputs are complete and fully retrieved/verified, consume
their versioned artifacts plus original donor native arrays. Build ten records
per pair: idealized clean and blink/lateral/mixed at -5/0/+5dB. Use the established
homogeneous field/shared reference multiplier and whole-trial calibration prefix.
No algebraic recovery of recipient background from old mixed calibration.
Persist generation truth separately from denoiser inputs, all source hashes,
trial/window identities, regional input SNR and low-ocular/reference limitations.
The 320 examples are ten variants of one 5.12-second recipient window per
participant, not 320 independent recordings. Ocular waveforms are reused from
five donors. Report that limited temporal and donor diversity explicitly.
Both recipient and donor identities remain separated across the frozen folds.
The regression-generated-field bias remains and requires native evaluation.

## Module C: five-arm validation

Pin completed repaired corpus parents after full recovery. New EEG requires new
frontal modes; each decomposition is computed once and saved with its input hash.
Compute nonfrontal correction once per record and share it across frontal arms.
Save every prediction, gate decision, mode/residual, convergence and failure.
Allow no fit access to paired scoring truth. Per-record unscored calibration is
permitted; no global recipe selection or tuning occurs in this stage.

Report MSE, temporal RRMSE, Pearson CC, output SNR and improvement separately;
channel-first and mean pooled-window SNR; blink/lateral/mixed and input levels;
frontal/posterior/middle/whole; recipient/donor/fold summaries and source-aware
uncertainty. Report pilot-exposed versus other recipients/donors explicitly.
Never add regional dB scores. Keep all eligible failures and count pass-throughs.
The mode diagnostic table stores measured centres/bandwidths and signed peak
mode/residual associations over +/-20 samples (+/-100ms). Those associations
are internal diagnostics, distinct from ordinary zero-lag paired Pearson CC.
Illustrate the first chronological lateral/0dB record and first frontal channel,
including failed decomposition if present. Injected-field truth appears only
in this evaluator-side figure, never in mode selection or correction.

Engineering limits remain clean modification<=1%, alpha/beta<=0.5dB,
covariance<=0.02, paired CC deterioration<=0.005 versus both direct controls.
Native OSF suppression and resting preservation follow controlled validation;
no native reconstruction SNR, teacher qualification or students from this stage.

## Execution checklist

- [x] Freeze and verify full source-only cohort and shard membership.
- [x] Pass broader module fixtures on Kaggle:330 tests, source4daa3fb.
- [x] Recover both original-source shards on Kaggle and verify all artifacts:32/32 sources,542 payloads.
- [x] Generate and recover the repaired320-example corpus:330 tests,all398 payloads verified.
- [ ] Run and recover five-arm comparison with new fixed frontal modes.
- [ ] Review grouped conditions, exposure, uncertainty and preservation.
- [ ] Validate native OSF without retuning or opening reserved sources.
