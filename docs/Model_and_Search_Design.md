# Model and search design

The active experiment is a bounded feasibility study. It cannot certify complete EOG removal, unseen-montage accuracy, streaming safety, or superiority over the published/previous notebook. Those require the full research-contract gates.

## Inspectable processing path

EEG has shape channels by time. Filter and resample each continuous trial separately; no joining subjects/trials or filtering labels. Classical calibration uses the first 2,000 samples (10 seconds at 200 Hz), excluded from scoring. The smoke score is a subsequent 1,024-sample segment. The scored clean target never calibrates ICA, ASR, EOG regression or FCM.

The development search fits FCM on two training records and two declared channels per record. It then evaluates two development records, on those same channels, across K=3..10 and alpha={250,500,1000,2000,4000}. Correction strengths={0.25,0.5,0.75,1} reuse each computed artifact estimate: 40 decompositions configurations, 160 correction candidates. Identity/preservation is tested by feeding paired clean EEG through the already fitted correction. Feasible settings must improve development RMSE while clean-relative modification stays <=20% and alpha/beta errors <=1 dB. These thresholds are predeclared engineering guardrails, not final-study noninferiority margins. Raw tables, failures and all center diagnostics are retained.

The first ICA search used thresholds={0.2,0.3,0.4} with full component subtraction. Every candidate failed its clean-signal guardrail. The second development search therefore tests Picard and extended Infomax, thresholds={0.2,0.4,0.6,0.8} and strengths={0.25,0.5,0.75,1}. This change follows development preservation evidence, not held-out scores. ASR cutoffs={10,20,30}; EOG-reference regression penalties={0.1,1,10}. Spatial settings use minimum development RMSE among preservation-feasible candidates. The EOG-reference regression baseline remains explicitly available even if its counterfactual clean-EEG test fails; it is labeled a comparator, not automatically a safe teacher.

For each held-out montage, ICA fits all valid channels jointly. Candidate ocular sources get VMD refinement. The temporal VMD/FCM branch and ICA-source/VMD branch estimate residuals against the same raw EEG, and a convex mixture is subtracted once. Regional VMD weights 0.8 frontal, 0.2 posterior and 0.5 central/unknown are fixed hypotheses. They are not tuned from OSF Protocol Z. Since Klados electrode order is unverified, Klados uses only the shared 0.5 fusion; anatomical training/tuning is blocked until channel provenance is recovered.

## One variable-channel model

### OSF audit correction

The Kaggle audit found 8-second original OSF trials (1,600 samples at 200 Hz). A within-trial 10-second calibration is impossible. The OSF smoke comparison instead excludes the first five entire trials from scoring. Regression/ICA use these independent input samples, with each 1-Hz ICA fit copy filtered separately before pooling; no temporal operation crosses a trial join. ARMBR is fitted on individual calibration trials with its own input-only detector. ASR is unavailable when no independent continuous calibration trial reaches the predeclared ten seconds. Later trials use one genuinely annotated 1,024-sample interval per rest/horizontal/vertical/blink condition, identical across methods and with real event counts recorded. Selecting this evaluation interval does not supply labels to a cleaner.

Raw EOGL1..3/EOGR1..3 electrodes have type EEG in the source but are eye channels. They are excluded alongside derived HEOG/VEOG/REOG and retained separately. Spatial fitting prefers the verified bipolar HEOG/VEOG references. Named regional comparisons therefore use actual EEG electrodes only. Rest spectral stability compares contiguous annotated rest intervals with raw EEG; it is a proxy, not a clean neural target. Disjoint rest samples are never concatenated for a spectrum.

If the VMD feasibility guard fails, spatial alternatives are still evaluated and the student may train directly against paired targets. A rejected VMD teacher never supplies pseudo-targets or blocks the entire alternative-method search.

`student.py` implements one offline model for the combined artifact target `contaminated - clean`, rather than a separate blink model and lateral model. VMD/ICA can provide privileged training residuals. Its runtime input is EEG only, with channel masks and optional verified region IDs. It is a student of the VMD hybrid rather than a claim that a neural layer literally runs ICA.

Input tensors: EEG `[B,C,T]`, valid-channel mask `[B,C]`, region IDs `[B,C]`. Per-channel normalization, a shared two-layer temporal convolution encoder, feature LayerNorm, masked mean/variance spatial context, and a shared BiGRU produce per-channel residuals. A bounded router mixes two learned temporal/context heads, multiplied by an amplitude gate. Regional prior values are hypotheses; the gate/router are not calibrated probabilities. Padded channels are masked before encoding/pooling, and their inputs pass through unchanged. Region metadata must move with channels during permutation.

Parameter count is independent of C. Runtime and activation memory grow with C, and measured accuracy on larger/unseen caps remains a separate requirement. No transformer is required by VMD or this model. BiGRU, centered convolutions, interpolation and zero-phase preprocessing all use future information, so this implementation is explicitly offline. A causal student is a separate future phase, not a switch applied to this checkpoint.

The training smoke profile uses at most six training, two validation and two untouched test records, one 1,024-sample window each and 20 epochs. Loss combines normalized waveform error, derivative/spectral error, clean-EEG identity loss, and low-weight teacher residual distillation when the teacher improves training reconstruction. Teacher quality uses training targets only. This small profile checks training/checkpoint/data contracts; it is too small to make final scientific claims.

## Required progression

Kaggle numeric contract tests -> every-file dataset audit -> development classical feasibility -> frozen Klados/OSF smoke checks -> inspect preservation and method failures -> neural feasibility -> full grouped five-fold, three-seed validation -> participant-purged OSF adaptation -> separately validated streaming experiments.

Original notebooks and DOCX remain the historical reference. Failure at a gate must be investigated; no failed configuration is silently reported as a successful alternative.
