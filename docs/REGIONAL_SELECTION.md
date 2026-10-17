# Regional candidate selection: matched SGEYESUB and REG-ICA comparison

Status: author sources inspected; numerical comparison not launched. Fitting
scope was presented to the user: full montage, region only, or both. Recommended
primary experiment is full-montage fitting with region-specific scoring; this
is distinct from the completed region-only central/temporal experiment.

## Candidates and reproducibility

Add SGEYESUB and REG-ICA to frontal, central (FC/C/CP) and separately reported
temporal (FT/T/TP) evaluation. Keep historical VMD on frontal channels only.
Keep identity and matched direct HEOG/VEOG regression as controls. Include the
existing MWF baseline only under matching inputs, calibration and scoring.

- SGEYESUB author source: https://github.com/rkobler/eyeartifactcorrection,
  commit2c95b4f46f37670d25399ac0fdd705ae18248b25. The inspected author fit uses
  sample artifactclasses1/2 horizontal directions,3/4 vertical directions,
  5blink,6rest. Four trial-condition labels cannot replace these six sample
  classes. Original OSF reader exposes sample_labels; verify eligible class
  counts inside unscored calibration on Kaggle. Do not filter/interpolate class
  labels as continuous EEG. Reproduce the author solver and verify runtime
  compatibility; a generic subspace projector is not SGEYESUB.
- REG-ICA official distribution linked by SCCN plugin registry:
  https://sccn.ucsd.edu/eeglab/plugins/REG-ICAv1.00.zip,
  SHA256c0f2f6c696517940b2a011e2b53938336f6176ada5aba6ef192172bacc9b5a51.
  This contains the ICA/component-selector and adaptive regression routines.
  Preserve/report adaptive behavior; the completed fixed ridge implementation
  is not this algorithm. Verify noninteractive backend and numerical contracts
  on Kaggle before waveform evaluation. No claim of successful reproduction yet.

SGEYESUB is primarily eligible for native OSF when original sample classes
verify. Klados has no verified matching sample labels, so an exact Klados
reproduction cannot be claimed by inventing labels or consulting clean targets.
A modified-label approach would require a separate explicit adaptation decision.
REG-ICA may be evaluated on paired Klados and native OSF if dependencies pass.

## Fair comparison and information access

Use identical raw/target windows and EEG preprocessing for every method within
a comparison. Record calibration duration, channel inputs, reference access,
annotation use and whether any online adaptation occurs. Frozen-calibration and
adaptive algorithms must be disclosed separately; runtime EEG/EOG adaptation is
not supervised clean-target fitting. No scoring labels or clean targets select
components, parameters or gates.

Historical frontal16.1346dB and newer central18.3343dB come from different
calibration/filtering protocols. They are historical references, not a shared
ranking. Either verify identical saved scoring inputs or rerun matched controls.
Do not add regional dB. Original HEOG/VEOG evaluation references remain unchanged.

## Selection rules: validity, preservation, then efficacy

1. Eligibility: verified source/runtime, correct channel identities and masks,
   unscored calibration, no recipient/participant split violation, all failures
   retained. Ineligible methods remain explicitly not run.
2. Preservation: retain the agreed engineering limits, separately labelled from
   paper metrics. Worst-record/participant mean clean modification<=1%, clean
   alpha/beta error<=0.5dB, relative covariance error<=0.02, paired Pearson
   deterioration<=0.005 versus the declared matched direct-regression comparator.
   Native resting modification is a proxy, not known neural loss. Report each
   gate and comparator explicitly; identity is not a successful denoiser merely
   because its modification is zero.
3. Efficacy: paired output SNR and improvement, MSE/RMSE, temporal/spectral RRMSE,
   ordinary clean Pearson CC. Native blink/vertical VEOG and lateral HEOG signed
   and mean absolute correlations, resting RMSE and spectral preservation. Native
   reconstruction SNR is unavailable without clean targets. Report separate
   outcomes per region/condition and failures; a blink gain cannot hide lateral
   deterioration.
4. Uncertainty: paired method differences on common records/participants;
   use95% cluster bootstrap intervals with seed42 and2000 resamples, record-level
   for Klados (participant IDs unknown), participant-level for OSF. With only
   four native participants, intervals are exploratory and may be unstable.
   Reused development records are never fresh confirmation.
5. Shortlist: discard no failed cases. Among preservation-eligible candidates,
   retain the nondominated methods across reconstruction and both ocular axes.
   Do not invent a weighted score or force one winner across trade-offs. Keep
   at most two candidates per region; if more remain, prefer lower measured
   CPU runtime after equivalent accuracy, then lower memory. A practically
   equivalent paired SNR difference is<=0.1dB, but does not override CC/native
   trade-offs or uncertain differences. If more than two materially distinct
   trade-offs remain, report unresolved rather than inventing a tie breaker.
   If no correction qualifies, retain pass-through as working default and
   report no qualified corrective winner.

## Execution and closure

All numerical tests, author-code parity, signal processing, evaluation and plots
run on Kaggle. Modules are in Git; notebooks pin published SHA and parent
versions. Download actual executed cells, predictions, operators, diagnostics,
metrics, figures and checksummed manifests. Freeze a regional shortlist before
independent confirmation. Frontal-to-other-region messaging is a later ablation,
not silently added to this method comparison.
