# Matched raw versus VMD frontal-artifact context

## Question and fixed scope

The verified 320-example coverage run improved whole-cap output SNR from
9.094213 to 13.162704 dB and central/shared output SNR from 1.954134 to
10.903434 dB. The new experiment asks whether VMD-derived frontal artifact
context helps central transfer more than matched raw frontal-artifact context.
It does not repeat the forty-setting AutoVMD search, choose new K/alpha values,
train a model or evaluate reserved confirmation sources.

The previous raw-frontal comparator has central pass-through and therefore
cannot isolate the representation used for central context. This experiment
adds one raw-context arm; it preserves all six previous outputs as controls.

## Inputs, interfaces and original-input correction

Use the exact source/split/corpus and versioned regional, diagnostic and full
coverage artifacts recorded in `configs/context_match.json`. A fresh checkout
at a published exact SHA runs notebook 50 on Kaggle. Source/checksum, numerical
contract or parity failures block dependent execution.

`fit_context_calibration(..., representation="raw")` selects the same at-most
four nonoverlapping 1024-sample whole calibration chunks as the VMD arm.
Each chunk's estimated frontal artifact is fitted using only the complementary
chunks, with their boundaries supplied. Association .4, shared gain,
lags (-10,0,10), ridge .01 and signed left/right/midline/common/lateral
summaries remain fixed. No raw-path VMD solver or decomposition cache runs.
Valid zero rest chunks remain calibration examples; entirely zero or unusable
context makes the arm unavailable and passes through with recorded failure.

The central ridge fits these cross-fitted raw predictors and unscored central
EEG. At scoring, raw frontal context is the original frontal EEG minus the
archived calibrated raw frontal output. Both context arms retain the same
archived VMD frontal output and direct posterior output. Thus the only new
scored change is central transfer from a different estimated frontal field.
Calibration fitting accepts no paired target. Inference accepts neither
target, true artifact, mixture amplitude nor condition labels.

Both archived scoring frontal fields were originally estimated using their
record's full unscored calibration. The bounded four-chunk OOF policy is used
to fit central transfer for both arms; it is not a claim that the archived
scoring frontal fits used only those four chunks. The calibration-versus-score
fit distinction is disclosed rather than hidden.

Frontal, posterior and central masks remain disjoint. Unknown and masked
channels stay unchanged. The new central output is original central EEG minus
the estimated central artifact; posterior signals receive no VMD.

## Evidence, tests and experiment sequence

1. Publish modules/configuration/fixtures and a standalone notebook.
2. Run all numerical fixtures and a 50-example five-recipient Kaggle pilot.
3. Recover/checksum completion, fits, predictions, decisions and metric shards.
4. Verify exact previous-waveform parity, identical calibration chunks,
   failures, zero-input behavior, anatomy and numerical reporting.
5. After those contracts pass, run the same recipe on all 320 development
   examples; do not select parameters from the pilot.
6. Save channel-first paper MSE, temporal RRMSE, Pearson CC, output SNR and
   SNR improvement, with regional and blink/lateral/mixed/input-strength tables.
   Save recipient/donor-aware paired uncertainty as a project analysis method,
   separately from the paper metric definitions. Analyze the original full
   coverage table on Kaggle without repeating its fits.

Keep failure rows and fit availability in complete-cohort comparisons. Do not
report a favorable successful-only subset as primary. Per-example waveform,
decision and metric shards persist before progress advances. Source SHA,
configuration, exact parents, tests, environment, hardware, runtime and memory
remain in the run artifacts.

If raw matches VMD context, the evidence supports coverage and artifact-strength
context; VMD's additional value remains unestablished. A difference favoring
VMD must be reported with paired uncertainty and development scope. Neither
outcome establishes superiority to published methods or native qualification.

## Stop rules and next checkpoint

The existing frontal CC deterioration .00744148 exceeds the .005 gate. Native
resting modification and covariance remain unresolved. All engineering gates
and the whole-cap 15 dB target remain unchanged. Clean-control pass-through in
this regression-constructed corpus does not establish real-EEG preservation.
Keep `qualification=false`; models and confirmation remain closed in this
campaign. After the attribution experiment, prioritize the planned native
shared-gain/separate-axis-support ablation before further central optimization.
