# Original-trial support: verified native frontal results

13 October 2026. The new timing policy improves some preservation measures,
but all four original-trial correction paths remain outside the unchanged gates.
Retain this as a measured boundary ablation, not a qualified cleaning system.

## Execution, tests and recovery

- Main run: native-trial-support-001/1, Kaggle satyapaladugu/vmd-native-trial-support/1.
- Source:0e33d874373ab35d5b4f47400bbf553ab87ef4c8.
- Manifest:df7164091861707a8fc07e0531e18cbd4f3994ad8cc36a40226074916e142f70.
- Exact native parent manifest:63055a6fa7ed3c17b767e6ba64e0f7ad845b5d40af0afe0b78d59da7c83ca049.
- Exact corpus/2 manifest:93da62cb9cec2984ec9687ee27f308d2dbea6dd57abd62541468daafb89d5e0d.
- 224 Kaggle tests passed,14 deprecation warnings,17.74 seconds.
- 306 cached window detector/threshold checks;255 baseline prediction checks;
  40 condition-metric rows;204 reflected/zero-padding cropped-output checks.
- 51 scored trials,9 methods,8 inherited failed-window/channel entries retained.
- All194 manifest payloads downloaded and checksum verified;196 downloads include manifest/state.
- All51 predictions,51 original-support tracks,51 decision bundles and18 resting spectrum archives retained.
- Numerical ablation runtime69.685 seconds; runner312.013 seconds; peak370516KiB.

Primary status polling observed RUNNING at13:52:04UTC and COMPLETE at14:09:22UTC.
Do not confuse platform elapsed time with the saved numerical runtime. The
completed kernel log reports notebook conversion/publication separately.
Luna's independent status monitoring resumed successfully and confirmed completion.
No new VMD, global parameter selection, artifact-field fitting, neural training,
posterior modification or confirmation access occurred.

A parallel dataset-free numerical check used the exact same source: contracts
run trial-support-contracts-001/1, campaign vmd-eog-20261010, Kaggle
satyapaladugu/vmd-trial-support-contracts/1.224tests/14warnings passed in19.69s;
runner23.772s,peak105536KiB;all8payloads verified. Manifest
c3b72443f05c7f24d48be73694f9372285eaab36b6827100c8789b9cff6733c2.
This diagnoses code contracts only; it does not substitute for main-run data parity.

## Precisely what changed

Detector center and thresholds were reconstructed from the exact unscored corpus
calibration. Whole-ledger identity, trial partitions, per-sample labels and joins
were checked. The reconstructed detector reproduced every archived local support
before any new prediction. It fits no scored rest or scoring condition.

Original-trial HEOG/VEOG support is calculated before padding, then aligned to
cached overlapping windows. Original samples receive the same trial support
across overlaps. Reflected versus zero support outside original samples produces
identical cropped predictions. Artifact fields, shared gains, admission0.4,
multiplier1, K3/2000 and preprocessing remain unchanged. Every method estimates
its correction relative to original EEG and subtracts once. Failed fields stay
zero and all nonfrontal channels remain bitwise unchanged.

## Reconstructed detector calibration

All values retain original source-array units; thresholds were not selected on
scored rest. Threshold rows correspond to window STD and95th absolute amplitude.

| Record | HEOG center | VEOG center | HEOG STD threshold | VEOG STD threshold | HEOG95th threshold | VEOG95th threshold | Calibration rest windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| p02 |0.103845|0.110657|7.503800|5.266707|14.778196|10.341028|88|
| p03 |-0.079509|-0.073667|5.149000|5.972492|10.120371|11.470778|99|

The exact centers/thresholds and two calibration-file SHA256 values are preserved
in trial_support_calibration.json. Every local mask matched the original cache.

## Project engineering preservation gates

These supplement the prescribed paper metrics. Worst-record mean resting
relative modification <=0.01; alpha/beta error <=0.5dB; covariance <=0.02.
Resting native EEG is a preservation proxy, not proven artifact-free truth.

| Method | Worst modification | Worst alpha dB | Worst beta dB | Worst covariance | All preservation gates |
|---|---:|---:|---:|---:|---|
|VMD window separate|0.158564|0.220841|0.109023|0.108051|Fail|
|VMD window OR|0.185130|0.360004|0.152970|0.127834|Fail|
|Raw window separate|0.186919|0.249230|0.116666|0.115809|Fail|
|Raw window OR|0.220164|0.403878|0.165964|0.137110|Fail|
|Identity|0.000000|0.000000|0.000000|0.000000|Pass (identity only)|
|Raw trial OR|0.215976|0.405165|0.163458|0.131384|Fail|
|Raw trial separate|0.181440|0.249266|0.116342|0.109662|Fail|
|VMD trial OR|0.182014|0.361792|0.150607|0.122903|Fail|
|VMD trial separate|0.154503|0.221154|0.108621|0.102743|Fail|

VMD separate worst modification changes0.158564→0.154503 and covariance
0.108051→0.102743. Both remain far above0.01/0.02. All active arms pass the
alpha/beta gate, but fail modification and covariance. Identity passing
preservation is not evidence of ocular cleaning. No winner was selected.

## Paper-native unchanged-reference correlation

Complete-condition ordinary absolute Pearson:HEOG lateral,VEOG blink/vertical.
Scoring labels enter reporting only. No paired reconstruction SNR is available.

| Participant/condition | Before | VMD window OR | VMD trial OR | VMD window separate | VMD trial separate |
|---|---:|---:|---:|---:|---:|
|osf:p02/blink|0.913125|0.051475|0.050474|0.062773|0.061809|
|osf:p02/lateral|0.273208|0.024695|0.024691|0.035932|0.035915|
|osf:p02/vertical|0.455667|0.070346|0.070368|0.068313|0.068325|
|osf:p03/blink|0.958671|0.105604|0.107728|0.105517|0.107728|
|osf:p03/lateral|0.422304|0.045239|0.045253|0.041308|0.041242|
|osf:p03/vertical|0.685271|0.144703|0.144535|0.144195|0.144239|

Blink suppression slightly improves for p02 and slightly worsens for p03.
Lateral/vertical changes are small and mixed. This is not uniform superiority;
low EEG–EOG correlation does not establish preservation of clean neural content.
The two exposed participants cannot support population safety or Q1-level
superiority claims. The existing paper-inspired participant permutation table
is saved; no significance claim is made here.

## Resting RMSE and spectra

The following uses complete-condition moments, not mean per-trial RMSE.
Source-array units are unverified; do not label microvolts.

| Participant | VMD window OR | VMD trial OR | VMD window separate | VMD trial separate |
|---|---:|---:|---:|---:|
|osf:p02|1.196495|1.061528|1.087685|0.986531|
|osf:p03|1.710040|1.692576|1.583876|1.563118|

Resting PSD/relative powers are saved per record/method. Independent-trial Welch
uses two-second Hann segments,one-second overlap,constant detrending and1–40Hz
normalization; unspecified settings are declared paper adaptations. Spectrum
aggregation weights each trial by its Welch segment count. Engineering
alpha/beta errors retain their distinct existing calculation and aggregation.
Support disagreement rows use original-overlap samples only; padding-only rows
retain undefined disagreement values and are not interpreted as independent
observations. Original/aligned tracks and window identities are reproducible.

## Decision and next scientific question

The boundary policy has an effect, but does not close preservation shortfall.
Stop promoting timing/admission/attenuation variants as solutions to that failure.
This does not prove the resting corrections are neural loss: native rest can
contain ocular transients and has no clean reconstruction target. Do not relabel
resting trials or relax the agreed thresholds to manufacture a pass.

Next consider a separately declared fixed controlled artifact-shape ablation:
keep calibration field/gain unchanged, but restrict its waveform to the selected
frontal VMD representation. The rejected gain-only projection already changed
gain and still subtracted the full field; do not repeat it under a new name.
Archived controlled modes permit a cheap bounded comparison. Native has only
first-window mode examples, so a full native shape trial requires additional
fixed-recipe decomposition/extraction and must follow controlled evidence.
The stronger calibrated-raw comparator and contamination-construction bias
remain mandatory. No method superiority, integration, teacher or student
qualification follows from this native ablation.

Controlled SNR remains17.112814 frontal and13.162704 whole VMD-context /
13.183201 raw-context with the same scored frontal VMD branch. This native
experiment establishes no new reconstruction SNR. AutoVMD remains stopped.
