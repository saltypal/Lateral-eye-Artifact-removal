# Fixed frontal correction feasibility: verified native results

13 October 2026. No uniform fixed recipe in the declared bank passes both
exposed pilot recordings. This is a negative finding for this bank, not every
possible VMD repair. No recipe was selected or promoted.

## Run and recovery

- Run: `native-bounded-feasibility-001/1`.
- Kaggle: `satyapaladugu/vmd-native-bounded-feasibility/1`.
- Published source: `4719e529130f57b08871c66edb73c5dfca1796be`.
- Manifest: `2e784392b157558593f4c94327ca1701e452929899b4f68170a44e96c7607f63`.
- Exact parent manifest: `63055a6fa7ed3c17b767e6ba64e0f7ad845b5d40af0afe0b78d59da7c83ca049`.
- All122 manifest payloads retrieved and checksum verified;124 downloads include manifest/state.
- 215 Kaggle tests passed,14 warnings,14.76 seconds.
- 51 trials,255 baseline waveform checks,40 baseline condition-metric rows.
- Runtime207.616 seconds; runner235.729 seconds; peak471808KiB.

Twelve association/strength settings, each matched raw/VMD and OR/separate:
48 active arms plus identity. K3/alpha2000, calibration coefficients/shared
gains, original supports, preprocessing and overlap-add remain frozen.
No new decomposition, fitting, training, posterior correction or confirmation.
Nonfrontal rows are bitwise unchanged. All51 full-trial prediction bundles
and51 decision bundles remain recoverable. Primary monitored every45seconds.

## Engineering preservation and the trade-off

These are unchanged project gates, separate from paper-native metrics.
Worst recording mean resting modification must be <=0.01; alpha/beta <=0.5dB;
covariance <=0.02. Rest is a preservation proxy, not proven artifact-free EEG.

| Representation/support | Association | Multiplier | Worst modification | Worst alpha dB | Worst beta dB | Worst covariance | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| VMD OR |0.4|1.00|0.185130|0.360004|0.152970|0.127834|No|
| VMD separate |0.4|1.00|0.158564|0.220841|0.109023|0.108051|No|
| VMD OR |0.8|0.25|0.015416|0.003953|0.001880|0.019189|No: modification|
| VMD separate |0.8|0.25|0.015435|0.002652|0.001389|0.019245|No: modification|
| Raw separate |0.8|0.25|0.016205|0.002781|0.001456|0.020071|No: modification/covariance|
| Identity |—|0|0|0|0|0|Preservation only|

All48 active arms fail at least one gate. Identity passing does not demonstrate
cleaning. The selected rows above describe a fixed endpoint and the original
controls, not a chosen winner. The complete bank is in bounded_candidates.json,
bounded_native_rest_trials.csv, bounded_native_rest_records.csv and
bounded_native_gates.json. The figure bounded_native_preservation.png displays
all active candidates and unchanged thresholds.

## Paper-native suppression evidence

Verified source-ledger labels enter reporting after correction. Ordinary
absolute EEG–VEOG correlation is reported for blink/vertical and EEG–HEOG for
lateral, using complete-condition concatenated moments and independently
processed trials. These are suppression proxies, not clean reconstruction SNR.
No significance/winner claim is made from two development-exposed participants.

| Record / condition | Before | VMD OR0.4/1 | VMD separate0.4/1 | VMD OR0.8/0.25 | VMD separate0.8/0.25 |
|---|---:|---:|---:|---:|---:|
| p02 blink |0.913125|0.051475|0.062773|0.887296|0.885933|
| p02 lateral |0.273208|0.024695|0.035932|0.273208|0.273208|
| p02 vertical |0.455667|0.070346|0.068313|0.455667|0.455667|
| p03 blink |0.958671|0.105604|0.105517|0.946956|0.946948|
| p03 lateral |0.422304|0.045239|0.041308|0.418324|0.418317|
| p03 vertical |0.685271|0.144703|0.144195|0.625357|0.624641|

The conservative endpoint largely loses blink/lateral suppression while still
failing preservation. Raising one montage association threshold cannot distinguish
all legitimate ocular correction from remaining resting interventions here.
This does not identify the cause as neural loss: unpaired resting EEG has no
clean truth, and trial37's transient cannot be relabelled to force a pass.

Resting RMSE (complete-condition moments; original source units unverified):

| Record | VMD OR0.4/1 | VMD separate0.4/1 | VMD OR0.8/0.25 | VMD separate0.8/0.25 |
|---|---:|---:|---:|---:|
| p02 |1.196495|1.087685|0|0|
| p03 |1.710040|1.583876|0.313587|0.314365|

Do not confuse these with arithmetic means of per-trial RMSE in the engineering
rest-record table. The concatenated and trial-mean aggregations are distinct.
No microvolt or native reconstruction SNR claim is justified.

## Decision

Stop promotion of this bounded uniform admission/attenuation repair. Retain all
results. Do not extend the bank until a distinct mechanism and matched design
are declared. Any later controlled selection tests its entire frozen bank;
scored native rest cannot prune or select global candidates.

The controlled metrics remain17.112814 frontal VMD and13.162704 whole VMD-context
versus13.183201 raw-context (same scored VMD frontal branch). Whole15dB and
controlled frontal CC qualification still fail. No regional/neural/export
qualification follows. The next frontal question must concern correction
selectivity or boundary behavior rather than another K/alpha search.
