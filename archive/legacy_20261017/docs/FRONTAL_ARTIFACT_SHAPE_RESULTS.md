# Fixed frontal artifact-shape comparison: rejected challenger

13 October 2026. The experiment completed with valid numerical/provenance checks.
The fixed DC-preserving shape correction failed accuracy. Preserve the original
VMD implementation; do not promote this challenger to native extraction,
posterior integration or teacher/model training.

## Exact execution and recovery

- Kernel: satyapaladugu/vmd-frontal-artifact-shape/1.
- Run: frontal-shape-full-001, campaign vmd-frontal-replication-20261011.
- Source: 1444b4e226b1e879ad46b2fe638c4d77dc26a753.
- Result manifest: 7543c9799eb0911beee1c2c3c5e4c22d1b4fb245c088cbbcb7e5b33c7194586f.
- Parent controlled manifest: 1ab061565aca07fb826d48477087c949720dfa300facad90be3ebe2008876d20.
- Split SHA256: 9109ee9b28ab4faeb259773254211387dc8f53ac1c033064bdcc8399a2dd4f8c.
- All 665 outputs retrieved; all 663 manifest-listed payloads checksum-verified.
- 236 tests passed, 14 dependency warnings, 23.50 seconds in the full run.
- 960 archived baseline prediction checks and 112,320 underlying metric checks passed.
- 320 full-montage prediction bundles and 320 frontal decision bundles retained.
- 320 windows, 32 recipients, five ocular donors, 4,160 frontal channel rows per arm.
- Three inherited VMD failure rows retained in each VMD arm; none silently excluded.
- Stage numerical runtime 163.81937873000004 seconds; runner wall time
  438.324527839 seconds; peak resident memory 344,036 KiB. CPU, no GPU.
- Primary RUNNING observation 15:54:26 UTC; COMPLETE observation 16:02:49 UTC.
  Recovery receipt completed 16:09:40 UTC. API elapsed time differs from runtime.

This is development-exposed controlled evidence. Low-ocular LEMON EEG is not
proven artifact-free. Ocular mixtures are generated from calibration-regression
fields, which can favor regression. Recipient/donor identities remain recorded;
reserved confirmation remains closed. No claim of fresh final generalization.

## Fixed methods and information access

All methods use the same original EEG, references, calibration and scoring windows.
The VMD arms read archived frontal K3/alpha2000 vectors; no decomposition or
AutoVMD search executes. The original arm keeps its complete calibrated artifact
field and VMD-supported gain. The new arm projects the post-gain artifact's
centered variation onto selected centered EEG-derived modes on valid lag samples,
then restores the original artifact mean. No gain, detection threshold, lag,
correction strength or source partition changes. Posterior/masked/unknown rows
are unchanged. Targets enter scoring only; estimates subtract once from input.
The mode span contains contaminated EEG and is not established pure ocular space.

## Participant-balanced paper metrics

Accuracy summaries exclude clean controls and equalize condition/input strata
within recipient, then recipients. MSE is in the squared stored signal unit.
Output SNR is 10 log10(target energy / reconstruction-error energy).
SNR improvement is reported separately. Ordinary Pearson CC is used.

| evaluated_family | mse | rrmse_time | pearson_cc | snr_energy_db | snr_improvement_db | cc_deterioration |
| --- | --- | --- | --- | --- | --- | --- |
| K3_calibration_mode_gain | 4.298972132532952e-12 | 0.19218065978608248 | 0.973700403460646 | 17.112814353691874 | 22.58599547539414 | 0.00744147678203266 |
| K3_dc_shape | 3.839487712200015e-11 | 0.7778027311904521 | 0.6448721278462164 | 2.521294509275763 | 7.994475630978029 | 0.3362697523964623 |
| calibration_raw_gain | 3.0013215964183274e-12 | 0.16031121162925283 | 0.9811418802426786 | 18.19324666065019 | 23.666427782352457 | 0.0 |
| identity | 3.0582390300864297e-10 | 2.307985957107543 | 0.47281897982007526 | -5.473181121702266 | 0.0 | 0.5083229004226034 |

The original VMD score reproduced exactly at 17.112814 dB; calibrated raw gain
reproduced at 18.193247 dB. New shape scored 2.521295 dB/CC 0.644872. Its reduction
in accuracy is not an improvement in EEG preservation. Identity frontal SNR is
-5.473181 dB because nominal mixture levels are not frontal-channel SNRs.

## Every condition and nominal input level

These tables are exported Kaggle results, not locally recomputed summaries.

| evaluated_family | condition | input_snr_db | mse | rrmse_time | pearson_cc | snr_energy_db | snr_improvement_db |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K3_calibration_mode_gain | blink | -5.0 | 4.724716277086463e-12 | 0.19043677175645699 | 0.9776363837785584 | 16.842724357805302 | 28.509512265622874 |
| K3_calibration_mode_gain | blink | 0.0 | 1.0113751008487918e-12 | 0.12526900883004996 | 0.9887920870006466 | 20.062975065057103 | 26.72976310793107 |
| K3_calibration_mode_gain | blink | 5.0 | 7.612124392312467e-13 | 0.10408021156749837 | 0.9916384597668476 | 22.11064245860635 | 23.777430458424018 |
| K3_calibration_mode_gain | lateral | -5.0 | 1.693696247013856e-11 | 0.41748149044276034 | 0.9141254479237557 | 8.954401981614147 | 17.068230530915372 |
| K3_calibration_mode_gain | lateral | 0.0 | 5.952642255223183e-12 | 0.2593893675825281 | 0.9606781945593541 | 13.129188963526648 | 16.243017647884272 |
| K3_calibration_mode_gain | lateral | 5.0 | 4.336525126740917e-12 | 0.18614745465850238 | 0.9754076946106001 | 16.622937802753107 | 14.736766444054428 |
| K3_calibration_mode_gain | mixed | -5.0 | 2.830513191105037e-12 | 0.1965175848201306 | 0.9776549904625925 | 15.992406042976192 | 27.631332723907697 |
| K3_calibration_mode_gain | mixed | 0.0 | 1.234536576890084e-12 | 0.1374750614271053 | 0.9869061333501897 | 19.10044641754046 | 25.73937323352836 |
| K3_calibration_mode_gain | mixed | 5.0 | 9.022657555322897e-13 | 0.11282898698971039 | 0.9904642396932696 | 21.19960609334756 | 22.838532866279156 |
| K3_dc_shape | blink | -5.0 | 4.55430929315622e-11 | 0.8447030391444695 | 0.5586167513619831 | 1.6546012782622403 | 13.321389186079813 |
| K3_dc_shape | blink | 0.0 | 3.3100531672951065e-11 | 0.7462844852003693 | 0.6530934295499777 | 2.6218182019001306 | 9.288606244774101 |
| K3_dc_shape | blink | 5.0 | 2.2540855084416437e-11 | 0.6415608775040345 | 0.7537677543359931 | 3.9953021315561443 | 5.662090131373809 |
| K3_dc_shape | lateral | -5.0 | 7.727100685711834e-11 | 1.090712044216133 | 0.4564341776665484 | -0.3770072430554362 | 7.736821306245792 |
| K3_dc_shape | lateral | 0.0 | 4.2584094272944937e-11 | 0.8201403038329034 | 0.6400721667889793 | 2.0246294426494993 | 5.138458127007123 |
| K3_dc_shape | lateral | 5.0 | 2.2901732303817472e-11 | 0.6051578555729726 | 0.7955612173768487 | 4.762522692728179 | 2.8763513340294997 |
| K3_dc_shape | mixed | -5.0 | 4.485134253012957e-11 | 0.8532584379132859 | 0.5469265008978139 | 1.5245713017890625 | 13.163497982720571 |
| K3_dc_shape | mixed | 0.0 | 3.388958313252934e-11 | 0.7537805383851915 | 0.6473267030397489 | 2.5354336092080896 | 9.174360425195994 |
| K3_dc_shape | mixed | 5.0 | 2.2871655312532016e-11 | 0.6446269989447087 | 0.7520504495980547 | 3.949779168443953 | 5.588705941375553 |
| calibration_raw_gain | blink | -5.0 | 1.5165448711732449e-12 | 0.16030936318089545 | 0.9838947781494837 | 17.278728536706595 | 28.945516444524166 |
| calibration_raw_gain | blink | 0.0 | 8.712711552256905e-13 | 0.11762450451501613 | 0.9905118136876558 | 20.262943569655526 | 26.929731612529494 |
| calibration_raw_gain | blink | 5.0 | 6.67217239655074e-13 | 0.09860103888332569 | 0.992631880625398 | 22.340021043914213 | 24.006809043731874 |
| calibration_raw_gain | lateral | -5.0 | 1.3497105770242564e-11 | 0.3361457048925418 | 0.9382166677535364 | 11.028577099439076 | 19.142405648740304 |
| calibration_raw_gain | lateral | 0.0 | 3.72457080231086e-12 | 0.19405080191041418 | 0.9762751162630687 | 15.704484530409491 | 18.818313214767116 |
| calibration_raw_gain | lateral | 5.0 | 3.1338663932474082e-12 | 0.13539972546274343 | 0.9850640091226137 | 19.374809284279173 | 17.48863792558049 |
| calibration_raw_gain | mixed | -5.0 | 1.8195618026074843e-12 | 0.1713570810122932 | 0.982154508030363 | 16.518510443353534 | 28.157437124285046 |
| calibration_raw_gain | mixed | 0.0 | 1.0176708333536567e-12 | 0.12502601593857904 | 0.9895725443687122 | 19.562233792767564 | 26.201160608755472 |
| calibration_raw_gain | mixed | 5.0 | 7.640854999489628e-13 | 0.10428666886746647 | 0.9919556041832768 | 21.668911645326535 | 23.307838418258136 |
| identity | blink | -5.0 | 7.403422266942269e-10 | 4.1239474930779165 | 0.2546059565948502 | -11.666787907817575 | 0.0 |
| identity | blink | 0.0 | 2.3411677654089895e-10 | 2.3190661337910843 | 0.4180029554759048 | -6.6667880428739705 | 0.0 |
| identity | blink | 5.0 | 7.403422318546091e-11 | 1.3041067151689552 | 0.6190039671535059 | -1.666787999817665 | 0.0 |
| identity | lateral | -5.0 | 4.706787708651855e-10 | 2.834047685308765 | 0.3669296614005887 | -8.113828549301228 | 0.0 |
| identity | lateral | 0.0 | 1.4884170130108913e-10 | 1.5937021557405027 | 0.5579549515555366 | -3.1138286843576246 | 0.0 |
| identity | lateral | 5.0 | 4.706787749827606e-11 | 0.896204577214837 | 0.748554916904236 | 1.8861713586986808 | 0.0 |
| identity | mixed | -5.0 | 7.324623675368554e-10 | 4.099289636053002 | 0.252768909904321 | -11.638926680931508 | 0.0 |
| identity | mixed | 0.0 | 2.3162494620031827e-10 | 2.30520000159495 | 0.41751898359997885 | -6.638926815987905 | 0.0 |
| identity | mixed | 5.0 | 7.324623725547525e-11 | 1.2963092160178702 | 0.6200305157917552 | -1.6389267729315997 | 0.0 |

Lateral remains the original VMD system's weakest condition: output SNR
8.954402/13.129189/16.622938 dB at nominal -5/0/5 dB levels. The new shape arm
scores -0.377007/2.024629/4.762523 dB on those same lateral strata. Blink and mixed
also deteriorate. Mean accuracy cannot replace separate condition reporting.

## Uncertainty and engineering gates

Intervals use the declared paired recipient/donor Poisson multiplier procedure
with 2,000 requested draws, seed42, 1,991 valid draws. This is a project statistical
approximation, not a paper-native test or proof of independent final performance.
Failure rows remain included. All saved paired metric differences follow.

| comparator | metric | mean_delta | ci95_lower | ci95_upper | valid_draws |
| --- | --- | --- | --- | --- | --- |
| calibration_raw_gain | mse | 3.539355552558182e-11 | 2.1199320583569667e-11 | 5.964531142472732e-11 | 1991 |
| calibration_raw_gain | rrmse_time | 0.6174915195611992 | 0.5748540199988842 | 0.6677099424888796 | 1991 |
| calibration_raw_gain | pearson_cc | -0.33626975239646223 | -0.365734603446637 | -0.307125002241228 | 1991 |
| calibration_raw_gain | snr_energy_db | -15.671952151374427 | -17.94613827554859 | -13.461239252082839 | 1991 |
| calibration_raw_gain | snr_improvement_db | -15.671952151374427 | -17.94613827554859 | -13.461239252082839 | 1991 |
| K3_calibration_mode_gain | mse | 3.40959049894672e-11 | 2.0361023313705934e-11 | 5.7859690814429626e-11 | 1991 |
| K3_calibration_mode_gain | rrmse_time | 0.5856220714043695 | 0.5486148437473912 | 0.6209388454560991 | 1991 |
| K3_calibration_mode_gain | pearson_cc | -0.32882827561442957 | -0.35550263403627336 | -0.3034039000453185 | 1991 |
| K3_calibration_mode_gain | snr_energy_db | -14.59151984441611 | -16.953027647607094 | -12.161185887607413 | 1991 |
| K3_calibration_mode_gain | snr_improvement_db | -14.59151984441611 | -16.953027647607094 | -12.161185887607413 | 1991 |

New-minus-original VMD SNR is -14.591520 dB with interval [-16.953028,-12.161186].
New-minus-calibrated-raw SNR is -15.671952 dB with interval [-17.946138,-13.461239].
Both comparisons reject a benefit for this fixed challenger on the exposed cohort.

All four arms leave synthetic clean controls exactly unchanged: clean modification,
alpha/beta error and covariance error are zero here. Such gated synthetic clean
controls do not establish native resting-EEG preservation or preservation during
ocular contamination. Shape fails the 15 dB gate and CC gate: mean loss relative
to calibrated raw .3362697524 exceeds .005. Original VMD passes 15 dB but fails
CC (.0074414768 > .005). Calibrated raw passes the controlled gates on this biased
mixture family; its prior native preservation failures and independent evidence
remain unresolved. Neither VMD arm reaches the20dB stretch target.

## Interpretation and next diagnostic

This is a failed fixed correction hypothesis, with reproduced stronger baselines
and passed operator fixtures. It does not establish that VMD is generally invalid
or that mode removal caused biological neural loss. The removed artifact variation
may contain useful ocular correction and may interact with the original residual
error. Current accuracy tables alone do not distinguish these mechanisms.

The next bounded evidence step is frozen-output error decomposition on Kaggle:
D=A_full-A_shape; e_shape=e_full+D; MSE change=mean(D squared)+2mean(e_full times D).
No recipe selection, parameter fitting, new waveform correction or target training.
See FRONTAL_SHAPE_DIAGNOSTIC_PLAN.md. Full native shape extraction stays stopped.
Preserve all failed runs and the original stronger implementation. Regional and
neural qualification remain false; no engineering threshold is relaxed.

Local operations were authoring, static inspection, hashes, formatting existing
CSV/JSON and artifact retrieval. No local numerical processing or tests ran.

## Diagnostic completion

The frozen-output diagnostic subsequently completed and passed243tests/640prediction/8320MSE/4160DCchecks. Discarded-field mean square3.032235725690054e-11 plus positive cross term3.773547732566658e-12 explains the MSE increase3.40959049894672e-11. Allninecontaminated strata worsen. See FRONTAL_SHAPE_DIAGNOSTIC_RESULTS.md; rejection and original baseline remain unchanged.
