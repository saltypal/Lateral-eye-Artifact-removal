# Posterior/middle regularized CCA adaptation

14October2026. User requested the next approach after fixed-strength diagnosis.
All numerical work runs on Kaggle. FrontalVMD K3/2000 stays frozen. AutoVMD,
neural training and reserved data remain closed.

## Scientific question and scope

Can calibration-selected EEG/EOG canonical directions improve the artifact field
when scalar gain cannot fix severe lateral reconstruction? Six fixedmethods:
directjointlagridge; previousrest-whitenedreference-rank2; referenceCCA rank1/2;
EEGCCA rank1/2. Each has posterior-only,central-only andboth compositions, plus
identity and two inheritedcontextcontrols:21arms. Same exactcorpus002 and
matched-context-full001 parents, unscoredcalibration, lagbank[-10,0,10], ridge.01,
fieldadmission.4 andgaincap8. No broadsearch or scoring-target fitting.

## Literature and labelled adaptations

Primary paper and author implementation:
https://doi.org/10.3390/s23198214
https://github.com/downeyryanj/iCanClean
The paper uses CCA to identify reference-correlated directions, choose candidate
noise components and subtract a regressed projection. Its cited ocular benchmark
is phantomEEG, not evidence of our human reconstructionSNR. Reference-side,
EEG-side and joint source choices are distinct. Our frozen-calibration,regularized,
lagged lowrank implementation is an adaptation. MATLAB parity and published
settings are not claimed. Source code is newly authored; author code is not vendored.

## Algorithm and interfaces

fit_experts(calibrationEEG,references,masks,trialtypes,boundaries,policy) accepts
no scoringtarget,trueartifact or generationmultiplier. Standardize EEG and lagged
reference predictors using valid calibration moments. Exclude outerlag edges and
trialjoin edges before covariance estimation. Whiten both covariance matrices
withfixedregularization.01 in standardizedcoordinates; SVD their crosscovariance.
Record regularizedsingularvalues and actual paired calibrationPearsonCC separately.
Admit canonicaldirections at squared actualCC>=.16, corresponding to|CC|>=.4.
This threshold matching is a project choice, not the paper's publishedoptimum.
Retain first1/2admitteddirections in regularizedSVD order; fewer available are
explicitly recorded; no admitteddirection gives scoredpass-through withfailure.

Reference arm: truncate the standardizedreference design in canonical coordinates
then apply the same frozenridgecoefficients. Fullreferencebasis must reconstruct
identity and exactly reproduce matchedridge within numericaltolerance. Rank1/2
truncation, rather than naming a fullrank regressionCCA, is the substantivechange.

EEG arm: apply calibrated canonicalEEG directions to scoringEEG and the calibrated
ridgebackprojection. These directions can retain neuralcontent; use as a separate
preservation-risk control. Every branch uses the same originalinput fieldassociation
gate. Referencearms estimate a boundedregion gain from observedEEG/candidatefield;
EEGarms preserve unitcorrection strength. HEOG/VEOG access and differences are
reported. No target, conditionlabel or knowninputSNR enters correction.

Outputonly verified regions1/2. Frontal andunknown/maskedrows remain frozen.
Each regional correction refers to originalEEG and is composed once. Save fitted
matrices,means/scales,actualCC,admissionindices,regularizedspectra,coefficients,
rank andfailures. Reload and replay every fit bit-for-bit before scoring.

## Coupling prerequisite and current execution decision

Producerinspection confirms legacycorpus fields are scaled while EOGreferences
are not; the paired construction is regression-derived and development-exposed.
The original RC01-RC05 repair remains pending; do not claim it passed or relabel
this corpus as coupled/fresh. Current next-approach request proceeds with a matched
legacyengineeringpilot, with explicit limitation, rather than qualifying a model.
Save a coupling audit and evaluator-only gatecoverage/trueartifact energy after
predictions. These values cannot affect admission or inference. Corrected-source
and nativevalidation remain required before any qualification or promotion.

## Tests, pilot and stop rules

Tests coverfullreference/ridgeparity,boundaries,sourceunits/scaling,channelpermutation,
constantreferences,invalidpolicies,outputregionmasks,lag-edgezero andsavedfitreplay.
Kaggle runner executes all fixtures before50source-balancedexamples. Compare all
eligible blink/lateral/mixed/clean cases, runtime/memory and failurepass-through.
Report paperMSE,temporalRRMSE,ordinaryCC,SNRoutput/improvement separately, plus
middle20channel andtemporal6channel diagnostics. Same engineeringgates; native
EEG receives suppression/preservation proxies, not reconstructionSNR.

Full320replication follows only verified pilot contracts/recovery and a useful
matchedresult. If CCA is inferior or degenerates to ridge, retain negative evidence,
stop this recipe, and prioritize coupled-source repair/eligibleSGEYESUB rather
than another blindrank/thresholdgrid. No automatic neural promotion. Frontal
VMD superiority still requires the matched rawcontrol; fixedfrontaloutput never
constitutes a new VMD gain. Source-aware uncertainty is supplemental, not exact
paper-native participantpermutation. All savedoutputs pin publishedSHA/parents.


## Pilot evidence and revised full check

Pilotv1 primarywholemeans:directridge19.601765,CCAreference2 17.549658,previous
rank2 16.728031dB.297tests passed. CCAimproves the older rank2point, but the
simpledirectcontrol is stronger. Preserve primaryscores; subgroup preservation
copybug is documented and repaired before use. Revisedsource computes middle/
temporalmetrics on their own channels, adds pooledSNR and labelled strongest
newdirectcontrol plus matchedCCA-versus-rank2/direct uncertainty. Gates unchanged.
The next full320run tests samefixedrecipes after focusedKagglerepairfixtures;
no pilotselectedweight/rank, no newthresholdbank or sourceconstruction change.


## Reproduce the full corrected comparison

From the project root after Kaggle authentication:

```powershell
python tools/kaggle_campaign.py submit --stage nonfrontal-cca --run-id nonfrontal-cca-reproduction-001 --profile full --config configs/nonfrontal_cca.json --source-sha b0c861373a8bc66c4538b78269a3b9b988ce3171 --kernel-slug vmd-nonfrontal-cca-reproduction --kernel-source satyapaladugu/vmd-eog-corpus/2 --kernel-source satyapaladugu/vmd-matched-context-full/1
```

Choose a new run ID if that one exists. This is local orchestration only;
all numericaltests,fitting,evaluation andplotting execute on Kaggle. Check and
recover the submitted run with:

```powershell
python tools/kaggle_campaign.py status --run-id nonfrontal-cca-reproduction-001
python tools/retrieve_bounded.py --run-id nonfrontal-cca-reproduction-001
```

Numberednotebook62pins the actualfullsourceSHA andversionedparents. Earlier
pilot source/notebook/results remain archived; laterdocumentation commits do
not change runningexperiments. No neuralmodel istrained by these commands.

## Completed comparison and decision

The corrected full320 run completed at b0c8613 with 299 passing Kaggle tests.
Complete calibrated reference ridge: 17.533717 dB channel-first mean and
14.780705 dB pooled-window mean. Reference CCA rank2: 15.833574 / 13.923972;
previous rest-whitened rank2: 15.155692 / 13.480444. All use identical frozen
frontal VMD. CCA improves the old rank2 control but loses to complete ridge:
matched CCA-minus-ridge SNR interval [-2.548602, -.943181] dB. It also fails
the strongest direct-control CC gate (.006414969 deterioration > .005).

Retain complete ridge as a development candidate and preserve the negative
CCA evidence. No wider rank/threshold grid, teacher or student is authorized
by these outcomes. The severe lateral -5 dB stratum remains 13.278714 dB,
the legacy corpus coupling mismatch remains unrepaired, and native frontal
preservation and matched no-VMD frontend qualification remain unresolved.
Prioritize the bounded RC01-RC05 source-coupling repair and native evaluation.
See NONFRONTAL_CHALLENGERS_RESULTS.md for all regions, conditions, metrics,
uncertainty, recovery identities and qualification limitations.
