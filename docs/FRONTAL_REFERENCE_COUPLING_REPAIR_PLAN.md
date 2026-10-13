# Bounded development-corpus reference-coupling repair

14 October 2026 IST. Proposed next engineering checkpoint, not implemented or
launched. The independent benchmark remains unavailable under the audited source
contracts. This repair does not turn the existing regression-generated corpus
into an independent or biologically clean benchmark.

## Problem and preserved evidence

In corpus.py the calibration EEG receives a scaled donor field while its EOG
references retain their source amplitude. Scoring EEG receives another mixture
multiplier while its EOG references remain unscaled. The existing gain estimates
partly compensate, but EOG amplitude-based support and calibration-to-scoring
interpretation do not represent consistently coupled generated sensor signals.
This was previously disclosed; the source audit confirms it rather than discovering
a new performance failure. Preserve all original arrays, hashes, splits, metrics
and the17.112814dB frontal VMD baseline.

## Scope and constraints

Create a separate, explicitly named development-corpus version and pilot protocol.
Reuse verified recipient/donor identities and only their already eligible
development partitions. Keep all confirmation and external reserved sources
closed. Preserve sampling/filtering/window/trial contracts and separate calibration
from scoring before construction. The inference APIs still receive no paired
target, true artifact, generation multiplier or future samples.

The producer must define a reference baseline and a homogeneous signed forward
operator F on the reference excursions: F(aR)=aF(R). Apply the same source multiplier
to the generated reference excursion and injected field, independently for
calibration and scoring. Do not just multiply raw references while leaving an
affine offset or inconsistent lag-specific baseline in the field generator.
Record baseline, coefficients, units, finite-lag masks, source multiplier and
operator identity as generation evidence. Such values are producer/evaluator
information and remain inaccessible to correction functions.

Reacquire or reuse checksum-verified unscored recipient prefixes and original
donor calibration where necessary. Do not reconstruct an unavailable recipient
prefix from a mixed calibration array by an unverified algebraic assumption.
Unknown physical source units remain explicitly unknown; generated driver-unit
conventions must be documented. Do not introduce invented neural leakage, field
signs, head geometry or learned propagation parameters.

Retain identity, calibrated raw and original full-field frontal K3/alpha2000 VMD.
No additional method or K/alpha bank. New EEG inputs require their own exact
waveform/source/cache identities; historical mode bundles cannot be reused merely
because channel names match. Posterior and unknown/masked rows remain pass-through.

## Modular tasks and gates

| Part | Input | Output | Acceptance |
| --- | --- | --- | --- |
| RC01 provenance audit | donor/recipient/calibration manifests and producer code | recoverable unscored source and baseline contract | exact source, reference, units, split and boundary identities; no guessed reconstruction |
| RC02 generator contract | verified sources and declared homogeneous operator | versioned coupled EEG/reference records | consistent multipliers, finite boundaries and target-only producer access |
| RC03 Kaggle fixtures | small controlled examples | numerical linearity, zero-drive, scaling, lag/baseline and leakage checks | all tests pass; no local numerical work |
| RC04 bounded source pilot | two deterministic eligible development recipient/donor pairs | saved calibration and clean/blink/lateral/mixed records | new hashes, complete recovery, reserved sources closed |
| RC05 matched frontal pilot | verified new records and three frozen methods | paper metrics, preservation, runtime and failure tables | equal information access; frontal-only VMD; no superiority/promotion assumed |

Clean zero-driver controls are idealized and cannot establish native false-positive
behavior. Real EOG donor waveforms may contain nonocular activity. Paired references
remain retained recipient EEG. Regression-derived field bias and prior native
preservation failures remain unresolved. Coupling repair may leave a gain-adjusted
method's output unchanged; an SNR increase is not guaranteed.

Keep the old and repaired corpus comparisons distinct. Any further source or
generator change needs its own identity/protocol. IB02-IB04 independent-benchmark
work and model promotion stay closed until their separate scientific requirements
are met.
