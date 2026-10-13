# Next fixed controlled ablation: artifact-shape selectivity

13 October 2026. Planned, not implemented or launched. GPT-6.1 Sol extra-high
review recommendation follows the verified failed trial-support preservation
experiment. No new K/alpha search, posterior VMD, model or reserved evaluation.

## Scientific question and limits

Can restricting the estimated artifact waveform to selected frontal VMD vectors
improve paired accuracy/preservation, while retaining the existing calibration
field and gain? The prior selected-mode gain projection scored10.890616dB and
still subtracted the full field. Prior per-component EOG projection also failed.
These negatives lower confidence; they do not establish this different operator.
One fixed cached controlled comparison is defensible. It must not become another
parameter search or a renamed repeat of the gain-only experiment.

The controlled corpus constructs contamination through a calibration regression
field, which can favor regression. A positive result needs an independent
contamination benchmark before a broad VMD superiority claim. Existing LEMON
recipient references are low-ocular, not proven artifact-free.

## Frozen four-arm comparison

1. Identity.
2. Stronger fixed calibrated-raw shared-gain comparator (archived18.193247dB).
3. Original K3/alpha2000 VMD gain with full calibration-field subtraction
   (archived17.112814dB).
4. The same VMD artifact/gain/admission, with the DC-preserving shape projection
   defined below. No new gain estimate and no temporal-support change.

Use the unchanged controlled development cohort, original preprocessing,
recipient/donor partitions, calibration identities and archived all-window K3
vectors/residuals. Verify their exact parent manifests and launch SHA. Keep
association0.4, mode association0.2, lags[-10,0,10], ridge0.01 and multiplier1.
Baseline original predictions and paper metrics must reproduce before the new
arm is interpreted. Do not select a recipe on the exposed native resting data.

## Shape operator and original-input composition

For each eligible frontal row, let A be its original VMD-supported artifact
estimate and M the selected archived K3 mode vectors. On the existing lag-valid
samples only, center A and each selected mode. Project the centered artifact
onto the centered selected-mode span using a fixed numerical least-squares
solve with a fixed recorded rank rule, then add the ORIGINAL artifact mean back. This projects into the mode
span plus a constant; it preserves DC rather than changing the correction mean.

Keep invalid lag edges zero. An empty/degenerate centered span retains only
the original artifact mean on valid samples; record that case explicitly.
Failed VMD, masked and nonfrontal rows retain zero artifact and pass through.
Use no clean target, true artifact, input mixture level or scoring label in
projection, component selection, calibration or gain estimation.

Subtract the new artifact once from ORIGINAL EEG. Retain original input outside
the estimated correction; do not discard the reconstruction residual. This
does not imply residual content is mathematically untouched, since VMD
components need not be orthogonal. Do not
reconstruct cleaned EEG by dropping whole modes. The selected EEG-derived basis
is itself contaminated: numerical projection is not proof of pure ocular/neural
separation. Save rank, singular values, selected modes, DC, valid sample counts,
projection coefficients, retained/discarded artifact energies and correction
differences without converting them into primary
accuracy metrics.

## Tests, reports and decision

Kaggle fixtures must cover in-span/off-span artifacts, projection idempotence
and orthogonality, DC preservation, rank
deficiency, empty selection, invalid lag edges, mode permutation, signed fields,
unit scaling, failed/masked rows, variable channel counts, original subtraction,
reconstruction and baseline parity. Local work remains authoring/static/Git only.

Primary paired metrics remain MSE, temporal RRMSE, ordinary Pearson CC and
separately labelled output SNR/improvement. Keep per-channel, per-example,
recipient, donor, condition and input-level tables; all failures stay scored.
The primary CC comparator is the stronger fixed calibrated-raw arm, not a
weaker earlier direct-regression score. Engineering gates remain unchanged.
Report recipient/donor-aware development uncertainty without independence claims
for previously exposed records. No regional dB addition or native SNR claim.

Failing numerical/provenance parity stops interpretation. Failure against
accuracy/preservation criteria rejects promotion of this fixed challenger.
A passing controlled result still needs native preservation testing and an
independent benchmark; it does not authorize a teacher or deployment model.
Native archives contain only first-window modes, so full native validation
requires a separately pinned fixed-K3 decomposition/extraction stage after
controlled evidence. Current real-data and controlled qualification remain false.

| Part | Input | Output | Acceptance |
|---|---|---|---|
| SC01 cache/interface audit | frozen controlled artifacts and mode metadata | exact parent/identity contract | all required mode/field/window identities available |
| SC02 shape helper and fixtures | selected vectors, original artifact, valid mask | DC-preserving artifact | all numerical contracts pass on Kaggle |
| SC03 four-arm notebook | exact published SHA/versioned parents | complete paired development evidence | frozen baseline prediction/metric parity |
| SC04 review | accuracy, preservation, failures and cost | accept/reject fixed challenger | unchanged stronger comparator and gates |

No SC implementation or job has started at plan publication.


## Implementation and interface audit

Exact controlled parent frontal-monotonic-events-001/1 at939fadf435440b3cac7436
f03c3ce6d80dfa52aa, manifest1ab061565aca07fb826d48477087c949720dfa300facad90be
3ebe2008876d20, contains separate post-gain raw/VMD fields and selected-mode
metadata for all320windows. Attach corpus/2, K3-cache/1, readiness/1 and frontal
fixture/1. The K3 cache is historical evidence: only K3/alpha2000 entries are
read; AutoVMD selection and decomposition are never invoked.

Reusable frontal_artifact_shape.py and notebook55 are authored. DC projection
uses fixed rcond=eps64*max(valid-sample-count,selected-mode-count), returns rank,
coefficients/singular values and retained/discarded centered artifact energy.
The helper accepts no paired target and preserves invalid lag edges as zero.
Underlying baseline metrics are checked before new CC deterioration is computed
against calibrated raw. Legacy nonfinite nulls are normalized for parity only;
new CSV retains infinity/NaN rather than inventing finite clean SNR.

Nine artifact-shape fixture functions (including three invalid-selection cases)
and an explicit frontal-only uncertainty-scope fixture are queued for Kaggle.
The generic uncertainty helper's default remains all original regions; only
this stage explicitly requests frontal scope. No region rows are fabricated.
Source/notebook syntax is checked locally; no numerical work has run locally.

## Kaggle execution checkpoint

236 focused tests passed on Kaggle at source 1444b4e226b1e879ad46b2fe638c4d77dc26a753. All eight manifest payloads recovered and verified. Full controlled run frontal-shape-full-001/1 submitted at that same SHA; observed RUNNING at 15:54:26 UTC on 13 October 2026. Numerical results pending; no promotion.
