# Independent frontal benchmark source audit

14 October 2026 IST. Implement IB01 as a metadata/provenance audit with notebook57.
Keep K3/alpha2000 and the original full-field correction unchanged. This stage
does not create mixtures, decompose EEG, compare cleaning accuracy or select a
method. The output is an eligibility decision, not an SNR result.

The required paired benchmark needs accessible selection sources, verified frontal
labels and grouping, a declared paired target, an independent reproducible
generator, signed per-channel fields, compatible physical units/references,
explicit HEOG/VEOG polarity and separated calibration/scoring inputs. Unknown
requirements remain unknown; none are filled by plausibility or hemisphere guesses.

The publisher Ear-EEG supplement is inspected as documentary evidence. Download
the public supplementary archive, read only Data_Sheet_1.docx and verify SHA256
0f7ee5133c6d3b6ac8d39af51ca831c267831410b8793b8e37f615061bcbedc8. Extract Table S3
as text and preserve its row groups/column labels. Do not interpret grouped
F7/8 values as signed individual electrode coefficients. Dipole direction vectors
do not supply missing geometry or reference conventions.

Reconcile the old Klados audit: exact publisher-to-export matches did not establish
row anatomy, units or participant groups. Keep legacy evidence and its limitations.
AFFiNE's paired simulation informs requirements, but its data-release statement
prevents claiming an available benchmark. Native OSF supplies unpaired proxies.
Magdeburg stays reserved. EEGdenoiseNet does not supply verified regional metadata.

Implement a strict candidate contract register with verified/unresolved/absent/closed
states, source links and reasons. Kaggle checks the register, source hash, document
structure and fail-closed fixtures, then saves the full source table, eligibility
matrix, blocking reasons, environment and completion state. All status inputs are
documentary assessments; they are not empirical measurements or paper metrics.

Accept software execution only if tests and source identity pass. A completed audit
can have benchmark_ready=false. If no candidate meets the source contract, stop
IB02-IB04 rather than inventing a generator, opening reserved sources or reporting
another SNR. A full public signed map and compatible reference metadata, or a
verified paired publisher dataset, would permit a new evidence-based protocol.

## Launch checkpoint

frontal-source-eligibility-001/1 submitted at36b21a12f8f2c1d46368ca7aa72b8d1acbdc5c5e. First primary RUNNING21:37:10UTC13October2026 (14OctoberIST). No datasets/parents. Software tests, pinned document and table checks precede the documentary verdict; results pending.

## Preserved failure and source-loading repair

frontal-source-eligibility-001/1 at36b21a1 passed263tests in19.20s, then failed reading the EuropePMC HTTP archive with a30second read timeout. All seven failed-run manifest payloads were recovered/verified. The repair vendors the unchanged previously retrieved DOCX with attribution and the SAME0f7ee513 checksum into Git. A new version verifies this source from its pinned checkout without live publisher HTTP. Three fixtures check byte identity, hash rejection and checkout path containment. Eligibility thresholds and source states are unchanged; the failed run is retained.

## Completed repaired outcome

frontal-source-eligibility-002/2 at27eeefe completed266tests, originalDOCXhash and TableS3structure checks. All15manifestpayloads verified. Documentary assessment completes IB01 butbenchmark_ready=false for all eight reviewed candidates. IB02-IB04 remainheld; no newSNR. See FRONTAL_BENCHMARK_ELIGIBILITY_RESULTS.md.
