# Replacement campaign execution log

## 17 October 2026, 09:05–09:31 IST

- Preserved legacy Git `ca91a42bcd741d69af22d1d97c45f8eb8e1d6dc4` in a
  verified 7,946,863-byte archive. AutoVMD working source hash remains
  `4c560d0ddc55569042485cdc1ee0c6d0a7f4386a82e137a76f667d10131d7be3`.
- Created isolated `active/` package and one Kaggle notebook. Existing old
  outputs, open mentor documents and the other chat's untracked files were
  preserved. The active package imports no legacy package.
- Rechecked original Klados provenance. All 53 local records have verified
  publisher correspondence; electrode row order and participant mapping remain
  unverified. Therefore fresh paired Klados comparisons remain nonregional.
- Preserved the original OSF confirmation/development participant identities.
  No new LEMON waveform or generated mixture is an input to this campaign.
- Authored fixed frontal VMD, regression, ICA component regression/rejection,
  GEVD–MWF and disjoint regional composition. Copied the historical 2,499-parameter
  network unchanged for frozen replay and a separately labelled frontal adaptation.
- Published initial source `dfe4093`; submitted
  `satyapaladugu/klados-osf-clean-comparison/1`, run `klados-osf-pilot-001`.
- **Version 1 failed before numerical contracts or EEG processing.** The old
  NumPy/SciPy pins conflicted with Kaggle's Python 3.13 environment and its
  preloaded numerical packages; importing SciPy ended with
  `ModuleNotFoundError: No module named 'numpy.char'`. Dependency setup consumed
  about 430 seconds. The failed source, specification and live log are retained.
- Repair: retain the coherent preinstalled numerical stack; install only
  pymatreader, pytest and tabulate with a bounded timeout. Record exact actual
  versions. Use `/tmp` for source/data working copies so they do not inflate
  saved Kaggle outputs. Match native direct regression to VMD's 1024-sample
  windows. Add explicit preservation tables and frozen frontal model adaptation.

No new measured cleaning score is claimed at this point. Contract and quality
results will be entered only after the repaired notebook produces them.

## 09:39–09:55 IST: verified baseline and simplified tree

- Pilot version 2 completed with 20 contracts passed in 5.22 seconds.
- The historical checkpoint reproduced 17.92720970519272 dB on its exact cached eight-record benchmark.
- Eight new paired Klados records and one native OSF session completed. Eleven figures were rendered.
- All 101 manifest payloads were downloaded and SHA256 verified. This receipt does not imply scientific qualification.
- Legacy code was physically relocated under archive/legacy_20261017; 380 moved files were hash verified. Other chats' untracked/open documents were preserved.
- The user requested six frontal methods on Klados and use of paper-listed electrode names. The operational mapping is saved in docs/provenance/klados_montage.json, with its row-binding assumption explicit.
- The two full ocular-method PDFs are unavailable to the user. No further request for them is pending. Exact reproduction gaps remain documented while available author implementations are pursued.
- BandRouteNet source pinned to 7f4dc0304b780cd15ada4caa829c7405cf4aa7d5; EEGOAR-Net source pinned to 4abeaee2b999e78a4abf01a00b9033c745dde8bc. No claim that these adapters have yet been evaluated.
- Kaggle GPU quota check returned 30.00 hours remaining; this is a quota snapshot, not a completion-time estimate.


## 17 October: six-method frontal scope

User requested historical VMD vs SVMD-FCM vs VME-GMETV vs EEGOAR-Net vs GEVD-MWF vs BandRouteNet, all on Klados frontal rows using paper-listed order. Added seven-row matched cached-record comparison, pinned author-file hashes, original GMETV Algorithm3 adapter, real successive decomposition, author pretrained and training adapters, method contracts, and nbclient-based executed-notebook recovery. Numerical checks and new scores remain pending Kaggle execution at this entry. See FRONTAL_SIX_METHODS.md for assumptions and unavailable reproduction settings. No LEMON or AutoVMD search launched.

### Six-method smoke001: adapter failure preserved

Kaggle GPU source53a8cd3 passed27 numerical tests in4.76s and verified all author-source Git blobs. EEGOAR weights loaded strictly, then label loading failed because the64-channel metadata is a vector, not a scalar dictionary. Retrieved9 hash-verified payloads including actual partially executed notebook (3of6 code cells,1error). Repair accepts the original array format, validates64 labels and retains strict source/weight identities. Added MAE and source-defined spectral RRMSE plus numerical fixtures; repaired run must pass before full launch.

### Six-method smoke002: incomplete author montage

Kaggle GPU source025be10 passed29 tests in5.33s. The64-label author montage is a Unicode vector containing FPz but no FP1/FP2; the adapter correctly refused nonexistent exact names. Retrieved9 verified payloads including partial executed notebook. Repair uses the unchanged channel-shared temporal network with anonymous computational slots; strict spatial-width-one inspection and equal first/last-slot output must pass on Kaggle. No electrode is renamed. OAuth expired during monitoring; restored through refresh token without user intervention and added expiry checks to orchestration. All three parent-cache hashes are now pinned.

### Six-method smoke003: spatial contract rejected shortcut

Sourceed82c07 passed29 tests in4.40s, then the explicit spatial-width contract rejected anonymous frozen EEGOAR slots. Initial inspection of temporal convolutions was incomplete: depthwise spatial convolutions, width pooling and3x3 decoder layers mix electrodes. No anonymous-slot result was produced. Retrieved9 payloads and the partial executed notebook. Repair retains author architecture and pretrained initialization, defines ncha8 with seven fixed Klados frontal columns plus one absent column, and fine-tunes only on training records with validation-only selection and the same15-epoch budget as BandRouteNet. This is explicitly a Klados fine-tuning adaptation.

### Six-method smoke004: successful GPU integration

Source8d51804 completed all six methods on one archived Klados test window, using one training window per train/validation split and one epoch per neural model. No method exceptions. Passed29 numerical tests. Historical frontal output17.937102dB on this single window is a smoke-only result, not final mean. Saved88 manifest payloads including actual executed notebook and12 inline figures. Author weights from the64-channel model loaded strictly into the8-column custom-montage architecture; spatial fine-tuning forward/backward succeeded. Next: full37/8/8 comparison with15epochs per neural model. Corrected the MWF bibliography to the verified JNE2018 paper DOI10.1088/1741-2552/aaac92.

### User requested explicit clean-EEG CC and native frontal OSF

FullKlados runc196374 completed24 windows/eight records, allsix methods, both neural15epoch budgets and29 contracts; zero method exceptions,922 outputs,13figures. Historical frontal output16.134624dB, cleanEEG CC0.984644 versus raw0.698520; the oldall19 score17.927210 remains separate. User now requested explicit before/after cleanEEG CC and frontal-only OSF before/afterHEOG/VEOG. Implemented independent NumPy CC recomputation, names/source row indices embedded inside labelled KladosNPZ copies, checksum-verified frozen model transfer, and matched five common OSF frontal electrodes F3/F4/F7/F8/Fz because FP1/FP2 are absent and FPz is not an alias. EEGOAR uses its fixed learned slots with two absent-input masks; missing-context transfer is flagged, not retrained on native scoring. FourdevelopmentOSFsessions,two scoringtrials per condition; calibration/scoring disjoint, confirmation closed. No native pairedclean target or reconstructionSNR is invented. Numerical checks and execution of this extension remain pending at this entry.

### OSF launch metadata repair

First SaveKernel rejected a51-character title. A shorter title was accepted and Kaggle assigned canonical slugfrontal-klados-and-osf-correlations (kernel139163159,version1). All parent/dataset sources accepted. Saved original requested run specification plus explicit server submission receipt; local monitoring/retrieval specification reconciles the canonical slug. Numerical sourceb9df0e6 and configuration unchanged. Fixed future authoring title/slug. This was a submission failure, not a numerical experiment failure.


### Per-mode clean EEG audit

User's clean-reference request is covered at both levels: cleaned output versus
paired clean EEG, and every saved VMD/SVMD/VME mode versus paired clean EEG,
HEOG and VEOG. The CPU audit reuses completed decompositions and retains signed
and absolute correlations, measured centers, source rows and mode counts.
It also audits residuals separately. Targets are diagnostic only and never
select runtime modes. A known orthogonal fixture verifies the three column
identities. Native resting summaries are calculated on Kaggle from the completed
OSF participant results. No neural training or decomposition is repeated.

### Native mode-reference diagnostic repair

The completed native before/after prediction table uses unchanged HEOG_lpf and
VEOG_lpf evaluation references correctly. Inspection found that its separate
mode-diagnostic CSV used the model-preprocessed EOG inputs instead. Preserve
that original evidence. Repair future diagnostics to use evaluation references,
and publish a new CPU audit version recomputing each saved native mode's CC
against the unchanged H/V arrays embedded in its NPZ. Independently recompute
every native prediction correlation from saved arrays and require agreement
with the original table; require complete row coverage. No new denoising,
training, calibration fitting or model selection occurs in this repair.

### Frontal extension complete: measured findings and recovery

Native source b9df0e6 completed four development participants p02–p05 and32
trials, five exact frontal names, no method exceptions. All240 payloads verify;
the executed notebook has5/5 code cells,16 figures,4 HTML tables and no errors.
Klados independent recomputation agrees for1512 clean-CC rows. The paired
mode audit checks2005 components/residuals; no undefined paired coefficients.

Historical frontal Klados output16.134624dB and cleanCC0.984644 versus raw
0.698520 are development-exposed evidence. Native historical blink abs VEOG
CC0.918516→0.328867 and lateral abs HEOG CC0.358906→0.214818 improve, but
mean resting relative RMS change0.636283 fails preservation. This is signal
modification, not a proven neural-loss percentage. All six methods and both
controls are retained in RESULTS.md, including poor/worsening results. No native
reconstruction SNR or paired clean correlation is claimed.

CPU reference-audit version2 uses exact published source0876931804df1c4344b70f165f0fb907f3a28bc8.
It passed32 numerical tests (14 recorded warnings), independently verified every
one of1280 native prediction correlation rows against the saved scoring table,
and repaired5071 native component coefficients using unchanged saved H/V.
No denoising/training changed or repeated. Retrieved25/25 payloads with hashes,
3/3 executed code cells,3 figures,3 HTML tables and no error cells. Original
mode CSV and version1 remain unchanged. Native-before/after table results stand.

Source templates and reusable modules are published; actual executed notebooks
and all numerical artifacts are local under notebooks/executed/ and results/.
Representative paired mode heatmap visually inspected. Readiness of a production
frontal system remains false because native preservation is unresolved.

### Pretrained-only extension: source discovery and eligibility

User approved assessing EEG-WNet and Diff-ADN only if pretrained weights are
available. No training or adaptations authorized. Eight successful exact/compact
GitHub and Hugging Face release searches returned no matches. Crossref and
Europe PMC records verified DOIs/dates/access metadata. WNet publisher HTML
returned403; publisher XML eventually returned200 but metadata only, no article
body. IOP returned200 after redirecting to a bot challenge; this is not full-text
access. Original XML429 attempt was an access failure, not absent supplements.
No compatible source/checkpoint pair was verified; neither new model is eligible.
A numbered CPU notebook will verify completed Klados/mode-audit parents and
preserve an executed closeout with blocked-candidate table and existing metrics.
It creates no new predictions or denoising scores. Existing frontal failures and
central/posterior boundaries remain unchanged. No GPU job or author contact.

### Pretrained closeout001: source-text checksum portability failure

Source266657f passed35 contracts in5.79s, then stopped before copying parent
tables: the discovery snapshot's raw-byte digest was calculated on Windows
CRLF bytes, while Git/Kaggle checked out LF bytes. No scientific artifact
identity changed. Retrieved all6 failure payloads and actual2/3-cell notebook
with one error. Repair defines discovery identity as canonical JSON SHA256
(sorted keys, ASCII escapes, compact separators), explicitly records the
encoding and adds a CRLF/LF equivalence plus changed-content rejection fixture.
Scientific parent result/data SHA256 and byte-count checks remain unchanged.
A new saved version will rerun contracts and closeout; old failure preserved.

### Pretrained closeout002 completed and recovered

Exact sourcea8c7b7171ad15b373422e233357f670159b0dee8, kernelversion2, run
frontal-pretrained-closeout-002. Passed36 contracts with14 recorded library
warnings in4.86s. RUN_COMPLETE reached at87.17s kernel runtime. Retrieved
22/22 payloads with SHA256 and byte-count verification; actual executed
notebook3/3 code cells,2 figures,4 HTML tables, zero error outputs. Both PNGs
visually inspected: all paired methods present, native resampling-only control
properly absent, before-reference lines and resting modification clearly shown.
First failed version retained. New inference/training0, both model candidates
blocked, parents pinned and unchanged, no native preservation resolution.
Source snapshot dates and access limitations remain explicit. This closes the
requested frontal experiment/release-review scope without claiming a qualified
frontal method or starting central/posterior information passing.

## Central stage — user-confirmed anatomical boundary

User selected FC/C/CP as central and temporal results separate. This stage
fits/evaluates the two groups independently; it does not alter the historical
broad data.region grouping or relabel frontal/posterior channels. Klados
core central C3/C4/Cz follows the requested paper-list binding assumption;
T3/T4/T5/T6 are temporal. Native OSF uses exact FC/C/CP and FT/T/TP labels.

Predeclared recipes: identity, calibration-only joint HEOG/VEOG ridge penalty
0.001 at strengths0.25/1, calibration-only GEVD–MWF rank1 at strengths0.25/1.
No search, per-scoring-window fitting, VMD, neural training or frontal messaging.
Use eight archived test record identities, first2000samples calibration and
three1024-sample scoring windows after independently filtering both sides of
the boundary. This protocol differs from archived frontal cache preprocessing,
so cross-region dB values are not a matched comparison and are not added.
Four native development sessions use disjoint whole-trial calibration/scoring,
2trials per rest/blink/lateral/vertical condition; confirmation remains closed.

Save per-channel metrics, clean-input preservation, original H/V evaluation
references, named predictions, calibration operators, failures and eight figures.
Native preservation is a proxy, not reconstruction accuracy. Reference-access
and failure outcomes stay visible. Numerical tests execute on Kaggle only.

### Central run recovered and reviewed

Kaggle status COMPLETE. Verified248/248 artifacts,3/3 actual executed code
cells,eight inline figures,four HTML tables,zero errors. No calibration failures.
Viewed pairedSNR, lateral H/V, pairedclean preservation and native rest
preservation PNGs. Native selected central21 and temporal6 channels; no
overlapping masks or frontal context. Individual negative-SNR windows and
worst-record clean modifications retained in RESULTS.md; no qualified correction
selected. Working central/temporal pass-through remains the safe default for
future system composition; it is not artifact removal. No additional parameter
tuning, model training or regional communication was started after review.

## Regional comparison requested: SGEYESUB and REG-ICA, including frontal

Inspected SGEYESUB author commit2c95b4f46f37670d25399ac0fdd705ae18248b25 and
REG-ICA official SCCN plugin archive SHA256
c0f2f6c696517940b2a011e2b53938336f6176ada5aba6ef192172bacc9b5a51.
SGEYESUB needs six sample-level artifactclasses, not four trial conditions.
OSF reader exposes those original sample labels; Klados lacks verified matching
labels, so do not invent an exact Klados reproduction. REG-ICA has adaptive
regression options; fixed ridge is not REG-ICA. Backend/runtime and numerical
reproduction remain unverified, no new scores or jobs claimed.

Recorded selection protocol in docs/REGIONAL_SELECTION.md: reproducibility and
preservation eligibility before efficacy ranking; paper metrics separated from
project gates; paired-source uncertainty; separate blink/lateral outcomes and
regional scoring; no forced winner from conflicting trade-offs. Fitting-scope
choice presented asynchronously (full montage recommended, region only, both).
No implementation choice dependent on that answer has been executed.

## Full montage explicitly approved

User selected full-montage fitting and asked that the choice and reason be
explained explicitly. Updated REGIONAL_SELECTION.md: all verified EEG inputs
(19Klados,58 in current OSFstudy01 sessions), separate frontal/central/temporal
scoring, EOG/annotations kept distinct. More spatial observations may help
separate ocular and neural activity; restricting Klados central ICA to3channels
limits separable components. This is a rationale, not proven improvement.
The shared spatial fit is not explicit frontal-to-back messaging. VMD remains
frontal-only. New author-code comparisons remain unexecuted; backend/calibration
contracts must pass before reporting numerical results.

Additional static author-code inspection: REG-ICA standalone regica.m refers
to regression function names not shipped under those exact names; the archive
contains crls_regica/scrls_regica. A runtime adapter must call verified routines
explicitly and document compatibility repairs rather than claim the untouched
wrapper works. scrls_regica depends on shipped fl.m; no symbolic toolbox was
identified in that routine. No numerical solver was run locally.

## Frontal-first spatial author comparison implementation

User clarified sequencing: compare frontal SGEYESUB/REG-ICA first; other
regional approaches are subsequent research, no central/posterior scoring now.
Prepared original author backend through Octave, pinned SGEYESUB2c95b4f and
EEGLAB8ac485f plus REGICA archive hash. One-hot ind2vec helper replaces only
a missing MATLAB toolbox encoding. Author solvers remain unchanged.
REGICA calibration-frozen Infomax/FD selection and author SRLS trial-local
adaptation are explicitly labelled an evaluation adaptation; first2 component
outputs retained to avoid original zero-initialization boundary deletion.
Full58channel native fit, score common5frontal exact parent inputs.
A focused readiness run must verify true six-class calibration and original
runtime before full four-participant evaluation. No new results claimed.

### Original spatial backend readiness verified

Run frontal-spatial-readiness-001, exact published source
a77ca55d358f9c91aae781aaff58c939209b005c, Kaggle frontal-sgeyesub-regica/1.
All40 contracts passed; original SGEYESUB fit/apply on real study01_p02
full58-channel calibration passed (13.293s fit/apply), original Infomax/FD/SRLS
small fixture passed. Recovered52/52 checksummed payloads and actual3/3-cell
notebook, one figure/table, no errors; waveform visually inspected. Runtime
success is not efficacy or qualification. Full frontal-only comparison adds
REG-ICA to the exact original paired Klados windows and both methods to four
native OSF sessions. Original ICA learning-rate history/step cap saved; all
failures retained. SGEYESUB remains ineligible on Klados without verified six
calibration classes. New paired calibration uses first2000 independently
preprocessed samples and a matched calibrated ridge control; earlier scoring-
window ridge remains distinct. No new VMD, LEMON, central/posterior scoring,
message passing or neural training. Full numerical results pending.

### Full spatial version2 input-contract failure and bounded repair

Sourcee8d0021/runfrontal-spatial-full-001 passed41 contracts, completed all8
Klados records with no method exceptions, fitted original SGEYESUB/REGICA for
OSFp02/p03 and saved all8p02 scoring trials. At p03 scoring, strict zero-
tolerance equality found1/8000 values differed by1.4901161e-08 (onefloat32ULP)
from the archived frontal input. Preprocessing source files are unchanged;
the difference is consistent with rounding, but its hardware/library cause
has not been isolated. The algorithm fits did not raise the error.
Recovered359/359 verified payloads, actual2/3-cell notebook with1error, no plots.
This remains an incomplete failed run, not a completed native result.

Repair: reject shape/nonfinite and differences above onefloat32ULP, log each
representational difference, replace only frontal scoring rows with the exact
verified archived bytes before new inference, and assert exact equality.
Scientific thresholds are unchanged. Add a Kaggle regression fixture accepting
one step and rejecting two. Resume only the declared failed parent/source;
verify copied bytes/hashes, paired-window coverage and calibration-input
identity before reusing outputs/models. Reused fit costs remain in parent logs,
distinct from new-runtime rows. No clean EEG selects this repair, no missing
trials excluded and no other regions run.

Repair submitted as frontal-spatial-full-002/source
d87b2e43116c9a7d58dd3adf755ab8df6a0698e1, Kaggle version3.42 contracts passed,
including the one-versus-two-ULP regression. Kaggle accepted the previous
failed version as a metadata parent but did not expose its manifest to the
repair; the current run therefore recomputes instead of restoring. This is an
operational recovery failure, not a denoising score. User was explicitly told
the earlier reuse statement was premature. Failed359-payload archive is safe
locally. Future runner logs a missing recovery mount explicitly and writes
the fresh-recomputation decision; no successful cache-recovery claim is made.
