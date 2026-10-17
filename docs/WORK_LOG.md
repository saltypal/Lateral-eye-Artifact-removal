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
