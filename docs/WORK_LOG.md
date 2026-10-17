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
