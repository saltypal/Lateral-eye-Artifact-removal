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
