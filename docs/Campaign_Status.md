# Kaggle campaign status

Updated 2026-10-04. This is an execution ledger, not a performance report.

## Measured and verified

- The original source inventory contains 375 files totaling 3,753,247,751 bytes. Local preparation only packaged and hashed files; no local signal processing, model tests or training ran.
- Private dataset `satyapaladugu/lateral-eye-complete-dataset` version 3 is published. Its opaque ZIP research bundle is 3,369,877,058 bytes with SHA256 `eb257fb91be83600b93e62f771f5458bdaa27b6603821ad4bcf03dc2441c3d98`. The published bundle size matches. Every-file SHA verification is a separate Kaggle audit gate.
- Kaggle contract run at Git `cdc06df93025c7b61490facc5558619f8cd56255` passed 2 tests in 3.54 seconds.
- Kaggle contract run at Git `ebe61d20bc420689e844465c8dd3492c05e1e31a` passed 6 tests in 5.71 seconds, including EEG annotation preservation, variable channel shapes, padding isolation, permutation behavior and checkpoint reload. These are implementation tests, not cleaning-quality results.
- Audit version 6 at Git `5f37a816397069e4bbb61ffd6943436e95de064c` passed 7 tests in 5.43 seconds and all 375 source hashes. Klados dirty/clean shapes are exactly `[53,19,5401]`; there are no hash-identical duplicate clean records. The OSF loader gate failed because the files use MATLAB v7.3/HDF5.
- Audit version 7 at Git `8afc57dde4ffaf976e3fd9b47109defe40736c1f` passed 8 tests in 7.98 seconds, including the added MATLAB v7.3 fixture. Original OSF external FDT pointers remained unresolved, so its usable-session inventory is not accepted evidence.
- Audit version 8 at Git `6c0d369e3bade14eb3fcaf66e4789594a63110ac` passed 10 tests and all 375 hashes, loading 30 original OSF sessions. Subsequent source inspection found raw EOGL/EOGR channels mislabeled as EEG; the benchmark and restoration commits correct their classification.
- Benchmark version 1 at Git `bbba266e6f4f` passed 16 tests in 6.98 seconds and completed the 40 K/alpha configurations, producing 320 record-level rows for 160 correction candidates. Spatial comparison and final feasibility decisions are still pending.
- Restoration version 1 at Git `31373e310ffa` passed 16 tests in 7.13 seconds, verified all 45 publisher-listed study04 files and recovered the 37 missing files on Kaggle. Its completed audit measured **45 usable original sessions and 39 global participants**, all four studies, with no missing-session exclusions. The 146,377,780-byte supplement has SHA256 `79a3c4f2e1c1d545b477aa3de05fdc9c1c07c0355b820755392d8a5519e67e65`. The original 375 files still match their original hashes.
- The old notebook named `VMD_Hyperparameter_Sweep_OSF_Klados_Baseline.ipynb` uses `K_VALUES=[5]` and `ALPHA_VALUES=[1000]`. It did not establish an optimal K or alpha.

## Running and pending

- The restored sources are persisted as a separate Kaggle output bundle. Subsequent training requires that supplement, checks container/file hashes and preserves mismatched existing files. This new attachment path awaits its next Kaggle regression test.
- K=3..10 quality decisions, center-frequency figures, spatial parameter choices, frontal/posterior proxy results and student training remain unverified until their corresponding jobs complete.
- A development smoke grid cannot justify final optimality. Five grouped folds, three seeds, matched old/recent baselines, OSF study/participant isolation and a separate streaming study remain required by the research contract.

## Resolved infrastructure failures

The initial dataset creation response was accepted but no usable full dataset appeared. A manifest-only bootstrap created a valid private dataset. A subsequent CLI version upload finished the large archive but failed on a Windows upload-cache filename containing a forward slash. The completed archive token was recovered without printing credentials or sending the archive again. Kaggle then marked its expanded-archive version 2 as Failed; the API exposed no underlying reason. Version 3 stores a complete opaque `.eogbundle` ZIP and lets the notebook verify/extract it explicitly. This avoids backend archive parsing and preserves nested archives/hidden files.

Audit versions 1 and 2 failed due to missing source attachments/data. Neither supplied valid dataset evidence. The controller now checks the current published inventory before launching a data-dependent phase.

Audit version 3 passed 7 fresh-subprocess tests but then failed from mixed in-process NumPy binaries after dependency installation. Version 4 exposed unavailable `ensurepip` in Kaggle's OS Python; version 5 referenced an invalid commit and was superseded. The notebook now creates a pip-free virtual environment, installs into it using host pip's `--python` option, and performs all numerical work in fresh subprocesses. The controller rejects unknown/unpushed commit hashes before submission.

## Reproduce

Run from the isolated `region_aware_research` Git checkout:

```powershell
python tools/kaggle_campaign.py inventory-check
python tools/kaggle_campaign.py status --phase benchmark
python tools/kaggle_campaign.py retrieve --phase benchmark
python tools/kaggle_campaign.py retrieve --phase restore
```

Use the explicit full SHA of the pushed source revision when submitting. `submit --phase train` attaches completed benchmark and restoration outputs. A rejected VMD teacher permits an explicitly labeled paired-only student; missing/incomplete benchmark evidence blocks training. Kernel-version outputs must be saved and retrieved; `/kaggle/working` alone is temporary.

## User concerns tracked explicitly

Center-frequency K sweep, fixed versus adaptive bins, evidence for selecting five modes, input-to-mode vector construction, computational complexity and transformer necessity, grid search, frontal/posterior fusion, combined blink/lateral cleaning and N-channel scalability are documented in `VMD_Concerns_and_Explanation.md` and `Model_and_Search_Design.md`. Writing those explanations does not substitute for their measured experiments.
