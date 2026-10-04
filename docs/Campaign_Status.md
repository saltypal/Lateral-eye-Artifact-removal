# Kaggle campaign status

Updated 2026-10-04. This is an execution ledger, not a performance report.

## Measured and verified

- The original source inventory contains 375 files totaling 3,753,247,751 bytes. Local preparation only packaged and hashed files; no local signal processing, model tests or training ran.
- Private dataset `satyapaladugu/lateral-eye-complete-dataset` version 3 is published. Its opaque ZIP research bundle is 3,369,877,058 bytes with SHA256 `eb257fb91be83600b93e62f771f5458bdaa27b6603821ad4bcf03dc2441c3d98`. The published bundle size matches. Every-file SHA verification is a separate Kaggle audit gate.
- Kaggle contract run at Git `cdc06df93025c7b61490facc5558619f8cd56255` passed 2 tests in 3.54 seconds.
- Kaggle contract run at Git `ebe61d20bc420689e844465c8dd3492c05e1e31a` passed 6 tests in 5.71 seconds, including EEG annotation preservation, variable channel shapes, padding isolation, permutation behavior and checkpoint reload. These are implementation tests, not cleaning-quality results.
- The old notebook named `VMD_Hyperparameter_Sweep_OSF_Klados_Baseline.ipynb` uses `K_VALUES=[5]` and `ALPHA_VALUES=[1000]`. It did not establish an optimal K or alpha.

## Running and pending

- Audit kernel version 3 at Git `33776ddd2775821a76d34d482e71fd28d62442cf` was submitted against the complete published dataset.
- Actual data shapes/labels/exclusions, K=3..10 development results, center-frequency figures, spatial parameter choices, frontal/posterior proxy results and student training remain unverified until their corresponding jobs complete.
- A development smoke grid cannot justify final optimality. Five grouped folds, three seeds, matched old/recent baselines, OSF study/participant isolation and a separate streaming study remain required by the research contract.

## Resolved infrastructure failures

The initial dataset creation response was accepted but no usable full dataset appeared. A manifest-only bootstrap created a valid private dataset. A subsequent CLI version upload finished the large archive but failed on a Windows upload-cache filename containing a forward slash. The completed archive token was recovered without printing credentials or sending the archive again. Kaggle then marked its expanded-archive version 2 as Failed; the API exposed no underlying reason. Version 3 stores a complete opaque `.eogbundle` ZIP and lets the notebook verify/extract it explicitly. This avoids backend archive parsing and preserves nested archives/hidden files.

Audit versions 1 and 2 failed due to missing source attachments/data. Neither supplied valid dataset evidence. The controller now checks the current published inventory before launching a data-dependent phase.

## Reproduce

Run from the isolated `region_aware_research` Git checkout:

```powershell
python tools/kaggle_campaign.py inventory-check
python tools/kaggle_campaign.py status --phase audit
python tools/kaggle_campaign.py retrieve --phase audit
python tools/kaggle_campaign.py submit --phase benchmark --sha 33776ddd2775821a76d34d482e71fd28d62442cf
```

Benchmark submission requires review of the completed audit. `submit --phase train` additionally attaches completed benchmark outputs, and the training module rejects missing/failed VMD feasibility gates. Kernel-version outputs must be saved and retrieved; `/kaggle/working` alone is temporary.

## User concerns tracked explicitly

Center-frequency K sweep, fixed versus adaptive bins, evidence for selecting five modes, input-to-mode vector construction, computational complexity and transformer necessity, grid search, frontal/posterior fusion, combined blink/lateral cleaning and N-channel scalability are documented in `VMD_Concerns_and_Explanation.md` and `Model_and_Search_Design.md`. Writing those explanations does not substitute for their measured experiments.
