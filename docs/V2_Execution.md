# VMD EOG removal implementation and execution

Workspace: `D:\Bunker\BrainComputerInterface\Lateral Eye Dataset\VMD_EOG_Removal`.
The Git modules are the implementation authority. Kaggle notebooks are the
execution interface for numerical tests, decomposition, training and scoring.

## Stage order and acceptance

1. `contracts`: numerical tests on synthetic fixtures, without source data.
2. `provenance`: checksum original sources and record electrode/participant evidence.
3. `split-freeze`: participant and duplicate-clean groups, before generating windows.
4. `corpus-fit`, `corpus-build`: fit eligibility on training sources and persist
   materialized controlled examples. OSF recipient references have unknown native
   cleanliness; their recovery score is not real clean-EEG SNR.
5. `teacher-cache`, `teacher-search`: K=3..10, alpha=250..4000, mode frequencies,
   convergence and preservation-constrained correction searches.
6. `teacher-oof`: require proof that fitted/selected recipe sources exclude each
   target recipient and donor. EOG availability alone never establishes OOF.
7. `student-paired`, `student-distill`: EEG-only regional TCN-BiGRU, clean identity,
   reconstruction, spectrum, covariance and optional SNR shortfall supervision.
8. `select-freeze`: select from preservation-feasible development checkpoints.
9. `final-eval`: fixed model and reserved grouped evaluation; no further tuning.
10. `export`: fresh CPU reload using a development fixture, without reopening test.

Every parent attachment declares its exact run ID, stage, kernel reference and
run-manifest SHA256. Every saved parent file is checked again. A run ID cannot be
resubmitted; failed and resumed jobs receive new IDs. Kaggle run output is saved
under `results/<campaign_id>/<run_id>` and retrieved into the same local hierarchy.

## Reproducible commands

Run from this workspace in PowerShell:

```powershell
python tools/kaggle_campaign.py auth-check
python tools/kaggle_campaign.py submit-v2 --run-spec configs/runs/<run-id>.json
python tools/kaggle_campaign.py status-v2 --run-id <run-id>
python tools/kaggle_campaign.py retrieve-v2 --run-id <run-id>
```

The RunSpec pins a published 40-character Git commit. Each notebook creates a
fresh checkout and interpreter, installs pinned research dependencies, records
hardware/package versions and executes a module in a fresh process. The portable
launcher requires explicit opt-in for Local/Colab execution; this study remains
Kaggle-only. CUDA/CPU are supported, TPU training remains unqualified.

## Limits that accompany results

The earlier 17.9272 dB result requires runtime EOG and VMD and was development
exposed. It cannot be reported as this student's performance. Unknown Klados
electrode order is routed through the shared fallback. Five modes are a search
candidate, not a justified fixed choice or a guaranteed mapping to EEG bands.
VMD vectors are numerical band-limited modes; mode number is not a stable
biological identity across windows. The main student uses full-rate temporal
filters and offline bidirectional recurrence; no transformer or streaming claim
is needed. Production accuracy and journal-level evidence require completed
grouped experiments, measured preservation, external validation and export checks.
