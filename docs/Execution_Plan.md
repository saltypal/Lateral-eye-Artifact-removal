# Region aware EOG execution plan

Updated 2026-10-04. Active mode: Super Code. The user authorizes private Kaggle execution, VMD-first development, parameter search and a frontal/posterior hybrid. Computation and model tests run on Kaggle; local tasks only author, statically check, version, package, upload and retrieve.

## Modules and gates

| Phase | Goal and inputs | Outputs | Dependencies | Owner and acceptance |
|---|---|---|---|---|
| A | Package full original folder and audit it remotely | Every-file hashes, shapes, label/participant/electrode provenance, exclusions | Kaggle OAuth, source data, exact Git SHA | Codex; all hashes match, arrays align, at least one usable OSF session |
| B | Tune classical VMD-first candidates on development records | K/alpha frequency diagnostics, correction/preservation metrics, train-only FCM, selected config | A | Codex with inspectable algorithm; identity/alignment/calibration tests pass; development improvement does not authorize superiority |
| C | Compare shared versus verified regional correction | Per-region OSF suppression/preservation proxies, unknown-location fallback | B plus verified electrode names; paired anatomical Klados gate requires provenance | Joint research checkpoint; use no guessed channel order |
| D | Train one EEG-only shared-channel student for both artifact classes | Best/last checkpoints, frozen Klados test predictions, OSF Protocol Z | B/C feasibility, independent splits, preservation constraints | Codex; paired quality, mask/permutation/reload tests, recorded parameters and runtime |
| E | Adapt gate/normalization on allowed studies | Participant-purged Protocol A, matched recent baselines, grouped statistics | D and audited annotation coverage | Codex; test study and overlapping participants never supervise adaptation |
| F | Separately train and evaluate streaming model | Causality/state/boundary/latency/quality evidence | D; independent online preprocessing contract | Codex; full acceptance tests in research contract |

The smoke profile is a feasibility experiment, not the full five-fold/three-seed study. No DOCX rewrite or claim of complete EOG removal/superiority follows from a smoke result. The user's original architecture and notebooks remain unchanged.

## Repository and reproducible operations

Work is in the isolated clone under `EOG_Artifact_Framework/region_aware_research`, branch `research/region-aware-eog-kaggle`, in the existing remote `saltypal/Lateral-eye-Artifact-removal`. The reusable root package remains `eog_vmd_fcm_bgru`. No legacy `src` import is added.

Run these in the isolated repository:

```powershell
python tools/kaggle_campaign.py auth-check
python tools/kaggle_campaign.py prepare --source "D:\Bunker\BrainComputerInterface\Lateral Eye Dataset\complete_dataset"
python tools/kaggle_campaign.py package-opaque
python tools/kaggle_campaign.py upload-opaque
python tools/kaggle_campaign.py dataset-status
python tools/kaggle_campaign.py submit --phase audit --sha EXACT_40_CHARACTER_COMMIT
python tools/kaggle_campaign.py status --phase audit
python tools/kaggle_campaign.py retrieve --phase audit
```

The controller isolates CLI configuration so an obsolete legacy key cannot shadow OAuth. It never prints credentials. The dataset is private, contains all constituent files, and uses `other` license metadata because constituent upstream terms apply. It does not assert one public redistribution license for the entire bundle.

The commands above describe preparation of a new dataset version. Version 3 is already complete: use `inventory-check` and attach it rather than uploading it again. Source-only/report phases attach the required saved kernel outputs without unpacking the 3 GB original bundle. The OAuth controller refreshes expiring credentials through the official SDK.

The notebook starts from a new temporary exact-commit clone, records environment and package versions, and rejects non-Kaggle experiments by default. For an explicitly authorized portable Local/Colab run set `EOG_ALLOW_NON_KAGGLE=1`, `EOG_DATA_ROOT`, `EOG_SOURCE_MANIFEST`, and a persistent `EOG_PERSISTENT_RESULTS`. CUDA is detected; multiple GPUs are recorded but not automatically used; unsupported TPU falls back explicitly to CPU rather than pretending CUDA code works on XLA.

Saved Kaggle kernel-version output is the persistence unit. `/kaggle/working` alone is temporary. Retrieve completed outputs and, when resuming training, attach a private versioned checkpoint dataset. Aborted jobs do not guarantee persistence.
