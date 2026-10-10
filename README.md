# Regional VMD EOG removal

Kaggle notebooks execute and explain the complete research workflow. Reusable
algorithms live in `src/vmd_eog`. Git branch
`research/vmd-eog-removal-notebooks` is the source of truth. Each fresh launch
resolves the latest published SHA and pins it for the whole run.

Local work is authoring/static review/Git/CLI orchestration. Numerical computation
is guarded to Kaggle. Run-specific outputs are saved and retrieved under `results`.
See `TARGETS.md` and `configs/campaign.json` for the completion and acceptance contract.

Teacher: frontal VMD + joint HEOG/VEOG projection, posterior GEVD-MWF/ICA with raw
frontal support. Signed left/right context retains lateral polarity. Estimates
share one input and are subtracted once. User permits the EEG-only student phase
automatically if all classical gates pass; gate artifacts enforce this condition.

From this project directory, inspect or retrieve an archived run with:

```powershell
python tools/kaggle_campaign.py status --run-id vmd-001
python tools/kaggle_campaign.py retrieve --run-id vmd-001 --file-pattern '(summary|run_spec|execution_state)\.json$'
```

An independent new experiment resolves the latest published source once:

```powershell
python tools/kaggle_campaign.py submit --stage vmd --run-id vmd-new --profile full --kernel-source satyapaladugu/vmd-eog-corpus/2
```

Use `--source-sha <exact archived SHA>` for reproduction. Keep parent kernel
versions pinned and give every rerun a new ID. `docs/SCIENTIFIC_EXPLANATION.md`
explains mode vectors, frequency bins, regional cooperation, metrics and cost.
