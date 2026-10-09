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
