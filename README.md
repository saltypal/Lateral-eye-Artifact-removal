# Regional VMD EOG removal

Kaggle notebooks execute and explain the classical research stages. Reusable
algorithms live in `src/vmd_eog`. Git branch
`research/vmd-eog-removal-notebooks` is the source of truth. Each fresh launch
resolves the latest published SHA and pins it for the whole run.

Local work is authoring/static review/Git/CLI orchestration. Numerical computation
is guarded to Kaggle. Run-specific outputs are saved and retrieved under `results`.
See `TARGETS.md` and `configs/campaign.json` for the completion and acceptance contract.

Candidate teacher: frontal VMD + joint HEOG/VEOG projection, posterior GEVD-MWF/ICA with raw
frontal support. Signed left/right context retains lateral polarity. Estimates
share one input and are subtracted once. User permits the EEG-only student phase
automatically if all classical gates pass; gate artifacts enforce this condition.

The pipeline is not yet research-qualified. VMD solver/parity and correction
mechanism contracts passed on Kaggle, but the 15–20 dB target and full regional
preservation gates remain unestablished. Neural notebooks currently define
planned interfaces; model implementations and training remain gated. Do not
describe an authored notebook, synthetic fixture or mechanism pilot as final
accuracy evidence.

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

The review that includes complete-trial native evaluation waits for immutable
parents before submission. It remains strictly classical:

```powershell
python tools/phase_a_chain.py --vmd-run vmd-001 --posterior-run posterior-001 --corpus-run corpus-002 --contracts-run contracts-007 --fixture-run classical-fixture-005 --regional-run regional-paper-001 --native-run native-protocol-full-001 --review-run review-paper-002 --interval 45
```

Notebook 04a diagnoses VMD reconstruction, correction gates and ocular baseline
handling; notebook 07b evaluates complete native conditions. Native OSF receives
paper-based correlation/RMSE/PSD measures, never clean-reference reconstruction
SNR. Five-participant/eight-second null sampling is an explicitly declared
adaptation. See `docs/EVALUATION_PROTOCOL.md` for source equations and limitations.

Notebook 04b separately tests the prescribed VMD paper's mode-domain SOBI idea,
with disclosed EOG/entropy selection and delay-bank adaptations. Its bounded
pilot/comparison cannot select a final teacher. To launch a new pilot after
publishing source, use:

```powershell
python tools/kaggle_campaign.py submit --stage vmd-sobi --run-id vmd-sobi-new --profile pilot --kernel-source satyapaladugu/vmd-eog-corpus/2 --kernel-source satyapaladugu/vmd-eog-vmd/1
```
