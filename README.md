# EEG ocular artifact removal

One project, one comparison notebook, one results document.

- [PLAN.md](PLAN.md): the task checklist and regional decisions.
- [RESULTS.md](RESULTS.md): measured results and their limits.
- [docs/PAPERS.md](docs/PAPERS.md): paper methods, datasets and replication status.
- [notebooks/Frontal_Comparison.ipynb](notebooks/Frontal_Comparison.ipynb): Kaggle execution.
- `src/eog_clean/`: reusable implementation; `tests/`: numerical contracts.
- `configs/experiment.json`: declared parameters and dataset identities.
- `archive/`: the old implementation, preserved rather than mixed into current code.

**Active data: Klados and original OSF. No LEMON.** Fixed VMD starts at K=3,
alpha=2000. AutoVMD is preserved unchanged; its broad search is not restarted.

Numerical processing, training, tests and figures run on Kaggle. Local work
handles authoring, Git and artifact retrieval. Every run pins a published SHA.

```powershell
python tools/run.py submit --run-id frontal-comparison-001 --pilot
python tools/run.py status
python tools/logs.py --run-id frontal-comparison-001
python tools/retrieve.py --run-id frontal-comparison-001
```

The committed notebook is the readable template; submission embeds the exact
published source SHA. Results remain under `results/<campaign>/<run_id>/`.

Klados supports paired reconstruction metrics but its electrode row map remains
unverified. OSF has named channels but no paired clean target. These limitations
are explicit in every comparison. Existing mentor documents remain untouched.
