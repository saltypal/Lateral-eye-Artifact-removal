# EEG ocular artifact removal

One project, three frontal evaluation notebooks plus one pretrained closeout,
and one current results document.

- [PLAN.md](PLAN.md): the task checklist and regional decisions.
- [RESULTS.md](RESULTS.md): measured results and their limits.
- [docs/PAPERS.md](docs/PAPERS.md): paper methods, datasets and replication status.
- [notebooks/Frontal_Comparison.ipynb](notebooks/Frontal_Comparison.ipynb): Kaggle execution.
- [notebooks/Frontal_OSF_Correlations.ipynb](notebooks/Frontal_OSF_Correlations.ipynb):
  independent paired-clean CC check and frontal-only native OSF transfer.
- [notebooks/Frontal_Mode_Clean_Correlations.ipynb](notebooks/Frontal_Mode_Clean_Correlations.ipynb):
  every saved mode versus clean EEG/HEOG/VEOG and native reference audit.
- `src/eog_clean/`: reusable implementation; `tests/`: numerical contracts.
- `configs/frontal_six.json`: active six-method comparison and data identities.
- `configs/experiment.json`: completed baseline pilot configuration.
- `archive/`: the old implementation, preserved rather than mixed into current code.

**Active data: Klados and original OSF. No LEMON.** Fixed VMD starts at K=3,
alpha=2000. AutoVMD is preserved unchanged; its broad search is not restarted.

Numerical processing, training, tests and figures run on Kaggle. Local work
handles authoring, Git and artifact retrieval. Every run pins a published SHA.

```powershell
python tools/frontal.py submit --run-id frontal-six-smoke-001 --smoke
python tools/frontal.py status
python tools/logs.py --run-id frontal-six-smoke-001
python tools/retrieve.py --run-id frontal-six-smoke-001
```

The completed full comparison is `frontal-six-full-001`. Open its downloaded
results directly at `notebooks/executed/frontal-six-full-001.ipynb`.

For the frozen frontal OSF follow-up:

```powershell
python tools/frontal_osf.py status
python tools/logs.py --run-id frontal-osf-correlations-001
python tools/retrieve.py --run-id frontal-osf-correlations-001
```

OSF uses five exact common frontal names: F3/F4/F7/F8/Fz. FPz is not substituted
for missing FP1/FP2. Klados labelled NPZ copies contain channel names and source
row indices. Native OSF receives before/after HEOG/VEOG correlations and resting
preservation proxies; paired clean EEG correlation is available on Klados only.

The committed notebook is the readable template; submission embeds the exact
published source SHA. Results remain under `results/<campaign>/<run_id>/`.

The six-method comparison applies the Klados paper-listed electrode order as
requested. Frontal rows are FP1/FP2/F3/F4/F7/F8/Fz; the row binding remains an
explicit assumption. OSF has named channels but no paired clean target.
Executed result notebooks are saved after each cell and included in retrieval.
Existing mentor documents remain untouched.

The completed native run is `frontal-osf-correlations-001`; its actual outputs
are in `notebooks/executed/frontal-osf-correlations-001.ipynb`. The native mode
reference audit completed as `frontal-mode-clean-audit-002`; its executed
notebook is in `notebooks/executed/`. Status and retrieval are reproducible:

```powershell
python tools/frontal_modes.py status
python tools/logs.py --run-id frontal-mode-clean-audit-002
python tools/retrieve.py --run-id frontal-mode-clean-audit-002
```

Paired accuracy and native preservation are separate findings. The historical
frontal model reaches 16.134624 dB on reused Klados records, but changes native
OSF resting input by 63.63% mean relative RMS. See RESULTS.md before selecting
a method or making a deployment claim.

Pretrained-only extension: `notebooks/04_Frontal_Pretrained_Closeout.ipynb`
records release eligibility for EEG-WNet and Diff-ADN and carries forward
verified frontal results. Neither new model has a verified compatible public
checkpoint; no training or substitute architecture is used.

```powershell
python tools/frontal_pretrained.py status
python tools/logs.py --run-id frontal-pretrained-closeout-001
python tools/retrieve.py --run-id frontal-pretrained-closeout-001
```

The discovery audit can be refreshed with `python tools/audit_frontal_pretrained.py`.
It updates the evidence snapshot, so reviewed eligibility/configuration identities
must be refreshed and committed before another notebook submission. A repository
search hit alone never authorizes treating an unverified checkpoint as evaluated.
