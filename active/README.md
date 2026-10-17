# Klados and OSF: readable replacement pipeline

This is the active implementation from 17 October 2026. Start here; the older
root-level modules and numbered notebooks are historical, not dependencies.

## The experiment in plain language

1. Klados supplies paired contaminated/reference EEG and HEOG/VEOG. It supports
   reconstruction SNR, MSE/RMSE, temporal RRMSE and ordinary Pearson correlation.
   Its electrode row order and participant identities remain **unverified**.
   Therefore the Klados comparison is explicitly nonregional.
2. OSF supplies named electrodes, eye-movement/blink trials and resting trials.
   This supports verified frontal/posterior/central comparisons, but no clean
   target exists: **no reconstruction SNR is reported for native OSF**.
3. Fixed VMD uses K=3, alpha=2000. Modes are tested against both ocular references;
   only their reference-explained contribution is subtracted. The reconstruction
   residual is retained. OSF VMD runs only on verified frontal electrodes.
4. Direct EOG ridge regression, ICA component regression and GEVD–MWF are matched
   controls. Posterior and central signals never receive VMD in regional runs.
5. The historical 17.9272 dB checkpoint is an exposed, EOG-assisted learned model.
   Replaying it is provenance verification, not a fresh accuracy claim.

LEMON, VMD–SOBI, anchored corrections and broad parameter searches are absent.
AutoVMD source and historical outputs remain unchanged in the preserved legacy
snapshot. The initial execution does not restart its search.

## Files to read, in order

- `PLAN.md`: tasks, six-hour priorities and acceptance boundaries.
- `config.json`: every selected parameter and source version.
- `src/eog_clean/data.py`: datasets, verified channel groups and trial boundaries.
- `src/eog_clean/methods.py`: regression, ICA and MWF correction equations.
- `src/eog_clean/vmd.py`: fixed decomposition and mode diagnostics.
- `src/eog_clean/metrics.py`: exact scoring definitions.
- `src/eog_clean/experiment.py`: calibration, comparison and recoverable outputs.
- `notebooks/Klados_OSF_Comparison.ipynb`: the single execution notebook.

All numerical work and figures run on Kaggle. Local work authors source and
submits/retrieves jobs. Every saved version pins an exact published Git commit.
Results live under `results/klados-osf-20261017/<run_id>/` on Kaggle and after
retrieval. A completed notebook is not automatically a qualified scientific result.

## Scientific limits

Low EEG–EOG correlation alone does not establish preservation. We also measure
resting modification, alpha/beta spectral change and covariance change, and test
paired clean inputs with their original EOG references. Calibration and scoring
are separate; fitting a regression does not prove its coefficients are pure EOG
transfer coefficients. Original Klados contamination is itself linear in EOG,
which favors regression and must be disclosed.

Published sources: [Klados](https://doi.org/10.1016/j.dib.2016.06.032),
[VMD](https://doi.org/10.1109/TSP.2013.2288675),
[ICA–regression](https://doi.org/10.3389/fnhum.2016.00193),
[MWF](https://doi.org/10.1088/1741-2552/aaac92), and
[EEGOAR-Net](https://github.com/dmarcos97/EEGOAR-Net).
Our ICA uses ridge regression rather than that paper's RLS; our MWF uses an
instantaneous regularized covariance model. These are labelled adaptations.
GEVD–MWF is not EEGOAR-Net. EEGOAR-Net/ASR require separate validated adapters.
