# Fresh VMD EOG removal: task ledger

Started 2026-10-10. Fresh project, independent source modules and notebooks.
Numerical work runs exclusively on Kaggle. No neural training before the
classical experiment checkpoint is reviewed.

- [x] Verify Kaggle authentication as satyapaladugu.
- [x] Preserve prior draft and create a clean VMD_EOG_Removal project.
- [ ] Define preprocessing, region metadata and source-group splits.
- [ ] Pass VMD reconstruction/parity, lag alignment and posterior routing tests.
- [ ] Run K=3..10 and alpha={250,500,1000,2000,4000} sweep.
- [ ] Save center frequencies, duplicate-mode flags, convergence and runtime.
- [ ] Compare identity, direct EOG ridge, VMD all-channel, frontal VMD + posterior MWF, and frontal VMD + posterior ICA.
- [ ] Evaluate paired-clean preservation and HEOG/VEOG suppression separately.
- [ ] Select a feasible classical recipe from development sources only.
- [ ] Freeze recipe and run grouped reserved evaluation.
- [ ] Decide whether the classical evidence supports neural distillation.
- [ ] Implement and train EEG-only student from source-separated teacher targets.
- [ ] Verify CPU export, dynamic cap handling and independent external validation.

The earlier draft contract run had 55 passed and 2 failed tests involving lag
semantics. It is not fresh-project validation and is not an SNR result.
Klados electrode and participant labels are unknown. OSF controlled recipient
reference recovery is not paired native-clean EEG SNR. Report these cohorts
separately. Targets are 15 dB, 20 dB stretch, with <=1% paired-clean modification,
<=0.5 dB alpha/beta distortion, <=0.02 relative covariance distortion and <=0.005
Pearson deterioration. These are engineering gates, not biological guarantees.
