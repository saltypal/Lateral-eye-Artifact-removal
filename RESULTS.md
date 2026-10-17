# Results: current verified checkpoint

Verified Kaggle pilot: `satyapaladugu/klados-osf-clean-comparison/2`, run
`klados-osf-pilot-002`, source `da7e2778fcc0187565c23064829bf67357584ea7`.

| Evidence | Measured result | Scope |
|---|---|---|
| Numerical contracts | 20 passed in 5.22 seconds | Software correctness fixtures |
| Historical frozen VMD + EOG model | 17.927210 dB output SNR | Eight reused Klados records, all 19 unnamed rows, 24 archived windows |
| New paired classical pilot | Eight records completed | First eight Klados records; different cohort/preprocessing from historical replay |
| Native frontal/regional pilot | One OSF session, four scoring trials | study01_p02; rest, blink, lateral and vertical |
| Frontal VMD diagnostics | 168 decompositions; zero invalid/capped | Three modes each; both HEOG and VEOG correlations saved |
| Figures | 11 figures rendered in executed notebook | Verification of local retrieval/visual inspection is separate |

The historical score reproduced within 0.001 dB. It requires VMD and EOG at
inference and is development-exposed. It is not a fresh frontal SNR result.

The frozen model's OSF frontal adaptation changed resting EEG by 56.95% in the
pilot and failed preservation. It is not selected as the regional method.
Fixed VMD left the sampled resting trial unchanged; that alone does not establish
sufficient ocular removal or generalization. No new regional winner is declared.

All detailed numbers, failure logs, predictions, mode diagnostics and preservation
checks remain in the saved Kaggle outputs. The first pilot version failed during
dependency setup; its evidence is retained in `docs/WORK_LOG.md`.

The four requested literature methods have not yet completed matched evaluation.
Their paper-reproduction gaps and next tasks are in `docs/PAPERS.md` and `PLAN.md`.
