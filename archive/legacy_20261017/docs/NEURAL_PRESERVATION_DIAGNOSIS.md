# Clean preservation diagnosis and matched loss pilots

The five-epoch fixed-band and student pilots reached development output SNR
7.772145 and 7.364268 dB respectively. Both failed the four measured preservation
gates, with worst-source clean relative error near 0.479. They remain screening
evidence and are not qualified teachers or deployment models.

1. Run `neural-diagnosis` on Kaggle with the frozen corpus and both completed
   prediction sets. Check exact clean input/target equality and zero added
   artifact, mixture closure, masks, saved target/input identities, complete
   validation coverage and identity-control preservation. Record projection gains
   to distinguish amplitude shrinkage from other waveform errors.
2. After those contracts pass, run fixed-band and TCN-BiGRU pilots with the existing
   `preservation` loss. Retain outer fold 0, inner fold 0, seed 42, learning rate
   0.001, five epochs and the source-balanced pilot data. Keep configured identity,
   log-PSD and covariance weights 1.0/0.1/0.05. These are project design weights,
   not values prescribed by a paper or guaranteed to meet the gates.
3. Compare loss profiles using the same source/condition outputs and clean controls.
   Investigate any contract failure before further training. If scientific
   preservation fails again, retain the failure and diagnose it before selecting
   a loss or publishing a qualification claim.
4. Continue the VMD cache jobs. Preserve the already launched MSE regional pilot.
   Hold subsequent GPU search jobs in the original backlog until diagnosis and
   matched pilot outputs have been reviewed by the engineering agent.

Reserved confirmation sources stay closed. No clean/scoring labels enter the
deployment forward interface. The engineering acceptance limits are unchanged.
Passing software contracts is distinct from passing preservation and accuracy.
