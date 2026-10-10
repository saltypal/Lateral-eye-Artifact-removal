# VMD–SOBI frontal candidate: bounded implementation plan

## Why this candidate is necessary

Reference-projected VMD alone has not beaten direct regression. In the fixed
K4/alpha2000 Fz mechanism experiment, two-stage channel/mode gating protected
the scored clean inputs but lag-bank-1 mode-threshold-0.2 correction scored
8.8051 dB versus 10.4617 dB for matched direct regression. This cannot qualify
the teacher. The VMD solver's parity and reconstruction checks passed, so a
different correction mechanism is warranted before rejecting decomposition.

[Xiong et al. 2024, sections 2.1–2.4](https://mdpi-res.com/d_attachment/sensors/sensors-24-01642/article_deploy/sensors-24-01642-v2.pdf?version=1709543478)
uses a detector, VMD, SOBI on its modes and approximate-entropy source
selection. This candidate adapts the mode-domain SOBI step, with available
HEOG/VEOG supplying additional source evidence. It is not a full replication
of their SVM/GA pipeline or an already superior approach.

## Module contracts

| Component | Input | Output | Acceptance |
|---|---|---|---|
| Symmetric joint diagonalizer | delayed symmetric covariance matrices | orthogonal rotation, objective and convergence | known commuting matrices diagonalized; objective decreases |
| SOBI | finite mode vectors `[K,T]`, declared delays | whitened sources, unmixing/mixing operators, mean and rank | recover identifiable mixed autoregressive sources; retain rank residual and units |
| Approximate entropy | one source, m=2, r=0.15 SD | ApEn including self matches | independent direct-count oracle; amplitude invariance; explicit constant rule |
| Candidate correction | modes, EOG, SOBI operator, raw-channel trigger | mode-domain artifact, summed once into raw EEG | targets never enter inference; failed fits pass through |
| Notebook 04b | corpus/2 and VMD/1 | paper metrics, failure counts, clean costs, diagnostic figures | Kaggle contracts pass, then bounded pilot; no model authorization |

The primary agent authors modules; all numerical checks and figures run on
Kaggle. No neural architecture or student training is introduced.

## Bounded search and interpretation

The pilot reuses immutable K4/alpha2000 vectors and the original lag-zero
penalty/strength. It evaluates two explicit SOBI delay banks: samples
`[1,2,4,8,16,32]` and `[1,5,10,20,40,80]` at 200 Hz. These are declared choices;
the inspected paper does not specify its complete delay bank. Center/scale
normalization and a relative covariance-rank criterion avoid unit-dependent
source fits. Cap Jacobi sweeps at 100 with rotation tolerance 1e-7; record
convergence, diagonalization cost and off-diagonal residual.

Compare whole identified-source subtraction, reference-projected source
subtraction and entropy-plus-EOG selection. Use the existing association
threshold bank, retain residual and means, and use the separate raw-channel
trigger to protect clean windows. Approximate-entropy threshold 0.4, embedding
2 and radius fraction 0.15 are explicitly identified adaptations of the paper;
our 5.12-second windows and EOG-assisted selection differ.

The pilot first covers 12 development examples. A passing software pilot
permits a 128-example mechanism comparison, not teacher approval. Promotion
requires recipient/donor-grouped parameter selection, full frontal channels,
regional/posterior integration, native suppression/preservation and all
existing acceptance gates. Confirmation sources remain closed.
