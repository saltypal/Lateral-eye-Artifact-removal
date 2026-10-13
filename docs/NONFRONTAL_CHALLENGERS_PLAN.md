# Posterior and central correction challengers

14 October 2026 IST. User-authorized expansion of exploratory regional experiments.
This permits experiments despite the current frontal/native qualification failure;
it does not promote a teacher, open confirmation sources or authorize neural training.

## Question and frozen context

Can selectively correcting ocular spatial components improve posterior and central
reconstruction over the existing regional hybrid? Preserve frontal K3/alpha2000
full-field predictions (17.112814 dB frontal development mean). No posterior or
central VMD, no AutoVMD and no repetition of the 737-candidate posterior grid.
Full archived central mean was10.903434 dB and posterior13.541703 dB; whole
VMD-context13.162704 dB. Regional dB scores cannot be added.

## Evidence influencing design

- Issa and Juhasz2019: wavelet correction within ocular ICA components and
  contaminated intervals rather than deleting entire components. Primary paper:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6956025/ . Our fixed sym4/MAD thresholds,
  calibration-only Picard fit and EOG-assisted component selection are adaptations;
  no exact author replication or EEG-only inference claim.
- Mannan, Jeong and Kamran2016: process ocular ICs using regression, retaining
  their neural contribution. Primary paper:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4853904/ . Our lagged ridge with frozen
  calibration coefficients replaces their entropy/MAD/extended-RLS pipeline;
  label it component-regression adaptation, not REG-ICA reproduction.
- Kobler et al2020 SGEYESUB: calibration-trained ocular subspace subtraction
  addresses dipole and eyelid activity and protects resting/ERP signals.
  https://pubmed.ncbi.nlm.nih.gov/32497788/ ; author implementation
  https://github.com/rkobler/eyeartifactcorrection . Remains a later challenger
  requiring its actual calibration paradigm/contracts and author-code parity.
- GED identifies contrasts rather than biological truth. Reference-informed
  low-rank transfer is our project extension, not SGEYESUB or the original MWF:
  https://doi.org/10.1016/j.neuroimage.2021.118809 . Measure preservation separately.

## First experiment: six fixed mechanisms

Fit one full-available-montage ICA on each unscored calibration; output only
verified posterior or central/shared rows. Full support includes frontal channels
but never decomposes them with new VMD. Match support and association gate across
ICA correction arms. Use original EEG for every artifact estimate.

1. ICA whole selected-component subtraction: matched rejection control.
2. ICA component ridge: joint HEOG/VEOG lagged calibration regression of selected
   sources, back-projected to a complete artifact field; infer one bounded shared
   gain per output region from scoring input. Targets are absent from inference.
3. Wavelet ICA: subtract large-coefficient contribution from selected sources.
4. Selective wavelet ICA: same contribution restricted to source-amplitude events
   using calibration-rest MAD, dilation and taper. Sustained lateral events are
   evaluated explicitly; a blink-shaped detector is not assumed adequate.
5. Reference-field rank1 transfer: rest-whitened rank reduction of the calibration
   EOG-predicted field; scoring EEG determines bounded region gain.
6. Reference-field rank2 transfer: same parameters, second-rank ablation.

Each mechanism saves three compositions: posterior-only change, central-only
change and both changed. Frozen frontal output and the untouched other branch
remain exact. Baselines: identity, archived VMD-context hybrid and archived
raw-context hybrid (same frontal VMD output). No scoring-target recipe selection.

Fixed settings: ICA association0.4; calibration ridge lags[-10,0,10], penalty0.01;
gain0..8; wavelet sym4, maximum level6 subject to valid length; rest-MAD multiplier3;
source event dilation50samples and taper10samples at200Hz. These are declared
project settings, not published optimal parameters. New recipes are experimental.

## Tasks and acceptance

- NF01 Implement target-free estimation interfaces, disjoint composition and
  focused tests for shape/finite errors, boundaries, zero fields, exact pass-through,
  ICA source preservation, wavelet decomposition reconstruction and rank limits.
- NF02 Pin corpus002 and context-match-full001 manifests; verify every consumed
  array, split, calibration reference and metadata; reject reserved examples.
- NF03 Kaggle five-recipient/50-example pilot with all nine contaminated strata,
  clean controls and all methods. Save predictions, fits, progress and failures
  per example. Pilot metrics cannot establish a winner.
- NF04 After contracts/recovery pass, run same fixed recipes on320 examples/
  32recipients; save per-channel/record/region/condition/input paper metrics and
  source-aware uncertainty. No broad parameter search or pilot-driven recipe edits.
- NF05 Choose a shortlist only on grouped development evidence. Refit/selection
  evaluation excludes recipient and donor identities. Fixed-recipe exploratory
  comparisons using own unscored calibration are distinct from held-out selection.
- NF06 Native OSF ocular/rest validation of frozen shortlisted methods; unchanged
  references, no reconstruction SNR. Previous native failures remain recorded.
- NF07 Eligible author SGEYESUB, selective wavelet replication and ERP/BCI tests
  follow their source/calibration contracts; no guessed anatomy or head model.

## Reporting and stop rules

Primary paired MSE, temporal RRMSE, ordinary Pearson CC; output SNR and improvement
separate. Also report spectral and covariance preservation, runtime, fit failures
and pass-through. Gates unchanged: clean modification<=.01, alpha/beta<=.5dB,
covariance<=.02, CC deterioration<=.005. Evaluate comparator-specific deterioration
against both frozen matched context baselines and label comparator identity.
Controlled artifact fields are regression-generated and may favor regression;
EOG amplitude coupling is still imperfect. These are development experiments,
not independent validation or production qualification. Native tests required.

Retain all failures. No method wins because it includes VMD. Best whole-system
SNR is assessed from composed waveforms, never summed regional dB. If the fixed
challengers fail, report their shortfall and use the existing frozen baseline.
