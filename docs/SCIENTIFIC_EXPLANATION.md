# What the implementation estimates and what the evidence can establish

## Inputs, outputs and information boundaries

EEG is a matrix `[channels, samples]`. A 1024-sample window at 200 Hz spans
5.12 seconds. HEOG and VEOG form a separate `[2, samples]` reference matrix.
Their order comes from verified names, never guessed indices. Each expert
returns an artifact matrix with the EEG shape. The final output is original
EEG minus the composed artifact estimate. An expert never receives a paired
clean target during inference.

The classical cleaner is an offline, EOG-assisted candidate. The future
student is EEG-only. An acceptable teacher does not establish that an EEG-only
student can reproduce it: ocular and neural sources can overlap in frequency
and space, so some information is intrinsically ambiguous without references.

## Why five modes cannot be assumed

VMD jointly estimates `K` time vectors and their center frequencies. A mode
vector `u_k[t]` contains one amplitude at every sample; it is not a scalar
frequency, an electrode or a predefined physiological band. The variational
objective penalizes each mode's bandwidth after demodulation around its own
center. Alternating spectral updates revise the vectors and energy-weighted
centers until the declared tolerance or iteration cap is reached.

The number of conventional EEG bands is not an argument for `K=5`. A blink
and slow neural activity may inhabit the same mode, while one neural rhythm
can spread across modes. Whole-mode deletion may therefore remove useful EEG.
The projected alternative removes only the part of selected modes explained
by the joint ocular references, keeping their remaining activity.

The campaign decomposes the same frontal examples for `K=3..10` and alpha
250, 500, 1000, 2000 and 4000. RMS normalization makes the stopping criterion
comparable across units. Uniform initialization, no forced DC, tau=0,
tolerance `1e-6` and cap 2000 are explicit. Every mode vector, center,
bandwidth, overlap, residual and convergence outcome is saved. Centers closer
than 1 Hz trigger a diagnostic; this is not an automatic rejection rule.
Grouped recovery, preservation and cost determine whether additional modes
are useful. A fitted `K=5` result requires evidence from this search.

The residual is `original - sum(modes)` and is retained. The cleaner subtracts
an estimated artifact from the original window instead of reconstructing
only accepted modes. Iteration-limited or failed decompositions pass through
and remain in the score denominator.

## Frequency bins and correlation bins are different

Preservation metrics use delta `[0.5,4)`, theta `[4,8)`, alpha `[8,13)`, beta
`[13,30)` and upper `[30,40)` Hz from Welch spectra. These are analysis bins;
they do not constrain VMD centers or assign mode numbers to EEG bands.

An anatomical region is derived from verified electrode names. Fp/AF/F are
frontal; P/PO/O are posterior; remaining recognized central electrodes use
the shared branch. Unknown electrodes remain unknown. Frontal and posterior
regions are not inferred from the Klados array order.

Mode association is the maximum absolute signed EEG-mode/EOG correlation over
the declared finite lag bank. Signs are retained for diagnostics, so opposite
lateral polarities do not cancel. Thresholds 0.2/0.4/0.6/0.8 are searched;
their optimal value is not assumed. The fuzzy-clustering ablation uses mode
association, low-frequency fraction, kurtosis and relative mode energy.
Feature scaling and cluster centers are fitted outside the held-out recipient
and donor bucket. Clustering groups statistical features, not known EEG bands.

## Frontal and posterior cooperation

Frontal channels receive independent VMD correction. Posterior candidates use
calibration-only regularized GEVD-MWF or extended Picard ICA. Comparison arms
provide posterior EEG alone, raw frontal support, a full montage with only
posterior output correction, or signed frontal summaries. Summaries retain
left, right and midline evidence; signed right-minus-left supports lateral
movement, and common frontal activity supports blink evidence. Availability
flags distinguish absent context from an observed zero-valued channel.

The posterior input uses original frontal EEG. Removing frontal ocular evidence
before fitting the posterior separator could hide the signal needed to
identify ocular components. Frontal and posterior corrections are computed
against the same original EEG and applied only to their designated output
channels. Shared/central correction is disjoint. The composition subtracts once.

Identical posterior experts with direct regression replacing frontal VMD form
additional matched ablations. A gain over whole-montage direct regression alone
could arise from the posterior method; it would not establish VMD's contribution.

## SNR and preservation

Paired reconstruction SNR is `10 log10(sum(target^2)/sum((output-target)^2))`.
The target is retained recipient EEG in controlled mixtures, or supplied clean
EEG in legacy paired Klados. Low ocular burden does not establish biological
cleanliness of a LEMON reference. Native OSF does not have this target, so it
receives no reconstruction SNR.

Recipient-level means, separate blink/lateral/mixed results and participant
bootstrap intervals accompany SNR. The clean-control gate is worst-recipient
mean relative modification <=1%, alpha/beta error <=0.5 dB, relative covariance
error <=0.02 and paired-correlation deterioration <=0.005 versus direct
regression. Clean controls deliberately retain ocular reference signals while
injecting no artifact; the cleaner cannot use the known condition to pass them
through. A near-zero EEG output would reduce reference association but fail
reconstruction and preservation.

The native association metric reports separate absolute HEOG/VEOG associations
and joint reference-explained variance. Joint R-squared uses a pseudoinverse
projection, tolerates collinear references and is invariant to their polarity.
It remains an association proxy: neural activity can correlate with eye events,
and reduced association is not proof of complete artifact removal. Modification
of resting OSF EEG is reported as a proxy, not a clean-truth gate.

Controlled artifact fields come from calibration-fitted lagged HEOG/VEOG donor
maps on exact-name montage intersections. This mixture family can favor the
same class of regression methods. Its SNR cannot establish universal clinical
or native EEG superiority. OSF donors also carry their declared historical
development exposure. Those limits remain visible in the final report.

## Computational cost and scaling

For `I` iterations, `K` modes and window length `T`, VMD needs approximately
`O(I*K*T)` spectral updates plus FFT setup. Each frontal channel is independent
and can be batched or parallelized. The rolling implementation retains two
spectral iterates instead of the entire iteration history. Posterior covariance
and generalized eigendecomposition scale with the embedded channel dimension;
lag embedding increases that dimension and cubic fit cost. ICA also depends on
rank, calibration duration and convergence, which must be measured.

A transformer is not required. A shared temporal neural encoder, pooled regional
context and per-channel residual outputs can support variable channel counts
without a separate model for each cap. This is a planned student property until
its mask, permutation, missing-context, CPU reload and 10–50-channel benchmarks
pass after the classical gate. Offline filtering and bidirectional recurrence
do not constitute a causal streaming implementation.
