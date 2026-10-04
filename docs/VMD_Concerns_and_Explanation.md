# VMD parameter selection and implementation explanation

These are explicit user concerns to address in the final experiment write-up. They are not considered resolved merely by writing an explanation. Measured results must accompany the search and complexity sections.

## Did the old notebook perform the requested center frequency sweep

Inspection on 2026-10-04 found that `VMD_Hyperparameter_Sweep_OSF_Klados_Baseline.ipynb`, cell 10, actually sets `K_VALUES = [5]` and `ALPHA_VALUES = [1000]`. It evaluates one configuration. Cell 13 plots the centers for K=5; that is not a K=3..10 sweep. The current report states K=5 and alpha=1000 without evidence that they are optimal. We must correct that limitation, not retroactively claim an executed grid search.

## What VMD solves

For one EEG channel sampled at 200 Hz, the input is a vector x of T real amplitudes. At T=1000 it covers five seconds. VMD asks for K oscillatory, approximately narrow-band vectors u_1 through u_K whose sum approximates x. Each u_k has T amplitudes, just like x. The output matrix is K by T; for 19 simultaneous channels it is 19 by K by T when decomposition is channelwise.

Unlike empirical mode decomposition, VMD does not repeatedly peel off local envelope means. It jointly estimates all modes and their center frequencies through an optimization problem. Calling the outputs IMF1, IMF2, etc. is common shorthand, but `VMD mode` is the clearer term here. Mode 1 is not automatically EOG, and mode 3 is not automatically alpha rhythm.

The underlying objective minimizes the summed bandwidth of analytic modes after shifting each one to baseband, subject to reconstructing the input. The implementation iterates in the Fourier domain: subtract the other current mode spectra from the input, use a frequency-localized update centered on omega_k, update omega_k from the energy-weighted center of its positive-frequency spectrum, and update the reconstruction multiplier when tau permits it. Inverse Fourier transformation produces the real time-domain mode vectors.

The classic normalized-frequency update has a denominator of the form `1 + alpha * (f - omega_k)^2` in vmdpy. Larger alpha penalizes spread away from a center more strongly; alpha is not a low-pass cutoff in Hz. Implementation constants and normalized-frequency conventions matter, so penalty values are compared using one pinned implementation.

The decomposition residual is r = x - sum(u_k). We retain r explicitly. When suppressing an estimated artifact a, output is x - a, not just the sum of retained modes: the latter can silently discard r and alter the input even when no artifact is selected. Odd-length signals receive explicit padding before decomposition and trimming afterward; input, clean target, EOG and labels remain sample aligned.

Primary resources: [original VMD DOI](https://doi.org/10.1109/TSP.2013.2288675), [authors' VMD source](https://math.montana.edu/dzosso/code/), [vmdpy implementation](https://github.com/vrcarva/vmdpy/blob/master/vmdpy/vmdpy.py). The cited IEEE document 10940839 was inaccessible through the web tool on 2026-10-04; its specific recommendation is not treated as independently verified.

## Five modes do not mean five predefined EEG bands

There are three different uses of frequency here:

1. The preprocessing filter admits 0.5–40 Hz, with gradual filter transition behavior. Gamma above 40 Hz is excluded. This pipeline cannot recover or evaluate the full gamma band.
2. VMD learns center frequencies and mode bandwidths from the data. It does not split the signal into delta/theta/alpha/beta/gamma bins. Several modes may occupy one conventional band, and a single mode may span a boundary.
3. Welch PSD evaluation uses fixed frequency bins and non-overlapping physiological bands: delta [0.5,4), theta [4,8), alpha [8,13), beta [13,30). The declared FFT length determines resolution, delta_f = fs/nfft. Power is integrated in linear PSD units before converting band-power ratios to dB. The 30–40 Hz range can be reported separately; it is not called complete gamma.

For fs=200 Hz and nfft=512, bin spacing is 0.390625 Hz. Membership in each physiological band is computed by the explicit inequalities above, not by rounding frequencies into labels. The final report must state Hann window, segment length, overlap, nfft and integration rule. The mode's center returned by vmdpy is cycles/sample; multiply by fs to get Hz. A spectral centroid recomputed from Welch PSD is a different diagnostic and must not be mislabeled as the optimizer's center.

## What knowledge is needed to choose K

Understand the sampling rate and Nyquist limit, artifact and neural spectral overlap, the admitted analysis band, window duration/frequency resolution, VMD's bandwidth penalty and initialization, mode mixing and duplication, reconstruction residual, and the actual downstream objective.

K=5 is a starting hyperparameter, not a physiological fact. Counting familiar EEG bands is insufficient. Ocular activity overlaps genuine delta/theta and transient neural signals. A large K can split one rhythm into redundant modes; a small K can mix artifact and neural signals. Both situations can harm a downstream denoiser. Close center frequencies are a warning, not proof that K is too large: bandwidth overlap, mode similarity, energy, convergence, and held-out reconstruction/preservation also matter. Two distinct transient sources can share a center frequency.

We will run K=3..10 and alpha in {250,500,1000,2000,4000} on training/development data. We record sorted optimizer centers in Hz, energy fractions, bandwidth estimates, nearest-center separation, spectral overlap, residual ratio, iteration count, wall time and peak-memory diagnostics where measurable. The development objective compares paired correction error subject to alpha/beta, clean-window and covariance constraints; no OSF test result selects a zero-shot hyperparameter. Configuration scoring and complete candidate tables are saved, not just the winner.

The smoke grid uses a predeclared subset of training/development records to establish feasibility; it is not a full-cohort optimized result. Inner grouped validation and the five-fold/three-seed outer campaign remain necessary. ICA choices (Picard/extended Infomax and ocular correlation thresholds), ASR cutoffs (10/20/30), and ridge strengths are searched on development data. Regional fusion weights remain fixed hypotheses because Klados electrode mapping is unverified. Session-only spatial calibration consumes contaminated inputs; no test clean target enters fitting.

### First executed K-sweep evidence

The Kaggle benchmark at Git `bbba266e6f4f` executed all 40 K/alpha settings and all 160 correction candidates. FCM used two training records/two channels; selection used records 0 and 10 on those channels. It selected **K=5, alpha=2000, strength=1**, based on development reconstruction and preservation rather than counting EEG bands. Its saved development row records RMSE improvement fraction 0.27584749, clean-relative modification 0.01971193, alpha/beta clean errors 0.029606/0.007557 dB, zero iteration-limit fraction, mean nearest-center spacing 3.3360705 Hz and mean adjacent PSD overlap 0.03523281. These are averages over the declared subset, not full-cohort uncertainty bounds.

The center-frequency figure displays each K=3..10 for each of the five alphas. The example mode-vector page uses the historical K=5/alpha=1000 for explanation, **not** the selected alpha=2000. Its five optimizer centers are 1.5844, 8.7754, 11.1855, 17.7516 and 25.9230 Hz: two lie in the conventional alpha band, none is a gamma mode. It needed 450 iterations and retained a measured decomposition residual ratio of 0.079547. The saved calculated iterative-array lower bound is 98,304,000 bytes; whole-process high-water RSS is 709,952 KiB, which includes other loaded arrays/libraries and is not isolated VMD memory. The measured decomposition time is 0.304590 seconds on that Kaggle CPU run. This explains why actual vector/residual evidence and implementation-specific cost matter.

Source evidence is saved in `vmd_grid_summary.csv`, `vmd_centers.json`, `vmd_example.npz`, `vmd_example_metadata.json` and matching PNG/SVG figures, under the retrieved benchmark run directory. The second benchmark preserves this grid and broadens conservative ICA strength after the original full-subtraction choices failed development preservation.

## Hybrid and region handling

The VMD temporal expert and full-montage ICA source expert both estimate artifacts relative to the same input. Candidate ICA sources can receive VMD refinement, which preserves portions of a mixed component instead of rejecting it wholesale. Fixed regional fusion subtracts a convex mixture of the two estimates; a learned router is a separate ablation. Verified frontal electrodes get a stronger VMD prior, verified posterior electrodes get a more conservative spatial prior, and central/unknown channels use a shared rule. Neither branch is assumed to be anatomically optimal. Posterior pass-through is not unconditional because posterior EEG can also contain ocular contamination.

Klados channel order is currently unverified; no paired frontal/posterior claim is valid until its extraction mapping is recovered. Named OSF electrodes support regional proxy comparisons. Position metadata moves with a channel under permutation; EOG and annotation channels never enter the EEG-only model. Padded or flat channels are excluded by a mask. Channelwise weights and masked pooling make the parameter count independent of the number of cap channels, but performance for arbitrarily many or unseen layouts still requires experiments.

## Computational complexity and transformers

Let C be EEG channels, T samples, K VMD modes and I optimizer iterations. The mathematical spectral update costs approximately O(C * I * K * T), with FFT setup/reconstruction O(C * K * T log T). The exact vmdpy implementation also stores iterative arrays and performs reductions; its real memory/runtime must be measured, not inferred solely from this idealized complexity. Grid search multiplies cost by candidate count and development samples. Caching is valid only with input/configuration/Git hashes; cache reuse must not hide train/test fitting leakage.

Full-montage ICA has covariance/whitening cost roughly O(C^2*T + C^3); iterative source fitting often includes O(I*C^2*T). A shared per-channel convolution/GRU encoder and pooled spatial summary scales approximately linearly in C for fixed temporal/features settings. Fully connected channel attention can instead scale quadratically in C.

A transformer is not required to produce VMD modes or remove EOG. Standard temporal self-attention has quadratic token interaction cost O(T^2*d), and must demonstrate useful quality/latency gains to justify it. A small TCN/GRU model is the initial scalable candidate. Bidirectional GRUs and zero-phase filters use future samples; they are offline quality references. A separately trained causal GRU and causal preprocessing are required for live latency claims. Switching a trained BiGRU to a one-direction GRU is not an accuracy-preserving conversion.

## Research additions and evidence limits

[ARMBR](https://doi.org/10.1088/1741-2552/ade566) supplies an inspectable lightweight blink-specific spatial baseline. It does not establish lateral-eye removal; test both artifact classes independently. [EEGOAR-Net](https://doi.org/10.1016/j.bspc.2025.108147) supplies a modern montage-flexible neural competitor; matched Klados-only training avoids undisclosed pretrained overlap with OSF. [ASR documentation](https://mne.tools/mne-denoise/stable/asr.html) defines calibration/transform separation and explicitly warns that attenuation alone cannot certify preservation. These are candidates, not established winners on this campaign.

AlphaXiv discovery on 2026-10-04 returned mostly broad preprocessing/decoding papers and an unrelated anti-jamming paper; those were not used as direct evidence for VMD K selection. Exact primary-paper lookup is required rather than pretending every retrieved title is relevant.

## Final concern checklist

- State whether the actual K=3..10 search ran, and cite its saved candidate table and frequency plots.
- Explain adaptive VMD modes versus fixed Welch/physiological bins and the 40 Hz restriction.
- Justify selected K/alpha from development evidence; state if K=5 remains merely the starting baseline.
- Show raw vector to K by T mode matrix, optimizer centers, residual and alignment.
- Compare frontal/posterior/shared corrections only with verified metadata.
- Report measured runtime/memory/parameters and whether a transformer was needed.
- Separate blink and horizontal-eye suppression, paired reconstruction and clean-signal damage.
- Label remaining failures, missing provenance, and smoke versus full-study conclusions honestly.
