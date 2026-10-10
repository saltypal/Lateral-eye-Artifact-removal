# References that affect implementation

Evaluation equations, source locations, discrepancies and adaptations are
specified in [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). Project preservation
thresholds are not paper-standard metrics or published acceptance thresholds.

## VMD

Dragomiretskiy and Zosso, *Variational Mode Decomposition*, IEEE Transactions on Signal Processing (2014), [DOI](https://doi.org/10.1109/TSP.2013.2288675). The algorithm learns band-limited modes and center frequencies jointly. It does not justify K=5 by naming five physiological EEG bands.

[vmdpy reference](https://github.com/vrcarva/vmdpy) is the parity oracle for the independently copied rolling solver. The original implementation keeps iteration history; our solver keeps two spectral buffers and the reference return convention. RMS normalization makes stopping behavior independent of signal units. Uniform initialization, tau=0, no forced DC, tolerance 1e-6 and cap 2000 are declared. A separate residual preserves incomplete reconstruction.

Selection compares K=3..10, alpha 250/500/1000/2000/4000, lag banks, ridge penalties, association thresholds and strengths. Center proximity alone cannot prove over-segmentation: bandwidth and spectral overlap also matter. Retained EEG and grouped recovery decide whether extra modes help.

## Frontal VMD–SOBI candidate

[Xiong et al., SVM–IVMD–SOBI](https://mdpi-res.com/d_attachment/sensors/sensors-24-01642/article_deploy/sensors-24-01642-v2.pdf?version=1709543478),
sections 2.1–2.4, motivates a separate contamination detector and mode-domain
second-order source separation. `sobi.py` independently implements whitening,
symmetric delayed covariances and orthogonal Jacobi joint diagonalization;
`vmd_sobi.py` evaluates reconstructed source subtraction versus EOG-projected
source correction. Source means and the VMD residual stay in the input.
Declared delay banks and EOG-based detection are adaptations, not the paper's
unspecified delay bank or SVM/GA replication. Approximate entropy uses m=2,
r=0.15 SD with self matches and a declared 0.4 threshold for the optional
entropy-plus-EOG branch. No superiority is assumed before Kaggle evidence.
The bounded plan and promotion criteria are in `VMD_SOBI_PLAN.md`.

## Posterior MWF implementation

[Authors' MATLAB implementation](https://github.com/exporl/mwf-artifact-removal) provides calibration-mask covariance and generalized-eigenvalue artifact estimation. Our `posterior.py` adapts its GEVD operator, with Ledoit–Wolf covariance regularization and selectable positive excess rank. Analytic diagonal and nonorthogonal fixtures check the equation independently. This is an adaptation with changed covariance estimation, not a claim of full MATLAB pipeline replication.

Posterior-only estimation and signed frontal support are compared under the same inputs. Calibration lag columns exclude trial joins. Increasing rank or lag embedding changes preservation and cubic spatial-fitting cost; neither is assumed beneficial.

## ICA and qualified eye calibration

[Picard author implementation](https://github.com/pierreablin/picard) supplies extended nonorthogonal ICA. Component selection uses signed lagged HEOG/VEOG association on calibration data. PCA residuals are preserved by subtracting only selected-component artifacts from the original EEG. Failed convergence is a scored passthrough.

[Kobler eye-artifact correction code](https://github.com/rkobler/eyeartifactcorrection) motivates separate eye movement/blink calibration and native real-data evaluation. SGEYESUB is not called replicated until its specific calibration and reference implementation are qualified. An unvalidated approximation will not be reported as that method.

## Source evidence

[Raw LEMON publisher download page](http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON/downloads/download_EEG.html) supplies participant archive URLs. Only development raw EEG is opened before model freeze. The publisher OSF readme defines global IDs, EEG/EOG types and trial/sample labels. [Magdeburg DOI](https://doi.org/10.24352/UB.OVGU-2020-155) is reserved for independent waveform evaluation after freeze; metadata audit alone is not waveform validation.
