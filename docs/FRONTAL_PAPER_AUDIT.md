# Frontal paper-method audit

Reviewed 11 October 2026. Numerical execution is Kaggle-only. New VMD calls
are restricted to verified frontal electrode names; historical results with
unknown channel anatomy are not regional evidence.

## Liu and Zhang, Sensors 2022, 22:6698

Paper: https://doi.org/10.3390/s22176698
Author repository: https://github.com/LCR997/V-sobi (redirects to Readyou123/V-sobi).
The accessible repository contains VMD.m, sobi.m and v_sobi.m.

The wrapper selects sources using fuzzy entropy plus absolute ordinary Pearson
correlation with the supplied nt1 reference, rejecting scores greater than .3.
Its FuzzyEntropy call specifies embedding 1, radius .2 times source SD and
exponent 2. The called FuzzyEntropy, SNR1 and multiPlot helpers and a full data
driver are absent. The reference's identity/availability must be recorded.
True simulated ocular artifact is oracle information, not an inference input.

The bundled VMD initializes centers at .01/.04/.3 for its first three modes
despite labeling that branch uniform; it caps at 500 iterations. Its SOBI
uses SVD whitening, four lagged covariance matrices by default, Frobenius
weights and complex rotations. These differ from our rolling solver and
symmetric covariance Jacobi implementation. Source-code parity is **pending**.

Implemented here: independent selector-structure adaptation, two observed EOG
axes rather than an unspecified nt1, fixed project VMD, SD-normalized fuzzy
entropy, existing SOBI, and an outer EOG association trigger. Fuzzy entropy
uses centered embedding templates, excludes self pairs and has a bounded-memory
scalar-pair oracle test. Do not call this an exact author-code reproduction.

## Dora and Biswal, BBE 2020, 40:148-161

Paper: https://doi.org/10.1016/j.bbe.2019.03.002
Publisher abstract/snippets establish mMSE contamination detection, VMD-based
ocular estimation, regression and blink/flutter/lateral scope. CAP and Sleep-EDF
are reported sources. Complete detector thresholds, mode thresholds, fixed
K/alpha and numerical evaluation details are not verified. Status: **deferred**.
Do not fill in the missing method with low-frequency rejection and label it
reproduced. EOG-assisted mode projection below is a separate project adaptation.

## Xiong et al., Sensors 2024, 24:1642

Paper: https://doi.org/10.3390/s24051642
Sections 2.1-2.4: ten-second SVM detection, time/PSD/nonlinear features, mRMR,
GA-VMD, SOBI and ApEn-based rejection. ApEn radius .15 SD and threshold .4
are documented. The printed optimum (K,alpha)=(120,4) conflicts with stated
K=2..10/alpha=100..5000 ranges. The GA objective aggregation, exact feature
definitions/kernel parameters and complete SOBI delays are insufficiently
specified for exact reproduction. GA is explicitly deferred in this campaign.

Implemented: fixed K3/alpha2000, 5.12-second project windows, eleven explicit
features from time/PSD families, training-only StandardScaler and RBF SVM
C=1/gamma=scale/class-balanced, no mRMR, existing SOBI delay bank, ApEn m=2,
r=.15 SD and threshold .4 plus observed EOG source association. The SVM fits
outside both heldout recipients and donors; only one correction recipe is
predeclared to avoid retuning it with contaminated nested crossfit predictions.
Status: **fixed-parameter adaptation**, not full SVM-IVMD-SOBI reproduction.

## Metric evidence and scope

Use docs/EVALUATION_PROTOCOL.md. Ordinary Pearson CC, MSE and temporal RRMSE
remain primary paired paper metrics. Energy-ratio output SNR and improvement
are separate from linear input RMS ratio and PSNR. Native EEG has no paired
reconstruction SNR. Missing full methods do not justify a superiority claim.

The old SVMD notebook repeats K=1 VMD; it remains a practical approximation,
not the published successive variational optimization. No SVMD implementation
is promoted in this campaign without verifying its actual optimizer.
