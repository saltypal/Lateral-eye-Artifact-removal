# Six-method Klados experiment: exact decisions

## Data and comparison

All six methods use the same historical cache, not six differently generated
datasets. Original export correspondence and hashes were audited earlier. The
cache contains paired raw/clean EEG, HEOG/VEOG, fixed K3/alpha2000 decompositions,
validity flags and record/window identities. No LEMON signals enter this run.

Training/validation/test contain 37/8/8 disjoint records. Participant IDs cannot
be reconstructed; record separation is not participant separation. Historical
test results have already influenced development, so this is a development
comparison. No untouched confirmation or cross-dataset superiority is claimed.

Use the original Klados paper list order as instructed by the user on 17 October:
FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 Fz Cz Pz.
Frontal indices are 0,1,2,3,10,11,16. The publication lists names but does not
explicitly bind MATLAB row numbers; this assumption is saved in every run.
Source: https://doi.org/10.1016/j.dib.2016.06.032

## Method provenance

1. **Historical VMD plus EOG model.** Original 2499-parameter MLP checkpoint,
   hard support gate, K=3/alpha=2000, runtime HEOG and VEOG. Preserve its original
   metadata feature convention (unknown=3), restrict actual context and outputs
   to seven selected frontal channels. Its previously measured 17.927210 dB
   all-19-row replay is a separate result; it is not a promised frontal score.
2. **SVMD–FCM adaptation.** Ocular paper: *EOG-free EEG using successive
   variational mode decomposition and fuzzy c-means clustering*, DOI
   10.1016/j.bspc.2025.109434. Accessible text identifies synthetic, BCI Competition
   IV and SP300 BCI data, SVMD and moving-window FCM using K_i, MA, SD and HFD.
   Exact K_i definition, window, fuzzy settings and cluster labelling are not
   available. Our K_i=kurtosis interpretation is explicitly an assumption.
   PySDKit 0.5.0 SVMD follows Nazari/Sakhaei, DOI 10.1016/j.sigpro.2020.107610.
   Original MATLAB EEG demo specifies maxAlpha1000/tau0/tol1e-6/stopc1; the
   independently visible reference is recorded under the source audit.
   Our declared 200-sample/100-hop window, m2, two clusters, threshold0.8,
   <=8Hz eligibility and semantic cluster rule are project choices. No runtime
   reference is used by this adaptation. Max16 modes is a visible safety cap,
   not a tuned K or AutoVMD selector. Inner convergence is not exposed by the
   library and therefore not claimed verified.
3. **VME–GMETV adaptation.** Correct paper name is *EOG Artifacts Suppression
   From single channel EEG Signals by VME-GMETV model*, DOI
   10.1016/j.bspc.2023.105622. It extracts an ocular-dominated mode, estimates
   ocular activity inside it with GMETV and retains the remainder. Exact dataset
   name and full optimized settings are unverified from accessible text.
   VME uses alpha20000, omega0=0Hz, tau0, tol1e-7, max300. GMETV follows
   Selesnick/Lanza/Morigi/Sgallari2020 DOI10.1007/s10851-019-00937-5,
   Algorithm3 equations62a–c. Choose C=cI, B=cD, c²=gamma/(4lambda), gamma0.8,
   lambda0.5 after input RMS scaling. Since ||D||²<4, lambda||B||²<0.8<1:
   the total objective remains strongly convex. The inner minimisation becomes
   soft thresholding; the outer step is Condat TV of y+lambda*z. This is a
   valid explicit family member, not a claim to reproduce the ocular paper's
   unavailable optimized matrix. Exact finite iterations and convergence flags
   are saved; failures remain scored as pass-through.
4. **EEGOAR-Net.** Author architecture and weights from dmarcos97/EEGOAR-Net,
   exact SHA in author_sources.json. The original paper is DOI
   10.1016/j.bspc.2025.108147. Use author default 128Hz, 128samples, 64-channel
   mask. Only seven requested frontal channels are present, others zero/masked.
   Resample200→128 and back, with a separate resampling-only control. No Klados
   physical-unit conversion is guessed; pretrained cross-domain amplitude
   mismatch is an interpretation limitation. Do not silently tune pretrained
   inference on test targets.
5. **GEVD–MWF.** Existing regularized instantaneous implementation following
   Somers et al.2018, *A generic EEG artifact removal algorithm based on the
   multi-channel Wiener filter*, NeuroImage171:403–416,
   DOI10.1016/j.neuroimage.2017.12.034. It uses marked artifact and artifact-free
   covariance estimates to estimate artifact spatial structure; GEVD provides
   a low-rank approximation. Here training EOG activity top/bottom30% supplies
   approximate masks, Ledoit–Wolf stabilizes covariance, rank2 and strength1
   are fixed. This is not the complete lag-embedded author implementation and
   does not use test clean targets for fitting.
6. **BandRouteNet.** arXiv2604.24428v2; released LTPhat/BandRouteNet-BRN
   BRAANetV2 implementation, pinned SHA and file hashes. The paper uses
   EEGdenoiseNet (4515 EEG,3400 EOG,5598 EMG,256Hz512samples), MSE, AdamW,
   lr0.001, wd0.0001,15epochs,batch32. Our Klados adaptation keeps the author
   network, changes sampling200Hz, supplies explicit bands(.5–4,4–8,8–12,
   12–30,30–80,80–100), uses batch8 and splits each1024 window into two512
   blocks. The released code's 12–30 boundary avoids the paper text's12–13 gap.
   Input RMS scales both input and target. Validation MSE selects checkpoint;
   test records never choose epochs. Record actual parameter count; do not
   repeat the paper's headline count without measurement. No released licence
   was found in the inspected tree; author code is fetched for the experiment,
   attributed and pinned, rather than relicensed into our modules.

## Reporting and failures

Report per-channel/window and record-balanced MSE, RMSE, temporal RRMSE, Pearson
CC, output SNR and SNR improvement. Both pooled-window SNR and mean-channel SNR
are saved with distinct labels. Residual H/V correlation is diagnostic; true EEG
can share reference activity, so zero correlation is not a universal objective.
Klados lacks reliable condition annotations, so no fabricated blink/lateral
subgroups are produced. Native OSF provides those annotations in a later check.

Clean controls apply the same methods to paired clean EEG; EOG-assisted methods
receive the same measured references, exposing reference-to-neural leakage.
Keep failures as raw identity predictions, identify failed method windows and
retain partial outputs. A failed score is never silently excluded from means.

The Kaggle outer notebook executes the readable inner notebook through nbclient.
After each cell, it saves the actual executed notebook, including execution
counts, output tables, stream logs, errors and image payloads. Final artifact
hashing happens after notebook completion, avoiding a self-hash cycle. Both the
source notebook and executed notebook are downloaded and verified.
