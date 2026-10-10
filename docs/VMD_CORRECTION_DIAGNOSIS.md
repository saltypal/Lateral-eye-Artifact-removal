# VMD correction diagnosis

The completed forty-setting search is retained unchanged. Its preferred Fz
selection subset scored 5.2274 dB for projected VMD versus 10.4617 dB for direct
regression under the archived objective. This does not prove VMD decomposition
is invalid; it requires checking the correction layer before calling the method
successful. The rolling solver already has numerical parity tests against vmdpy.

Notebook 04a checks three distinct mechanisms on Kaggle:

- Reconstruct the original input from all saved modes plus the saved residual.
  This detects scaling, cropping or mode-vector errors independently of scoring.
- Project every mode and the residual against exactly the same references and
  ridge settings. Their sum must equal direct projection by linearity. This
  control cannot be described as an independent VMD improvement.
- Compare selected-mode projection using the scoring-window EOG mean versus
  an explicitly declared unscored calibration EOG baseline. Partition total
  MSE into squared mean bias and centered error variance to inspect offsets.

The calibration-baseline alternative estimates coefficients from centered EEG
and EOG, then applies them to EOG excursions relative to the calibration mean.
It consumes no clean target. Calibration lag means exclude samples crossing
trial boundaries. Its defaults do not replace any existing algorithm or running
experiment. Clean-input modification must be measured before it becomes a
candidate teacher. Corrected-mean SNR is not used as the output-SNR target.

Saved outputs include per-channel paper metrics, clean preservation, mode-gate
counts, reconstruction/projection closure, residual projection RMS and error
bias. True controlled artifact values appear only in diagnosis, never inference.
This first mechanism check retains the original Fz scope; an improvement there
must subsequently survive all-frontal, full-regional and grouped comparisons.

Verified first diagnostic (`vmd-diagnosis-001`, source fc79b30) passed 51 tests
and retained all 128 selection examples. Maximum relative reconstruction error
was 5.5863e-8 and mode-plus-residual projection closure error was 5.8510e-8.
Matched direct regression scored 8.6741 dB; the all-components algebraic control
scored identically. This matched comparator has lag zero and differs from the
original best direct-regression candidate (10.4617 dB with a wider lag bank).
VMD calibration baselines scored 5.2706 versus 5.2274 dB for the original mode
correction. Thus baseline handling alone does not explain the recovery deficit.
All six initial controls reported zero modification on the scored clean subset.

The next saved notebook version adds explicit all-mode, residual and capped-fit
controls, per-mode projection magnitudes and the declared association-threshold
ablations. It uses actual summed component projections for the closure control.
These are mechanism experiments, not a selected replacement teacher. Lower
thresholds must report their clean-input cost; no threshold is accepted just
because its dirty-input SNR improves.
