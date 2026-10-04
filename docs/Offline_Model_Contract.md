# Reproducible offline inference

Training exports `offline_model/model.pt` and `model_contract.json`, with the exact source Git SHA, model SHA256, architecture, preprocessing and feasibility limits. A separate fresh CPU interpreter predicts the actual held-out smoke windows and compares them with the original training-process outputs. `fresh_load_verification.json` records the measured difference; it must pass before claiming the saved checkpoint reproduces inference.

The input NPZ contains `eeg[B,C,T]` and scalar `fs=200`. Optional binary `mask[B,C]` excludes invalid/padded electrodes. Optional integer `regions[B,C]` uses frontal=0, posterior=1, central=2, unknown=3. Unknown is the default; unverified Klados channels stay unknown. Electrode ordering and region metadata must move together. Valid samples must be finite.

Prepare independent continuous trials using the source module's anti-alias resampling to 200 Hz and fourth-order 0.5–40-Hz zero-phase bandpass. Preserve the source reference. The CLI intentionally requires an explicit preprocessing declaration; it does not silently assume a raw file meets that contract. Under the campaign's execution restriction this command runs **on Kaggle**, not on the local computer:

```bash
python -m eog_vmd_fcm_bgru.inference --bundle /kaggle/input/TRAIN_OUTPUT/offline_model --input /kaggle/working/preprocessed_eeg.npz --output /kaggle/working/cleaned.npz --preprocessed --device cpu
```

The output retains aligned cleaned EEG, artifact estimate, gate/router diagnostics, masks and region IDs. Gate/router values are uncalibrated model controls. A variable-channel interface and fixed parameter count do not prove accuracy on every headcap. The BiGRU and preprocessing use future samples; this bundle is offline and has no streaming state or latency guarantee.
