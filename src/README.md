# Lateral Eye Movement Detection & Artifact Removal

A complete pipeline for detecting and removing lateral eye movement artifacts from EEG data using ICA and machine learning.

## 📁 Project Structure

```
Lateral Eye Dataset/
├── dataset/                                    # Raw EEGLAB .set files
│   ├── without_eog_channels.set
│   ├── without_eog_channels.fdt
│   ├── without_eog_channels_Filter_ExtRunica_ICA.set  # Pre-computed ICA weights
│   └── without_eog_channels_Filter_ExtRunica_ICA.fdt
│
├── src/
│   ├── preprocessing/
│   │   ├── preprocess.py                      # EEG preprocessing pipeline
│   │   ├── NewMethodTry.ipynb                 # Original notebook (reference)
│   │   └── runica.m                           # MATLAB ICA implementation
│   │
│   ├── model/
│   │   ├── lateral_eye_classifier.py          # Classifier implementation
│   │   └── lateral_eye_classifier_trained.pkl # Saved trained model
│   │
│   ├── train_pipeline.py                      # Full training pipeline
│   ├── detect_lateral_movements.py            # Inference & visualization
│   └── README.md                              # This file
│
└── work/                                       # Output directory (optional)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install numpy scipy scikit-learn matplotlib mne hdf5storage
```

### 2. Train the Classifier

```bash
cd src
python train_pipeline.py
```

This will:
- Load and preprocess EEG data (FIR filter 0.5-40 Hz)
- Load pre-computed ICA weights
- Extract windowed features with delta (rate-of-change) emphasis
- Train Random Forest classifier
- Evaluate on test set
- Save trained model

### 3. Run Inference

```bash
python detect_lateral_movements.py
```

This will:
- Load trained classifier
- Detect lateral eye movements in the data
- Visualize detections overlaid on IC activations
- Generate detection statistics

## 📊 Pipeline Details

### Preprocessing (`preprocess.py`)

**Class**: `EEGPreprocessor`

**Steps**:
1. **Load metadata and events**: Parse EEGLAB .set file for channel info and event markers
2. **Load EEG data**: Use MNE-Python to load raw EEG signals
3. **FIR filter**: Apply 0.5-40 Hz bandpass filter (zero-phase)
4. **Load ICA weights**: Extract pre-computed icaweights, icasphere, icawinv matrices
5. **Compute IC activations**: `icaact = (icaweights @ icasphere) @ data_µV`
6. **Validate ICA**: Check reconstruction error, independence, matrix inversion

**Usage**:
```python
from preprocessing.preprocess import EEGPreprocessor

preprocessor = EEGPreprocessor(
    data_dir="../dataset",
    set_file="without_eog_channels.set",
    ica_file="without_eog_channels_Filter_ExtRunica_ICA.set"
)

results = preprocessor.run_full_pipeline()
ic_all = results['ic_all']  # (epochs, ICs, samples)
```

### Classification (`lateral_eye_classifier.py`)

**Class**: `LateralEyeClassifier`

**Key Concept**: Detects lateral eye movements (eye-l, eye-r) with **focus on high-delta regions** (rapid changes).

**Features Extracted** (per window, per IC):

**Delta Features** (rate of change):
- `delta_mean`, `delta_std`, `delta_max`: First-order difference statistics
- `delta_sign_changes`: Jitter vs smooth signal
- `accel_mean`, `accel_max`: Second-order difference (acceleration)
- `increasing_ratio`, `decreasing_ratio`: Monotonicity (consistent rise/fall)
- `n_peaks_pos`, `n_peaks_neg`: Sharp transition count

**Statistical Features**:
- `mean`, `std`, `min`, `max`, `range`: Basic statistics
- `skewness`, `kurtosis`: Distribution shape
- `zero_crossing_rate`: Frequency proxy
- `energy`: Signal power

**Usage**:
```python
from model.lateral_eye_classifier import LateralEyeClassifier

classifier = LateralEyeClassifier(
    window_size=100,      # 0.5s at 200 Hz
    overlap=75,           # 75% overlap
    artifact_ics=[2, 4],  # IC3=2, IC5=4 (0-indexed)
    model_type='rf'       # Random Forest
)

# Create dataset
X, y, metadata = classifier.create_windowed_dataset(
    ic_all=ic_all,
    events=events,
    srate=200,
    pnts=1600
)

# Train
classifier.train(X_train, y_train, X_val, y_val)

# Predict
predictions = classifier.predict(X_test)
```

## 🎯 Why Delta Features?

The classifier specifically detects **high-delta regions** (increasing/decreasing signals) because:

1. **Lateral eye movements create sharp deflections** in ICs
2. **Delta (rate of change) captures transitions** better than amplitude alone
3. **Eye movements = rapid signal changes**, not just high amplitude

Example delta features in action:
```
During lateral eye movement:
  IC3 values:  [1.2, 1.5, 2.8, 5.2, 6.1, 5.8, 3.2, 1.5]
  Delta:       [0.3, 1.3, 2.4, 0.9, -0.3, -2.6, -1.7]
                    ↑    ↑                  ↑    ↑
                  HIGH DELTA = DETECTED!

During rest:
  IC3 values:  [0.2, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2, 0.1]
  Delta:       [0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1]
                         LOW DELTA = NOT DETECTED
```

## 📈 Expected Performance

**Typical Results**:
- **Accuracy**: 85-95% (depending on data quality)
- **Precision (Lateral)**: 80-90% (few false positives)
- **Recall (Lateral)**: 85-95% (catches most lateral movements)

**Top Important Features**:
1. `IC3_delta_max` - Maximum rate of change in IC3
2. `IC3_delta_std` - Variability of change in IC3
3. `IC5_delta_max` - Maximum rate of change in IC5
4. `IC3_increasing_ratio` - Proportion of increasing samples
5. `IC3_accel_max` - Maximum acceleration

**Note**: If delta features dominate top 10, the classifier successfully learned to focus on rate-of-change!

## 🧹 Next Step: Artifact Removal

After detection, use the predictions to remove artifacts:

### Strategy: Adaptive Regression

```python
# Pseudo-code for artifact removal
for timepoint in range(n_samples):
    if lateral_detected[timepoint] == 1:
        # Regress artifact ICs from each EEG channel
        for channel in range(n_channels):
            # Find correlation between channel and IC3
            beta = correlation(eeg_channel, IC3)
            
            # Subtract artifact contribution
            clean_channel = eeg_channel - beta * IC3

# Reconstruct clean EEG
clean_eeg = icawinv @ modified_ic_activations
```

See: **Jung et al. (2000)** - "Removing electroencephalographic artifacts by blind source separation"

## 🔬 Technical References

### ICA-Based Artifact Removal
1. **Gratton et al. (1983)**: Original EOG regression method
   - *Electroencephalography and Clinical Neurophysiology*, 55(4), 468-484
   
2. **Jung et al. (2000)**: ICA for artifact removal
   - *Psychophysiology*, 37(2), 163-178
   
3. **Delorme & Makeig (2004)**: EEGLAB
   - *Journal of Neuroscience Methods*, 134(1), 9-21

### Documentation
- **EEGLAB Tutorial**: https://eeglab.org/tutorials/06_RejectArtifacts/RunICA.html
- **MNE-Python ICA Guide**: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html

## 🛠️ Customization

### Adjust Window Size
```python
classifier = LateralEyeClassifier(
    window_size=200,  # 1.0s at 200 Hz (larger windows, more context)
    overlap=150       # 75% overlap
)
```

### Change Artifact ICs
```python
# If different ICs contain eye artifacts
classifier = LateralEyeClassifier(
    artifact_ics=[0, 1, 3]  # Try different IC combinations
)
```

### Use Gradient Boosting
```python
classifier = LateralEyeClassifier(
    model_type='gb'  # Gradient Boosting (slower, potentially better)
)
```

## 📊 Validation Checklist

After training, verify:

- [ ] Test accuracy > 80%
- [ ] Precision (lateral) > 75%
- [ ] Recall (lateral) > 80%
- [ ] Delta features in top 10 important features
- [ ] Detection rate for `eye-l` and `eye-r` > 80%
- [ ] Detection rate for `eye-u`, `eye-d`, `blink` < 30%

## 🐛 Troubleshooting

**Low accuracy (<70%)**:
- Check if IC3/IC5 actually contain eye artifacts (plot them)
- Try different `window_size` (50-200 samples)
- Increase `overlap` for smoother detection

**High false positives**:
- Increase window size (more context)
- Use Gradient Boosting (`model_type='gb'`)
- Adjust class weights for imbalanced data

**Missing detections**:
- Decrease `window_size` (catch shorter movements)
- Add more artifact ICs
- Check event labeling (are events correctly marked?)

## 📝 Authors & License

**Author**: BCI Preprocessing & Classification Pipeline  
**Date**: February 2026  
**License**: MIT (or your preferred license)

## 🙏 Acknowledgments

- **EEGLAB**: Delorme & Makeig (2004)
- **MNE-Python**: Gramfort et al. (2013)
- **ICA Theory**: Hyvärinen & Oja (2000)