# Full Technical Reference — Lateral Eye Movement BCI Pipeline

> **Location:** `Lateral Eye Dataset/`  
> **Python env:** `D:\Others\Miniconda\envs\BCI` (Python 3.12, PyTorch 2.6, MNE 1.x)  
> **Final model test accuracy:** **82.0%** | Macro-F1: **0.771**  
> **Date completed:** February 2026

---

## Table of Contents

1. [Dataset & Raw Files](#1-dataset--raw-files)
2. [Full Preprocessing Pipeline](#2-full-preprocessing-pipeline)
3. [The Critical Label Bug & Fix](#3-the-critical-label-bug--fix)
4. [CNN Architecture — How Conv1d Works on EEG](#4-cnn-architecture--how-conv1d-works-on-eeg)
5. [LSTM Architecture — Why We Need It After CNN](#5-lstm-architecture--why-we-need-it-after-cnn)
6. [CNN + LSTM Together — The Full Forward Pass](#6-cnn--lstm-together--the-full-forward-pass)
7. [Loss Function — Class-Weighted CrossEntropy](#7-loss-function--class-weighted-crossentropy)
8. [Training Loop — Every Decision Explained](#8-training-loop--every-decision-explained)
9. [Inference & Visualisation](#9-inference--visualisation)
10. [Final Results](#10-final-results)
11. [File Map](#11-file-map)
12. [All Commands Used](#12-all-commands-used)
13. [Bugs Fixed](#13-bugs-fixed)

---

## 1. Dataset & Raw Files

### What we have

| File | Type | Description |
|------|------|-------------|
| `dataset/without_eog_channels.set` | EEGLAB .set (HDF5) | 52 epoched EEG segments, 58 channels, EOG channels already removed |
| `dataset/without_eog_channels.fdt` | Binary float32 | Raw EEG data (companion to .set) |
| `dataset/without_eog_channels_Filter_ExtRunica_ICA.set` | EEGLAB .set | Pre-computed ICA weights (icaweights, icasphere, icawinv matrices) |
| `dataset/without_eog_channels_Filter_ExtRunica_ICA.fdt` | Binary float32 | ICA file companion data |

### Signal properties

```
Channels    : 58 (EEG only — EOG removed before ICA)
Epochs      : 52  (each epoch = one 8-second trial)
Samples/epoch: 1600
Sampling rate: 200 Hz  (1600 samples ÷ 200 Hz = 8 seconds per epoch)
Total samples: 52 × 1600 = 83,200 samples per channel
```

### Epoch task types

Every epoch belongs to exactly one of 4 task types, encoded by the FIRST event marker in that epoch:

| EEGLAB marker | Task | Epochs | What happens inside |
|---|---|---|---|
| `'1'` | fixation | 18 | Subject stares at centre dot — whole epoch is fixation |
| `'2'` | lateral | 11 | Eye goes LEFT → RIGHT → LEFT → … alternating |
| `'3'` | vertical | 12 | Eye goes UP → DOWN → UP → … alternating |
| `'4'` | blink | 11 | Deliberate blinks, separated by fixation |

### 6 output classes

```python
CLASSES = {
    "eye-l":    0,   # lateral left gaze
    "eye-r":    1,   # lateral right gaze
    "eye-u":    2,   # upward gaze
    "eye-d":    3,   # downward gaze
    "blink":    4,   # deliberate blink
    "fixation": 5,   # central fixation
}
```

---

## 2. Full Preprocessing Pipeline

### Why we need preprocessing

Raw EEG recorded at the scalp is a volume-conducted mixture of all brain sources. We need to separate them into Independent Components (ICs) so that eye-movement-specific sources are isolated and not contaminated by irrelevant brain activity.

### Step-by-step pipeline (`preprocess.py`)

```
Raw .set file
     │
     ▼
Step 1: Load metadata (hdf5storage)
        → meta dict: nbchan=58, pnts=1600, trials=52, srate=200, events list
        
     │
     ▼
Step 2: Load EEG signal (MNE)
        → mne.Epochs, shape (52, 58, 1600)  [trials, channels, samples]
        
     │
     ▼
Step 3: FIR Bandpass Filter  0.5 – 40.0 Hz
        → method="fir", fir_design="firwin", phase="zero"  (zero-phase, no delay)
        → removes DC drift (< 0.5 Hz) and high-freq noise / muscle artefact (> 40 Hz)
        → shape unchanged: (52, 58, 1600)
        
     │
     ▼
Step 4: Load ICA matrices (hdf5storage on ICA .set)
        → icaweights  : (58, 58)
        → icasphere   : (58, 58)
        → icawinv     : (58, 58)
        
     │
     ▼
Step 5: Compute IC activations
        data_uv = concatenate epochs → (58, 83200)  [µV]
        W_combined = icaweights @ icasphere          (58, 58)
        icaact     = W_combined @ data_uv            (58, 83200)
        ic_all     = reshape → (52, 58, 1600)        [trials, ICs, samples]
        
     │
     ▼
Step 6a: Per-sample labels  ← CRITICAL FIX (see Section 3)
         sample_labels: (52, 1600)  int64
         Each sample gets the class of the most recent EEGLAB event
         
     │
     ▼
Step 6b: Task-type array for stratification
         task_types: (52,)  int64
         Maps epoch → {0=fixation, 1=lateral, 2=vertical, 3=blink}
         
     │
     ▼
Step 7: Stratified train/val/test split  (70/15/15 at epoch level)
        Stratification is by task type — guarantees every fold sees
        fixation, lateral, vertical, and blink epochs.
        
     │
     ──────────────────────────────────────────
     │ for each of {train, val, test} subset:
     ▼
Step 8: Sliding window  (1.0 s window, 0.25 s stride)
        win    = 200 samples  (1.0s × 200 Hz)
        stride =  50 samples  (0.25s × 200 Hz)
        
        For each window of shape (200, 58):
          → X window: ic_epoch[:n_ics, start:end].T  →  (200, 58)  float32
          → y label : np.bincount(sample_labels[start:end]).argmax()
                      (majority vote — whichever class occupies most samples wins)
        
        Total windows:
          train: 1044   val: 203   test: 261
          Total: 1508 windows
     │
     ▼
Step 9: Z-score normalisation (ICANormalizer)
        mean_ = X_train.mean(axis=(0,1))  →  per-IC mean across time and windows
        std_  = X_train.std(axis=(0,1))   →  per-IC std
        
        ALL three sets normalised with train statistics only (no data leakage)
        X_train, X_val, X_test all → shape (N, 200, 58)
```

### Why Conv1d (not Conv2d)?

EEG data after ICA is **not** an image. It is a `(time, ICs)` matrix where:
- **Time axis** has temporal autocorrelation — adjacent samples belong to the same continuous sweep
- **IC axis** has no spatial adjacency — IC 3 and IC 4 are not "neighbours" in any meaningful sense

A Conv2d kernel would try to find 2D spatial patterns, which is meaningless here.  
A **Conv1d** kernel slides along the **time axis only** — exactly what we want: detecting the temporal waveform shape of an eye movement in each IC independently.

---

## 3. The Critical Label Bug & Fix

### What was wrong (the old `_build_epoch_labels`)

The original code assigned **one label per epoch** by scanning events in EEGLAB order and picking the first recognised class name it found:

```python
# OLD BROKEN CODE — DO NOT USE
def _build_epoch_labels(meta):
    labels = []
    for ep in range(1, trials+1):
        for ev in events:
            if ev["epoch"] == ep and ev["type"] in CLASSES:
                labels.append(CLASSES[ev["type"]])
                break   # <— stops at FIRST event
    return np.array(labels)
```

**The disaster this caused:**

A lateral epoch contains events like this (from `_debug_events.py` output):

```
Epoch 4 (task 2 = lateral):
  event[0]: type='2'      latency=4801   ← task marker
  event[1]: type='eye-l'  latency=4850   ← first recognised event
  event[2]: type='eye-r'  latency=5650   ← this was NEVER seen
  event[3]: type='eye-l'  latency=6450
  event[4]: type='eye-r'  latency=7250
  ...
```

- The loop always hit `eye-l` first → **epoch labeled eye-l**
- `eye-r` was **never** the first event in any epoch → **0 epochs labeled eye-r**
- `eye-u` similarly dominated by `eye-d` → only 2 epochs labeled eye-u

**Resulting class distribution (broken):**

| Class | Windows |
|-------|---------|
| eye-l | 261 |
| eye-r | **0** |
| eye-u | **29** |
| eye-d | 174 |
| blink | 204 |
| fixation | 322 |

The model was learning to **never predict eye-r** and rarely predict eye-u.  
When we saw val_acc = 86.2% — that was artificially inflated because predicting fixation almost always was safe (largest class).

### The fix: Per-sample labels with fill-forward (`_build_sample_labels`)

**Key insight:** EEGLAB stores not just the event type but its exact **latency** (absolute sample index). We can use this to label individual samples, not whole epochs.

```python
def _build_sample_labels(meta):
    labels = np.full((trials, pnts), CLASSES["fixation"], dtype=np.int64)
    
    for ep_1idx in range(1, trials + 1):
        ep_0idx     = ep_1idx - 1
        epoch_start = ep_0idx * pnts + 1    # absolute sample where this epoch begins
        
        # Sort all recognised events within this epoch by latency
        recog = sorted(
            [ev for ev in epoch_events[ep_1idx] if ev["type"] in CLASSES],
            key=lambda e: e["latency"]
        )
        
        for ev in recog:
            cls = CLASSES[ev["type"]]
            s   = int(ev["latency"]) - epoch_start   # 0-indexed within epoch
            s   = max(0, min(s, pnts - 1))
            labels[ep_0idx, s:] = cls                # FILL FORWARD from this sample
    
    return labels  # shape (52, 1600)
```

**How fill-forward works:**

```
Time → →  0    200   400   600   800  1000  1200  1400
                │           │          │
         eye-l event   eye-r event   eye-l event

Fill result:
  [fixation×200][eye-l×400][eye-r×400][eye-l×600]
```

Each sample gets the class of the most recent event, which perfectly captures the state of the eye at that moment.

**Resulting class distribution (fixed):**

| Class | Samples | % | Windows |
|-------|---------|---|---------|
| eye-l | 9,896 | 11.9% | 135 train |
| eye-r | **7,524** | **9.0%** | **97 train** |
| eye-u | **8,116** | **9.8%** | **103 train** |
| eye-d | 10,051 | 12.1% | 126 train |
| blink | 16,579 | 19.9% | 232 train |
| fixation | 31,034 | 37.3% | 351 train |

All 6 classes now properly represented.

### Majority-vote windowing

Because a 1-second window may span an eye-movement boundary, the window label is determined by majority vote:

```python
win_lbls = sample_labels[ep, start:end]            # 200 samples
cls = np.bincount(win_lbls, minlength=N_CLASSES).argmax()
```

Example: If a window covers 120 eye-l samples and 80 eye-r samples → label = eye-l.

---

## 4. CNN Architecture — How Conv1d Works on EEG

### What a Conv1d kernel actually does

A Conv1d layer with kernel size `k` and `out_ch` output channels applies `out_ch` different 1D filters to the input.

For input shape `(batch, channels, time)`:
- Each output feature map is produced by sliding one learned filter of shape `(in_ch, k)` along the time dimension
- The filter computes a dot product at each position: `output[t] = sum over c,k of (filter[c,k] × input[c, t+k])`
- This detects a specific temporal pattern (waveform shape) across all input channels simultaneously

**Applied to ICA activations:**
- Input: `(batch, 58, 200)` — 58 ICs, 200 time steps
- A filter of shape `(58, 5)` slides along time
- Each output sample = weighted sum of 58 ICs × 5 consecutive timepoints
- The model learns filters that fire when it sees the characteristic temporal shape of an eye movement

### The three CNN blocks

```
Input: (B, 58, 200)    ← batch × ICs × time_samples

Block 1:  Conv1d(58→32, kernel=5, padding=2)
          + BatchNorm1d(32)
          + GELU activation
          + Dropout(0.15)
          + MaxPool1d(2)
          ─────────────────────────────────────
          Output: (B, 32, 100)
          
          Learns: Which of 32 waveform patterns exist in the 58-IC signal?
          MaxPool halves the time resolution (coarser features)

Block 2:  Conv1d(32→64, kernel=3, padding=1)
          + BatchNorm1d(64)
          + GELU
          + Dropout(0.15)
          + MaxPool1d(2)
          ─────────────────────────────────────
          Output: (B, 64, 50)
          
          Learns: Higher-level combinations of the 32 primitive patterns

Block 3:  Conv1d(64→128, kernel=3, padding=1)
          + BatchNorm1d(128)
          + GELU
          + Dropout(0.15)
          + MaxPool1d(2)
          ─────────────────────────────────────
          Output: (B, 128, 25)
          
          Learns: 128 rich, abstract representations of 8-sample windows
```

After 3× MaxPool(2): 200 time steps → 25 time steps.  
Each of those 25 "tokens" summarises 8 original samples (40ms of EEG).  
The CNN has compressed the signal from `(200, 58)` → `(25, 128)`.

### Why GELU not ReLU?

GELU (Gaussian Error Linear Unit) is smoother than ReLU — it doesn't hard-zero negative activations but scales them by their Gaussian probability. In practice, it tends to produce better gradients for time-series models.

### Why BatchNorm?

BatchNorm normalises activations within each batch, preventing the network from becoming sensitive to the scale of EEG signals (which varies massively between ICs — IC1 std=12.34, IC6 std=2.11 in our data).

---

## 5. LSTM Architecture — Why We Need It After CNN

### What the CNN gives us and what's missing

After the CNN, we have `(batch, 25, 128)` — 25 time steps, each represented by 128 features.  
Problem: Conv1d has a **fixed receptive field**. Each output token only "sees" the original samples within its kernel window. Block 3 outputs see about 8 original samples (with pooling), Block 1+2+3 stacked sees maybe 24 samples.

But eye movements have **long-range temporal structure**:
- A lateral sweep might take 200ms → 40 CNN tokens
- The LSTM needs to track: "I saw high eye-r activity 100ms ago, now I'm in the mid-sweep return"

### How an LSTM works (the gate mechanism)

An LSTM cell maintains two state vectors at each time step `t`:
- `h_t`: **hidden state** — short-term working memory
- `c_t`: **cell state** — long-term memory (can persist across many steps)

At each step, 4 learned linear transformations (gates) control information flow:

```
Input gate   i_t = σ(W_i × [h_{t-1}, x_t] + b_i)    ← how much new info to add
Forget gate  f_t = σ(W_f × [h_{t-1}, x_t] + b_f)    ← how much old memory to keep
Cell gate    g_t = tanh(W_g × [h_{t-1}, x_t] + b_g)  ← candidate new info
Output gate  o_t = σ(W_o × [h_{t-1}, x_t] + b_o)    ← what to expose as output

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

The forget gate `f_t` can stay near 1.0 for many steps, allowing the LSTM to "remember" that it's in a lateral-scanning epoch even if individual timestep features are ambiguous.

### Our LSTM configuration

```
Input:        (B, 25, 128)   ← 25 CNN tokens, each 128-dimensional
LSTM hidden:  128
LSTM layers:  1              ← prevents overfitting on small dataset
Dropout:      0.0            ← not applicable for single-layer LSTM

Output:       (1, B, 128)    ← h_n (last hidden state, last layer, all batches)
              → squeeze → (B, 128)
```

We take **only the last hidden state** `h_n[-1]`. This is the LSTM's final summary of the entire 25-token sequence — a 128-dimensional "what did I just see in this 1-second window?"

### Weight initialisation

```python
LSTM weights → nn.init.orthogonal_   # prevents vanishing/exploding gradients
LSTM biases  → nn.init.zeros_
Conv weights → nn.init.kaiming_normal_ (fan_out, relu mode)
Linear weights → nn.init.xavier_uniform_
```

---

## 6. CNN + LSTM Together — The Full Forward Pass

```
Input: (B, T=200, C=58)   — batch × time_samples × ICs

    ┌─────────────────────────────────────────────────────────┐
    │   x.permute(0, 2, 1)  →  (B, 58, 200)                 │
    │                                                         │
    │   ── CNN ENCODER ──────────────────────────────────     │
    │   ConvBlock(58→32,k=5,pool=2)  →  (B, 32, 100)        │
    │   ConvBlock(32→64,k=3,pool=2)  →  (B, 64,  50)        │
    │   ConvBlock(64→128,k=3,pool=2) →  (B, 128,  25)       │
    │                                                         │
    │   x.permute(0, 2, 1)  →  (B, 25, 128)                 │
    │                                                         │
    │   ── LSTM ─────────────────────────────────────────     │
    │   lstm(x)  →  _, (h_n, c_n)                           │
    │   h_n[-1]  →  (B, 128)         last hidden state      │
    │                                                         │
    │   ── CLASSIFIER HEAD ──────────────────────────────     │
    │   Dropout(0.40)                                        │
    │   Linear(128 → 6)              raw logits             │
    └─────────────────────────────────────────────────────────┘

Output: (B, 6)  — CrossEntropyLoss expects raw logits, not softmax
```

**Why permute twice?**
- Conv1d wants `(B, channels, time)` — treat ICs as channels
- LSTM wants `(B, time, features)` — treat CNN output channels as features over time

**Parameter count:**

| Layer | Parameters |
|-------|-----------|
| ConvBlock 1 (58→32,k=5) + BN | 58×32×5 + 2×32 = 9,344 |
| ConvBlock 2 (32→64,k=3) + BN | 32×64×3 + 2×64 = 6,272 |
| ConvBlock 3 (64→128,k=3) + BN | 64×128×3 + 2×128 = 24,832 |
| LSTM (128→128, 1 layer) | 4×(128×128 + 128×128 + 128) = 132,608 |
| Linear (128→6) | 128×6 + 6 = 774 |
| **Total** | **~173,830** |

~174K parameters — deliberately compact to avoid overfitting on 1,044 training windows.

---

## 7. Loss Function — Class-Weighted CrossEntropy

### Why weighted loss?

The class distribution is imbalanced — fixation has 351 windows, eye-r has only 97.  
An unweighted model would just learn to predict fixation (most common) to minimise loss.

### How class weights are computed

```python
counts  = np.bincount(y_train)           # [135, 97, 103, 126, 232, 351]
weights = 1.0 / counts                   # inverse frequency
weights = weights / weights.sum() * 6   # scale so mean weight = 1.0
```

**Actual weights used:**

```
eye-l (n=135):    weight = 1.045   ← slightly upweighted
eye-r (n=97):     weight = 1.455   ← most upweighted (rarest)
eye-u (n=103):    weight = 1.370
eye-d (n=126):    weight = 1.120
blink (n=232):    weight = 0.608
fixation (n=351): weight = 0.402   ← most downweighted (most common)
```

CrossEntropyLoss multiplies the per-sample loss by its class weight before averaging. The model is thus penalised more for misclassifying a rare eye-r sample than a common fixation sample.

---

## 8. Training Loop — Every Decision Explained

### Optimizer: Adam with weight_decay

```python
torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

- **Adam**: Adaptive per-parameter learning rates + momentum (combines AdaGrad + RMSProp)
- **lr=1e-3**: Standard starting learning rate for Adam on medium-size problems
- **weight_decay=1e-4**: L2 regularisation — penalises large weights, fights overfitting

### Learning rate scheduler

```python
ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
```

Every time validation loss fails to improve for 5 consecutive epochs, the learning rate is halved. This allows fine-grained convergence once the large gradient updates are no longer useful.

### Gradient clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Limits the L2 norm of all gradients to 1.0 before each optimiser step. Prevents gradient explosions in the LSTM (LSTM is especially prone to this since gradients backpropagate through the recurrent loop).

### Early stopping

```python
patience = 15   # stop if val_loss doesn't improve for 15 consecutive epochs
```

Best checkpoint (lowest val_loss) is saved to `checkpoints/best_model.pt` and reloaded for testing. This means the final test is run on the model at epoch 7, not the model at epoch 22.

### Checkpoint contents

```python
{
    "epoch":       7,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict(),
    "val_loss":    0.4383,
    "val_acc":     0.857,
    "n_classes":   6,
    "class_names": ["eye-l", "eye-r", "eye-u", "eye-d", "blink", "fixation"],
    "norm_mean":   array of shape (1, 1, 58),   # ICANormalizer stats
    "norm_std":    array of shape (1, 1, 58),
    "args":        {epochs: 100, batch: 32, lr: 0.001, ...}
}
```

The normaliser parameters are saved in the checkpoint so that at inference time, new data is normalised with the **exact same statistics** the model was trained on. Using different statistics at inference produces garbage predictions.

### Training run (actual log)

```
Epoch  1/100  tr_loss=1.7565  tr_acc=0.218  val_loss=1.6538  val_acc=0.291
Epoch  2/100  tr_loss=1.3680  tr_acc=0.438  val_loss=1.4284  val_acc=0.300
Epoch  3/100  tr_loss=0.7850  tr_acc=0.673  val_loss=0.9330  val_acc=0.483
Epoch  4/100  tr_loss=0.5276  tr_acc=0.786  val_loss=0.6897  val_acc=0.660
Epoch  5/100  tr_loss=0.3380  tr_acc=0.899  val_loss=0.4566  val_acc=0.857  ✓
Epoch  6/100  tr_loss=0.2673  tr_acc=0.929  val_loss=0.5363  val_acc=0.793
Epoch  7/100  tr_loss=0.2101  tr_acc=0.951  val_loss=0.4383  val_acc=0.857  ✓  ← BEST
Epoch  8–22:  val_loss oscillates above 0.4383
Early stopping at epoch 22  (patience=15 exceeded)
```

---

## 9. Inference & Visualisation

### `infer_epoch.py` workflow

```
1. Load EEG + ICA  (same pipeline as training, no split)
2. Load checkpoint → restore model weights + normaliser stats
3. Extract one epoch's IC activations
4. Slide window over epoch → shape (N_windows, 200, 58)
5. Normalise with saved norm_mean / norm_std from checkpoint
6. Forward pass → logits → softmax → predicted class per window
7. Per-sample ground truth from sample_labels[epoch_0idx]  (fill-forward)
8. Per-window ground truth via majority vote (same as training)
9. Plot:
     ┌──────────────────────────────────────────┐
     │  Model predictions strip (per-window)    │
     │  MATLAB ground truth strip (per-sample)  │
     │  IC1 waveform  ─────────────────────────  │
     │  IC2 waveform  ─────────────────────────  │
     │  ...                                     │
     └──────────────────────────────────────────┘
```

### Epoch 4 inference result (lateral epoch)

```
 Win  t_centre          GT        pred      conf
   1    0.500s       eye-l       eye-l    99.3%  ✓
   2    0.750s       eye-l       eye-l    97.5%  ✓
   3    1.000s       eye-l       eye-l    52.6%  ✓
   4    1.250s       eye-r       eye-r    99.7%  ✓
   5    1.500s       eye-r       eye-r    99.4%  ✓
   6    1.750s       eye-r       eye-r    84.0%  ✓
   7    2.000s       eye-l       eye-l    96.8%  ✓
  ...
  15   4.000s       eye-r       eye-l    95.7%  ✗  ← boundary window
  16   4.250s       eye-r       eye-l    98.7%  ✗
  17   4.500s       eye-r       eye-l    99.1%  ✗
  18   4.750s       eye-l       eye-l    99.5%  ✓
  ...

Window-level accuracy: 89.7%  (26/29 correct)
```

The 3 misses are at eye-l→eye-r transition boundaries where the 1-second window straddles two classes — unavoidable given window size.

---

## 10. Final Results

### Test set performance (261 windows, held out from training)

```
Test accuracy: 82.0%
Macro F1:      0.771

              precision  recall  f1-score  support
     eye-l       0.76    0.84      0.80       31
     eye-r       0.81    0.78      0.79       27
     eye-u       0.59    0.61      0.60       28
     eye-d       0.49    0.60      0.54       30
     blink       1.00    1.00      1.00       58
  fixation       0.96    0.85      0.90       87
```

### Confusion matrix

```
          eye-l  eye-r  eye-u  eye-d  blink  fixat
  eye-l    26      5      0      0      0      0
  eye-r     6     21      0      0      0      0
  eye-u     1      0     17      9      0      1
  eye-d     0      0     10     18      0      2
  blink     0      0      0      0     58      0
  fixat     1      0      2     10      0     74
```

**Key observations:**
- **blink** and **fixation** are nearly perfect — their EEG signatures are very distinctive
- **eye-l / eye-r** confusion is ~19% — both are lateral movements, the model occasionally mislabels direction
- **eye-u / eye-d** confusion is ~33% — highest confusion pair; vertical ICA components less discriminative
- **No cross-category confusion** — blink is never mistaken for eye movement and vice versa

### Before vs after label fix

| Metric | Before fix (broken labels) | After fix (per-sample labels) |
|--------|--------------------------|-------------------------------|
| Test accuracy | 75.1% | **82.0%** |
| eye-r train windows | **0** | 97 |
| eye-u train windows | 29 | 103 |
| Macro F1 | ~0.65 (estimated) | **0.771** |
| Model actually learned eye-r? | No | **Yes** |

---

## 11. File Map

```
Lateral Eye Dataset/
├── dataset/
│   ├── without_eog_channels.set/.fdt           ← raw EEG (58ch, 52 epochs)
│   └── without_eog_channels_Filter_ExtRunica_ICA.set/.fdt  ← ICA weights
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py                          ← exports public API
│   │   └── preprocess.py                        ← full pipeline, label builder
│   │
│   ├── model/
│   │   ├── __init__.py                          ← exports LateralEyeCNNLSTM, build_loss
│   │   └── cnn_lstm.py                          ← architecture definition
│   │
│   ├── train.py                                 ← training loop, CLI, plots
│   ├── infer_epoch.py                           ← single-epoch inference + visualisation
│   │
│   ├── checkpoints/
│   │   ├── best_model.pt                        ← epoch 7, val_loss=0.4383
│   │   └── final_model.pt                       ← epoch 22
│   │
│   └── plots/
│       ├── training_curves.png                  ← loss + accuracy vs epoch
│       ├── confusion_matrix.png                 ← 6×6 test confusion heatmap
│       └── epoch_comparison.png                 ← last infer_epoch.py output
│
└── Full.md                                      ← this file
```

---

## 12. All Commands Used

> All run from `d:\Bunker\BaseCamp\BrainComputerInterface\Lateral Eye Dataset\src\`

### Training

```powershell
# First training run (old broken labels — do not repeat)
conda run -p D:\Others\Miniconda\envs\BCI --no-capture-output python train.py --epochs 100

# Retrain with corrected per-sample labels
conda run -p D:\Others\Miniconda\envs\BCI --no-capture-output python train.py --epochs 100
```

### Debugging the label bug

```powershell
# Verify per-sample labels and class distribution
conda run -p D:\Others\Miniconda\envs\BCI --no-capture-output python _verify_labels.py
# Output confirmed:
#   eye-r: 7524 samples (9.0%)  — was 0 before fix
#   eye-u: 8116 samples (9.8%)  — was ~29 before fix
```

### Inference

```powershell
# Run inference on epoch 4 (lateral, alternating eye-l/eye-r)
conda run -p D:\Others\Miniconda\envs\BCI --no-capture-output python infer_epoch.py --epoch 4

# Run inference on any epoch (1-indexed)
conda run -p D:\Others\Miniconda\envs\BCI --no-capture-output python infer_epoch.py --epoch 7

# Customise display
conda run -p D:\Others\Miniconda\envs\BCI --no-capture-output python infer_epoch.py --epoch 4 --n_ics 10 --spacing 8
```

### infer_epoch.py CLI reference

```
--epoch    INT     Which epoch to plot (1-indexed, default: 4)
--n_ics    INT     How many ICs to show in waveform panel (default: 8)
--spacing  FLOAT   Vertical spacing between IC traces in data units (default: 6)
--window   FLOAT   Window length in seconds (default: 1.0)
--stride   FLOAT   Stride in seconds (default: 0.25)
--out      PATH    Output plot path (default: plots/epoch_comparison.png)
```

### train.py CLI reference

```
--epochs   INT     Maximum training epochs (default: 60)
--batch    INT     Batch size (default: 32)
--lr       FLOAT   Initial learning rate (default: 1e-3)
--window   FLOAT   Window length in seconds (default: 1.0)
--stride   FLOAT   Stride in seconds (default: 0.25)
--n_ics    INT     Number of ICs to use (default: 58)
--patience INT     Early stopping patience (default: 15)
--seed     INT     Random seed (default: 42)
```

---

## 13. Bugs Fixed

### Bug 1 — Critical: Per-epoch label collapse (eye-r = 0 windows)

**File:** `preprocess.py`  
**Symptom:** Confusion matrix showed eye-r never predicted, 0 test windows for eye-r  
**Root cause:** `_build_epoch_labels` picked only the first recognised event per epoch; lateral epochs always start with eye-l → eye-r got 0 labeled epochs  
**Fix:** Replaced with `_build_sample_labels` using EEGLAB event latency timestamps and fill-forward strategy → 7,524 eye-r samples across 11 lateral epochs  

### Bug 2 — `torch.load` WeightsOnly error (PyTorch 2.6)

**File:** `train.py`, `infer_epoch.py`  
**Symptom:** `RuntimeError: Weights only load failed` when loading checkpoint containing numpy arrays  
**Root cause:** PyTorch 2.6 changed `weights_only` default from `False` to `True`; numpy arrays in checkpoint dict are not safe tensors  
**Fix:** Added `weights_only=False` to both `torch.load` calls  

### Bug 3 — `classification_report` ValueError on test set

**File:** `train.py`  
**Symptom:** `ValueError: Number of classes in y_true differs from target_names`  
**Root cause:** Test set (random 15% split) only contained 4 of 6 classes; `target_names=class_names` had 6 entries  
**Fix:** Added `labels=list(range(n_classes))` to `classification_report` call to force reporting all 6 classes regardless of test set presence  

### Bug 4 — Stratification by epoch label (wrong for multiclass windows)

**File:** `preprocess.py`  
**Symptom:** Val/test splits could miss entire task types (e.g., all vertical in test)  
**Root cause:** `stratified_epoch_split` was stratifying by per-epoch class label; after the label fix these were all `fixation` for mixed-movement epochs  
**Fix:** Replaced with stratification on `task_types` (fixation/lateral/vertical/blink) — guarantees all 4 task types in every split  

### Bug 5 — `infer_epoch.py` solid ground-truth strip

**File:** `infer_epoch.py`  
**Symptom:** Ground truth display showed one solid colour for entire epoch, hid the alternating eye-l/eye-r structure  
**Root cause:** Used `gt_class = epoch_labels[epoch_0idx]` (single label) instead of per-sample labels  
**Fix:** Replaced with `sample_labels[epoch_0idx]` and the same segment-drawing loop used for predictions  

### Bug 6 — Accuracy computed against single epoch label

**File:** `infer_epoch.py`  
**Symptom:** Accuracy was `mean(pred == gt_class)` — comparing each window against one epoch-level class  
**Fix:** Compute `gt_per_window` via majority vote from `sample_labels[epoch_0idx]` for each window, then `accuracy = mean(preds == gt_per_window)`  

---

*End of Full.md*
