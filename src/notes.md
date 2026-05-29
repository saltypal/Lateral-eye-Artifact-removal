The user is asking why we use Conv1d instead of Conv2d, and how Conv1d helps with EEG waveforms. This is a great question about the design choice.

Let me explain:

EEG data structure:

We have 58 ICs (channels)
200 time points per window
So the shape is (batch, 200, 1600) in full epoch, but windowed it's (batch, 200, 58)
If we wanted to use Conv2d, we'd need a 2D input. We could arrange it as:

(batch, 200, 58) - time × channels (makes sense semantically)
(batch, 200, 1) with 58 samples stacked somehow (doesn't make sense)
What Conv2d would do:

Conv2d has a 2D kernel (e.g., 3×3)
It would learn spatial patterns in both time AND channel dimensions simultaneously
Example kernel might detect: "correlations between IC3 and IC7 that persist for 3 timesteps"
What Conv1d does:

Conv1d has a 1D kernel (e.g., size 5)
It slides ONLY along the time axis
All 58 ICs are treated as separate feature channels (input channels)
A kernel of size 5 looks at "5 consecutive timesteps across all 58 ICs"
Why Conv1d is better for our case:

Time is special: Time naturally flows in one direction. We want to capture temporal patterns (sudden spikes, oscillations over 5ms windows)
Channels don't have spatial order: The ICs are not "spatially" arranged; they're just 58 independent components. ICA already decorrelated them.
Parameter efficiency: Conv1d is purpose-built for sequences, whereas Conv2d would squander capacity trying to learn spatial relationships between ICs that don't actually exist spatially
Clarity of learned features: When Conv1d kernels move through time, it's straightforward to interpret what temporal patterns they're capturing
The key distinction: Conv2d suits images where both height and width are spatial dimensions, but Conv1d fits sequences where time is the primary axis and channels are just parallel feature streams.

Great question! **Short answer**: EEG *time series* is 1D data. Conv1d slides along *time*, which is exactly what we want.

---

## Why NOT Conv2d?

**Conv2d** is for **2D spatial data** (images):
- Kernel is 2D: e.g., a 3×3 filter
- It looks for spatial patterns (e.g., corners, edges) in both height AND width

**Visual example** (image with Conv2d):
```
┌─────────────┐
│ . . . . .   │
│ . X X X .   │  ← Conv2d kernel (3×3)
│ . X X X .   │     looks at edges in both
│ . X X X .   │     vertical & horizontal
│ . . . . .   │
└─────────────┘
```

**Our EEG data structure**:
```
(batch, time=200, channels=58)

Time →
↓  I₁  I₂  I₃  ... I₅₈
t₁  0.2  -1.5  0.8  ...  2.1
t₂  0.5  -1.2  0.9  ...  2.3
t₃  0.8  -0.9  1.1  ...  2.5
...
t₂₀₀
```

**If we used Conv2d** (kernel 3×3):
```
Time axis ↓
Channel →

       I₁    I₂    I₃
    ┌──────────────┐
t₁  │ 0.2   -1.5   0.8
t₂  │ 0.5   -1.2   0.9
t₃  │ 0.8   -0.9   1.1
    └──────────────┘

Would learn patterns like:
"when IC1 has value ≈0.5 AND IC2 has value ≈-1.2 
 AND these occur 2 timesteps apart, it's eye-left"
```

**Problem**: ICs don't have a "spatial" meaning. IC1 and IC58 are not neighbors; they're just independent components. Conv2d would waste parameters learning "which IC pairs matter spatially" when ICA already decorrelated them.

---

## How Conv1d Helps (Why It's Better)

**Conv1d** is for **1D sequences**:
- Kernel is 1D: e.g., size 5
- Slides **only along time**
- All 58 channels are treated as the input feature dimension

**Visual example** (our setup with Conv1d):

```
Time →
┌────────────────────────────────────────┐
│ t: 0    1    2    3    4    5  ... 200 │
│    ─────────────────────────────────    │
│    ↑ Conv1d kernel (size 5)
│    Looks at 5 consecutive timesteps
│
│ Input: 58 ICs × 200 timesteps
│ Kernel: 58 input channels → 32 output channels, K=5
│
│ At each timestep, convolve:
│  [t₀:t₅] across all 58 ICs
│    ↓ produces 1 output value = "pattern detected?"
│  [t₁:t₆] across all 58 ICs
│    ↓ produces 1 output value
│  ...
│
│ Output: 32 features × 195 timesteps
└────────────────────────────────────────┘
```

**What Conv1d kernel learns for eye movement**:

```python
# Example: One kernel in Conv1d(58→32, k=5)
kernel = [
    [+0.3, -0.2, +0.1, ..., 0.05],  # IC1 weights (5 timesteps)
    [+0.1, +0.4, -0.3, ..., 0.02],  # IC2 weights (5 timesteps)
    [+0.2, +0.2, +0.1, ..., 0.08],  # IC3 weights (5 timesteps)
    ...
    [+0.01, +0.05, 0.0, ..., 0.12], # IC58 weights (5 timesteps)
]
# Shape: (58 input channels, 5 timesteps)

# At t=100, it computes:
output = sum of (kernel[i] * data[i, 100:105] for all 58 ICs)
# Result: "is this 5ms window (100:105) across all ICs similar to the pattern?"
```

**Interpretation**: "If IC3 spikes up over 5ms, AND IC7 slowly decreases, AND IC15 stays stable, this temporal pattern = eye-movement signature."

---

## Key Insight: Why Conv1d ≠ Conv2d for Sequences

| Aspect | Conv2d | Conv1d |
|--------|--------|---------|
| **What does it slide over?** | Height AND width simultaneously | Time axis only |
| **Kernel shape** | 2D (e.g., 3×3) | 1D (e.g., 5) |
| **Semantics** | Spatial patterns in 2D space | Temporal patterns in sequences |
| **For EEG?** | Treats IC-order as spatial (wrong!) | Treats time as temporal (right!) |
| **Efficiency** | More params (2D interactions) | Fewer params (1D temporal) |

---

## Concrete Example: Detecting Eye-Left Movement

**With Conv1d (what we use)**:
```
Kernel learns: "rapid positive deflection in IC3 over 3ms
               + sustained negative in IC7 over 3ms
               + small positive drift in IC12"
               
When this pattern appears at t=50-55ms,
output = high score → likely "eye-left"
```

**With Conv2d (hypothetical)**:
```
Kernel learns: "when IC3 and IC7 co-occur at t=50
              AND IC5 and IC12 co-occur at t=51
              this spatial-temporal cross-channel pattern = eye-left"
              
But wait... IC order is arbitrary (ICA can permute them).
This wastes parameters learning "which IC pairs spatially interact"
when in reality all ICs are independent (that's what ICA did).
```

---

## Bottom Line

- **Conv1d** respects the structure: **time flows, channels are features**
- **Conv2d** assumes: **both dimensions are spatial**
- For EEG: time is special (sequential, causal), channels are just features

Conv1d is **designed for exactly what we need**: sliding a temporal pattern detector along the time axis while seeing all 58 IC values at each timestep.









 You're right to question this. Let me investigate — something is off. Let me check the actual event data to see why eye-r gets 0 epochs and whether the stratification is actually working: 

Ran terminal command: cd "D:\Bunker\BaseCamp\BrainComputerInterface\Lateral Eye Dataset\src" ; conda run -p D:\Others\Miniconda\envs\BCI --no-capture-output python -c "
import sys; sys.path.insert(0, 'preprocessing')
from preprocess import load_eeg, _build_epoch_labels, CLASSES

meta, _ = load_eeg()
events = meta['events']

# 1. Count events by type
from collections import Counter
type_counts = Counter(e['type'] for e in events)
print('\n=== EVENT TYPE COUNTS ===')
for t, c in sorted(type_counts.items()):
    print(f'  {t:>10s}: {c}')

# 2. Show epoch->event mapping for ALL 52 epochs
print('\n=== EPOCH -> EVENTS ===')
from collections import defaultdict
epoch_events = defaultdict(list)
for e in events:
    epoch_events[e['epoch']].append(e['type'])

for ep in range(1, meta['trials']+1):
    evts = epoch_events[ep]
    recognized = [e for e in evts if e in CLASSES]
    print(f'  Epoch {ep:2d}: {evts}  -> recognized: {recognized}')

# 3. Show label assignment
print('\n=== LABEL ASSIGNMENT ===')
import numpy as np
labels = _build_epoch_labels(meta)
print(f'Labels: {labels.tolist()}')
"


Read [](file:///c%3A/Users/satya/AppData/Roaming/Code/User/workspaceStorage/e666156606ce0c232213f589a736a3da/GitHub.copilot-chat/chat-session-resources/eeb824f3-2057-4bd5-ae5d-cde42c2e2992/toolu_vrtx_01S315CfgJWgJx883ja6XUGp__vscode-1772043398076/content.txt#1-1), lines 1 to 120

Multiline script failed with conda. Let me write a quick debug script instead: 

Created [](file:///d%3A/Bunker/BaseCamp/BrainComputerInterface/Lateral%20Eye%20Dataset/src/_debug_events.py)

Ran terminal command: cd "D:\Bunker\BaseCamp\BrainComputerInterface\Lateral Eye Dataset\src" ; conda run -p D:\Others\Miniconda\envs\BCI --no-capture-output python _debug_events.py

Read [](file:///c%3A/Users/satya/AppData/Roaming/Code/User/workspaceStorage/e666156606ce0c232213f589a736a3da/GitHub.copilot-chat/chat-session-resources/eeb824f3-2057-4bd5-ae5d-cde42c2e2992/toolu_vrtx_017Vy4Ntw17NFgiC5mZ8P6Eb__vscode-1772043398079/content.txt)

**Found it.** Here's the problem:

The dataset has **no pure eye-r or pure eye-u epochs**. Look at the events:

```
Epoch  4: ['2', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r', ...] → eye-l
Epoch 36: ['3', 'eye-u', 'eye-d', 'eye-u', 'eye-d', 'eye-u', ...]          → eye-u
```

Each "lateral" epoch contains **both eye-l AND eye-r alternating** (the participant looks left then right repeatedly). Each "vertical" epoch has **both eye-u AND eye-d alternating**. Our labeler picks the **first recognized event** per epoch — so:
- Lateral epochs → always labeled **eye-l** (it comes before eye-r)
- Vertical epochs → usually labeled **eye-d** (it comes before eye-u most of the time)

**eye-r never wins** because eye-l always appears first in those epochs. And eye-u only wins in 2 epochs (36, 47) where it happens to appear before eye-d.

## The real label structure should be:

| Event group | What actually happens | Count |
|---|---|---|
| `'2'` epochs | Left-Right alternation | 11 |
| `'3'` epochs | Up-Down alternation | 12 |
| `'4'` epochs | Blinks | 11 |
| `'1'` epochs | Fixation | 18 |

So really there are **4 task types**, not 6 independent classes. We should either:

1. **Use the boundary events** (`'1'`, `'2'`, `'3'`, `'4'`) as the epoch labels — they're the actual task markers
2. **Label at the window level** — use each event's latency to assign eye-l, eye-r, eye-u, eye-d to the correct time window within each epoch

Option 2 is the right approach — each window gets the label of whichever event is active at that moment. Want me to fix `_build_epoch_labels` to do **per-window labeling** based on event latencies?

Made changes.       eye-u: 67
    fixation: 29

=== EPOCH -> EVENTS ===
  Epoch  1: events=['1', 'fixation']  -> first_recognized=fixation
  Epoch  2: events=['3', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-u', 'eye-d', 'eye-u']  -> first_recognized=eye-d
  Epoch  3: events=['3', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-u', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-u']  -> first_recognized=eye-d
  Epoch  4: events=['2', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l']  -> first_recognized=eye-l
  Epoch  5: events=['4', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink']  -> first_recognized=blink
  Epoch  6: events=['2', 'eye-l', 'eye-r', 'eye-r', 'eye-l', 'eye-r', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l']  -> first_recognized=eye-l
  Epoch  7: events=['3', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-d', 'eye-d', 'eye-u', 'eye-u', 'eye-d', 'eye-u']  -> first_recognized=eye-d
  Epoch  8: events=['1', 'fixation', 'fixation']  -> first_recognized=fixation  
  Epoch  9: events=['4', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink']  -> first_recognized=blink
  Epoch 10: events=['4', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink']  -> first_recognized=blink
  Epoch 11: events=['3', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-u', 'eye-d', 'eye-u', 'eye-d', 'eye-u']  -> first_recognized=eye-d
  Epoch 12: events=['4', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink']  -> first_recognized=blink
  Epoch 13: events=['2', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r']  -> first_recognized=eye-l
  Epoch 14: events=['1', 'fixation']  -> first_recognized=fixation
  Epoch 15: events=['4', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink']  -> first_recognized=blink
  Epoch 16: events=['1', 'fixation']  -> first_recognized=fixation
  Epoch 17: events=['2', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-r', 'eye-l', 'eye-r', 'eye-l']  -> first_recognized=eye-l
  Epoch 18: events=['1', 'fixation']  -> first_recognized=fixation
  Epoch 19: events=['3', 'eye-d', 'eye-u', 'eye-u', 'eye-d', 'eye-u', 'eye-u', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-d']  -> first_recognized=eye-d
  Epoch 20: events=['3', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-u', 'eye-d', 'eye-u', 'eye-u', 'eye-d', 'eye-u', 'eye-u']  -> first_recognized=eye-d
  Epoch 21: events=['1', 'fixation']  -> first_recognized=fixation
  Epoch 22: events=['1', 'fixation']  -> first_recognized=fixation
  Epoch 23: events=['2', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l']  -> first_recognized=eye-l
  Epoch 24: events=['1', 'fixation']  -> first_recognized=fixation
  Epoch 25: events=['1', 'fixation']  -> first_recognized=fixation
  Epoch 26: events=['2', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r']  -> first_recognized=eye-l
  Epoch 27: events=['4', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink']  -> first_recognized=blink
  Epoch 28: events=['3', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-d', 'eye-d', 'eye-u', 'eye-u', 'eye-d', 'eye-d', 'eye-u']  -> first_recognized=eye-d
  Epoch 29: events=['1', 'fixation']  -> first_recognized=fixation
  Epoch 30: events=['2', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l']  -> first_recognized=eye-l
  Epoch 31: events=['4', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink']  -> first_recognized=blink
  Epoch 32: events=['3', 'eye-d', 'eye-u', 'eye-d', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-d', 'eye-u']  -> first_recognized=eye-d
  Epoch 33: events=['3', 'eye-d', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-d', 'eye-u']  -> first_recognized=eye-d
  Epoch 34: events=['1', 'fixation']  -> first_recognized=fixation
  Epoch 35: events=['1', 'fixation', 'fixation']  -> first_recognized=fixation  
  Epoch 36: events=['3', 'eye-u', 'eye-d', 'eye-u', 'eye-u', 'eye-d', 'eye-u', 'eye-u', 'eye-d', 'eye-u', 'eye-u', 'eye-d', 'eye-d', 'eye-u', 'eye-u']  -> first_recognized=eye-u
  Epoch 37: events=['2', 'eye-l', 'eye-r', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r']  -> first_recognized=eye-l
  Epoch 38: events=['3', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-d', 'eye-d', 'eye-u', 'eye-d', 'eye-u', 'eye-d', 'eye-u']  -> first_recognized=eye-d
  Epoch 39: events=['1', 'fixation', 'fixation', 'fixation']  -> first_recognized=fixation
  Epoch 40: events=['4', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink']  -> first_recognized=blink
  Epoch 41: events=['1', 'fixation']  -> first_recognized=fixation
  Epoch 42: events=['2', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-r', 'eye-l', 'eye-r', 'eye-l', 'eye-r', 'eye-l']  -> first_recognized=eye-l
  Epoch 43: events=['4', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink']  -> first_recognized=blink
  Epoch 44: events=['1', 'fixation', 'fixation']  -> first_recognized=fixation  
  Epoch 50: events=['4', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink']  -> first_recognized=blink
  Epoch 51: events=['4', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink']  -> first_recognized=blink
  Epoch 52: events=['1', 'fixation', 'fixation', 'fixation']  -> first_recognized=fixation