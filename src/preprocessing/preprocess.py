"""
preprocess.py — Lateral Eye Movement EEG Preprocessing
======================================================
Mirrors exactly what NewMethodTry.ipynb does:
  1. Load EEGLAB .set metadata + events  (hdf5storage)
  2. Load EEG signals                    (MNE)
  3. FIR bandpass filter 0.5-40 Hz       (MNE firwin, zero-phase)
  4. Load pre-computed EEGLAB ICA weights from ICA .set file
  5. Compute icaact = (icaweights @ icasphere) @ data_µV
  6. Build windowed dataset (X, y) ready for CNN-LSTM training

Dataset layout expected:
    dataset/
        without_eog_channels.set / .fdt
        without_eog_channels_Filter_ExtRunica_ICA.set / .fdt
"""

from pathlib import Path
import warnings

import numpy as np
import hdf5storage
import mne

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Paths  (edit to switch participant or study)
# ─────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent                          # src/preprocessing/
_DATASET_ROOT = _SCRIPT_DIR.parent.parent / "dataset"                  # Lateral Eye Dataset/dataset/

# EEG source — generated .set/.fdt files (58 EEG channels, EOG removed)
EEG_DIR  = _DATASET_ROOT / "complete_dataset" / "complete_eeg"
SET_FILE = "study01_p01_eeg_only.set"   # ← change this to load a different participant

# ICA weights — pre-computed on study01_p01 (only available for this participant)
ICA_DIR  = _DATASET_ROOT
ICA_FILE = "without_eog_channels_Filter_ExtRunica_ICA.set"

# Legacy alias kept so callers that pass dataset_dir= still work
DATASET_DIR = EEG_DIR

# 6-class label map  (order defines class indices 0-5)
CLASSES: dict[str, int] = {
    "eye-l":    0,
    "eye-r":    1,
    "eye-u":    2,
    "eye-d":    3,
    "blink":    4,
    "fixation": 5,
}
N_CLASSES  = len(CLASSES)
CLASS_NAMES = list(CLASSES.keys())


# ─────────────────────────────────────────────────────────────
# Step 1 + 2 + 3 — Load & filter
# ─────────────────────────────────────────────────────────────

def _norm_mat_root(mat: dict):
    """
    Normalise the hdf5storage-loaded mat dict.

    - Original EEGLAB .set files (v5):  fields are at top level → return mat.
    - Our generated .set files (v5, scipy.io.savemat):  fields are inside
      mat['EEG'] which is a numpy structured-array record → return that record.
    """
    if "EEG" in mat and hasattr(mat["EEG"], "dtype"):
        return mat["EEG"].flat[0]   # structured-array record
    return mat                       # flat dict (original EEGLAB)


def load_eeg(dataset_dir: Path = EEG_DIR,
             set_file: str = SET_FILE) -> tuple[dict, mne.Epochs]:
    """
    Load metadata/events via hdf5storage and EEG signal from companion .fdt.

    Works with both:
      - Original EEGLAB .set files (top-level struct keys)
      - Our generated .set/.fdt pairs from batch_remove_eog.py

    Returns
    -------
    meta : dict
        nbchan, pnts, trials, srate, ch_names, events
    epochs_raw : mne.EpochsArray
        Raw (unfiltered) epochs object
    """
    set_path = dataset_dir / set_file
    fdt_path = set_path.with_suffix(".fdt")
    print(f"[load_eeg] reading {set_path}")

    mat  = hdf5storage.loadmat(str(set_path))
    eeg  = _norm_mat_root(mat)       # handles both struct layouts

    def _scalar(v) -> int:
        return int(np.asarray(v).flat[0])

    meta = {
        "nbchan": _scalar(eeg["nbchan"]),
        "pnts":   _scalar(eeg["pnts"]),
        "trials": _scalar(eeg["trials"]),
        "srate":  _scalar(eeg["srate"]),
    }

    # chanlocs — iterate over all records (shape may be (1,n) or (n,))
    meta["ch_names"] = [
        str(ch["labels"][0]) if hasattr(ch["labels"], "__getitem__") else str(ch["labels"])
        for ch in np.asarray(eeg["chanlocs"]).flat
    ]

    # events
    events = []
    for ev in np.asarray(eeg["event"]).flat:
        ev_type = ev["type"]
        events.append({
            "latency":  float(np.asarray(ev["latency"]).flat[0]),
            "type":     str(ev_type[0]) if hasattr(ev_type, "__getitem__") else str(ev_type),
            "epoch":    int(np.asarray(ev["epoch"]).flat[0]),
            "duration": float(np.asarray(ev["duration"]).flat[0]),
        })
    meta["events"] = events

    types = sorted({e["type"] for e in events})
    print(f"  channels={meta['nbchan']}  pnts/epoch={meta['pnts']}"
          f"  epochs={meta['trials']}  srate={meta['srate']} Hz")
    print(f"  events={len(events)}  types={types}")

    # Load raw signal from companion .fdt (float32, MATLAB column-major)
    # .fdt stores data in µV → divide by 1e6 for MNE (expects Volts)
    n_ch, n_pnts, n_ep = meta["nbchan"], meta["pnts"], meta["trials"]
    print(f"  loading {fdt_path.name}  ({fdt_path.stat().st_size / 1e6:.1f} MB)")
    flat = np.fromfile(str(fdt_path), dtype=np.float32)
    data = (flat
            .reshape((n_ch, n_pnts, n_ep), order="F")   # MATLAB Fortran-order
            .transpose(2, 0, 1)                          # → (n_ep, n_ch, n_pnts)
            .astype(np.float64) / 1e6)                   # µV → V

    info       = mne.create_info(ch_names=meta["ch_names"],
                                 sfreq=meta["srate"], ch_types="eeg")
    epochs_raw = mne.EpochsArray(data, info, tmin=0.0, verbose=False)
    print(f"  MNE shape: {epochs_raw.get_data().shape}")
    return meta, epochs_raw


def fir_filter(epochs_raw: mne.Epochs,
               l_freq: float = 0.5,
               h_freq: float = 40.0) -> mne.Epochs:
    """
    Zero-phase FIR bandpass filter (matches EEGLAB pop_eegfiltnew).

    Returns
    -------
    epochs_filt : mne.Epochs  (filtered copy)
    """
    print(f"[fir_filter] {l_freq}–{h_freq} Hz  firwin zero-phase")
    epochs_filt = epochs_raw.copy().filter(
        l_freq=l_freq, h_freq=h_freq,
        method="fir", fir_design="firwin", phase="zero",
        verbose=False,
    )
    d = epochs_filt.get_data()
    print(f"  shape={d.shape}  range=[{d.min():.4f}, {d.max():.4f}] V")
    return epochs_filt


# ─────────────────────────────────────────────────────────────
# Step 4 + 5 — ICA weights → IC activations
# ─────────────────────────────────────────────────────────────

def load_ica_and_compute_activations(
        epochs_filt: mne.Epochs,
        meta:        dict,
        dataset_dir: Path = ICA_DIR,
        ica_file:    str  = ICA_FILE,
) -> dict:
    """
    Load pre-computed EEGLAB ICA matrices and compute:
        icaact  = (icaweights @ icasphere) @ data_µV  →  (n_ics, total_samples)
        ic_all  = reshape icaact                      →  (trials, n_ics, pnts)

    Returns
    -------
    ica : dict
        icaweights, icasphere, icawinv, W_combined,
        icaact  (n_ics, total_samples),
        ic_all  (trials, n_ics, pnts),
        data_uv (n_ch, total_samples)
    """
    ica_path = dataset_dir / ica_file
    print(f"[load_ica] reading {ica_path}")

    ica_mat    = hdf5storage.loadmat(str(ica_path))
    icaweights = np.array(ica_mat["icaweights"])   # (58, 58)
    icasphere  = np.array(ica_mat["icasphere"])    # (58, 58)
    icawinv    = np.array(ica_mat["icawinv"])      # (58, 58)

    print(f"  icaweights={icaweights.shape}  icasphere={icasphere.shape}"
          f"  icawinv={icawinv.shape}")

    # Raw filtered epochs → µV, concatenated along time
    filt_data = epochs_filt.get_data()              # (trials, ch, pnts) [V]
    trials, nbchan, pnts = filt_data.shape
    data_uv = np.concatenate(
        [filt_data[i] for i in range(trials)], axis=1
    ) * 1e6                                         # (ch, total_samples) [µV]

    W_combined = icaweights @ icasphere             # (n_ics, ch)
    icaact     = W_combined @ data_uv               # (n_ics, total_samples)

    # Reshape into epochs
    ic_all = (icaact
              .reshape(meta["nbchan"], meta["trials"], meta["pnts"])
              .transpose(1, 0, 2))                  # (trials, n_ics, pnts)

    print(f"  data_uv={data_uv.shape}  icaact={icaact.shape}  ic_all={ic_all.shape}")

    print("  IC stats (first 6):")
    for i in range(min(6, icaact.shape[0])):
        v = icaact[i]
        print(f"    IC{i+1:>2d}  mean={np.mean(v):+9.2f}  std={np.std(v):8.2f}"
              f"  min={np.min(v):10.2f}  max={np.max(v):10.2f}")

    return dict(icaweights=icaweights, icasphere=icasphere,
                icawinv=icawinv, W_combined=W_combined,
                icaact=icaact, ic_all=ic_all, data_uv=data_uv)


# ─────────────────────────────────────────────────────────────
# Step 6 — Build windowed dataset  (X, y)
# ─────────────────────────────────────────────────────────────

# Task-type markers (the first event in each epoch, used for stratification)
_TASK_MAP: dict[str, int] = {"1": 0, "2": 1, "3": 2, "4": 3}
_TASK_NAMES = ["fixation", "lateral", "vertical", "blink"]


def _build_sample_labels(meta: dict) -> np.ndarray:
    """
    Return per-sample class labels: shape (trials, pnts), dtype int64.

    Each epoch contains multiple events with latencies.  For example a
    'lateral' epoch has alternating eye-l / eye-r events.  This function
    assigns each *sample* the class of the most recent recognised event,
    so the model learns fine-grained distinctions within an epoch.

    EEGLAB latency convention for epoched data:
        absolute_sample = (epoch_1idx - 1) * pnts + within_epoch_sample + 1
    """
    from collections import defaultdict

    trials = meta["trials"]
    pnts   = meta["pnts"]
    events = meta["events"]
    default_cls = CLASSES["fixation"]

    labels = np.full((trials, pnts), default_cls, dtype=np.int64)

    # Group events by epoch
    epoch_events: dict[int, list] = defaultdict(list)
    for ev in events:
        epoch_events[ev["epoch"]].append(ev)

    for ep_1idx in range(1, trials + 1):
        ep_0idx     = ep_1idx - 1
        epoch_start = ep_0idx * pnts + 1        # 1-indexed absolute sample

        # Sort recognised events by latency (ascending)
        recog = sorted(
            [ev for ev in epoch_events[ep_1idx] if ev["type"] in CLASSES],
            key=lambda e: e["latency"],
        )
        for ev in recog:
            cls = CLASSES[ev["type"]]
            s   = int(ev["latency"]) - epoch_start      # 0-indexed within epoch
            s   = max(0, min(s, pnts - 1))
            labels[ep_0idx, s:] = cls                    # fill forward

    # Summary
    total = labels.size
    print("  per-sample class distribution:")
    for name, idx in CLASSES.items():
        cnt = int(np.sum(labels == idx))
        print(f"    class {idx} ({name:>8s}): {cnt:6d} samples ({100*cnt/total:5.1f}%)")
    return labels


def _get_task_types(meta: dict) -> np.ndarray:
    """
    Return one task-type index per epoch  →  shape (trials,) int64.

    Uses the first event in each epoch ('1'/'2'/'3'/'4') that is a task
    marker.  This is used for *stratification only* (split epochs so that
    each fold sees every task type).
    """
    trials = meta["trials"]
    events = meta["events"]

    epoch_task: dict[int, int] = {}
    for ev in events:
        ep = ev["epoch"]
        if ev["type"] in _TASK_MAP and ep not in epoch_task:
            epoch_task[ep] = _TASK_MAP[ev["type"]]

    types = np.array(
        [epoch_task.get(i + 1, 0) for i in range(trials)], dtype=np.int64,
    )

    print("  task-type distribution (for stratification):")
    for i, name in enumerate(_TASK_NAMES):
        cnt = int(np.sum(types == i))
        print(f"    task {i} ({name:>9s}): {cnt:3d} epochs")
    return types


def make_windows(ic_all:         np.ndarray,
                 sample_labels:  np.ndarray,
                 srate:          int,
                 window_sec:     float = 1.0,
                 stride_sec:     float = 0.25,
                 n_ics:          int   = 58,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over every epoch and build (X, y).

    Parameters
    ----------
    ic_all         : (trials, n_ics, pnts)
    sample_labels  : (trials, pnts)  int64 class index per sample
    window_sec     : window length in seconds (default 1.0 s)
    stride_sec     : stride in seconds (default 0.25 s)
    n_ics          : how many ICs to use (first n_ics)

    Returns
    -------
    X : (n_windows, window_samples, n_ics)   float32
    y : (n_windows,)                          int64  (majority-vote per window)
    """
    win    = int(window_sec * srate)
    stride = int(stride_sec * srate)
    trials, total_ics, pnts = ic_all.shape
    n_ics  = min(n_ics, total_ics)

    X_list: list = []
    y_list: list = []

    for ep in range(trials):
        ic_ep  = ic_all[ep, :n_ics, :]        # (n_ics, pnts)
        lbl_ep = sample_labels[ep]             # (pnts,)

        start = 0
        while start + win <= pnts:
            end   = start + win
            x_win = ic_ep[:, start:end].T.astype(np.float32)  # (win, n_ics)

            # Majority-vote label for this window
            win_lbls = lbl_ep[start:end]
            cls = int(np.bincount(win_lbls, minlength=N_CLASSES).argmax())

            X_list.append(x_win)
            y_list.append(cls)
            start += stride

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)

    print(f"[make_windows] {len(X)} windows  shape={X.shape}")
    for name, idx in CLASSES.items():
        cnt = int(np.sum(y == idx))
        print(f"  class {idx} ({name:>8s}): {cnt:4d} windows ({100*cnt/len(y):5.1f}%)")
    return X, y


# ─────────────────────────────────────────────────────────────
# Normalise  (fit on train, apply to val/test)
# ─────────────────────────────────────────────────────────────

class ICANormalizer:
    """Per-IC z-score normaliser."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_:  np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ICANormalizer":
        """X : (N, T, C)"""
        self.mean_ = X.mean(axis=(0, 1), keepdims=True)   # (1,1,C)
        self.std_  = X.std (axis=(0, 1), keepdims=True).clip(min=1e-8)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ─────────────────────────────────────────────────────────────
# Stratified epoch-level split  (ensures every class in every fold)
# ─────────────────────────────────────────────────────────────

def stratified_epoch_split(
        ic_all:        np.ndarray,
        sample_labels: np.ndarray,
        task_types:    np.ndarray,
        srate:         int,
        window_sec:    float = 1.0,
        stride_sec:    float = 0.25,
        n_ics:         int   = 58,
        train_ratio:   float = 0.70,
        val_ratio:     float = 0.15,
        seed:          int   = 42,
) -> tuple:
    """
    Split *epochs* with stratification by task type, then window each subset.

    Stratification uses task_types (fixation/lateral/vertical/blink) so
    that each fold has a representative mix of all task types.  Per-window
    labels come from sample_labels via majority-vote.

    Returns X_train, y_train, X_val, y_val, X_test, y_test.
    """
    from sklearn.model_selection import train_test_split

    n_epochs = len(task_types)
    indices  = np.arange(n_epochs)

    test_ratio = 1.0 - train_ratio - val_ratio
    val_of_rest = val_ratio / (val_ratio + test_ratio)   # val share of non-train

    # First split: train vs (val+test)  — stratify by task type
    try:
        idx_tr, idx_rest = train_test_split(
            indices, test_size=(1.0 - train_ratio),
            stratify=task_types, random_state=seed)
    except ValueError:
        print("  [warn] stratify failed — falling back to shuffled split")
        idx_tr, idx_rest = train_test_split(
            indices, test_size=(1.0 - train_ratio), random_state=seed)

    # Second split: val vs test
    rest_tasks = task_types[idx_rest]
    try:
        idx_val, idx_te = train_test_split(
            idx_rest, test_size=(1.0 - val_of_rest),
            stratify=rest_tasks, random_state=seed)
    except ValueError:
        idx_val, idx_te = train_test_split(
            idx_rest, test_size=(1.0 - val_of_rest), random_state=seed)

    print(f"  epoch indices  train={sorted(idx_tr.tolist())}")
    print(f"                 val  ={sorted(idx_val.tolist())}")
    print(f"                 test ={sorted(idx_te.tolist())}")

    # Window each subset independently
    splits = []
    for name, idx in [("train", idx_tr), ("val", idx_val), ("test", idx_te)]:
        X, y = make_windows(ic_all[idx], sample_labels[idx],
                            srate=srate,
                            window_sec=window_sec,
                            stride_sec=stride_sec,
                            n_ics=n_ics)
        dist = "  ".join(f"{CLASS_NAMES[i]}={int(np.sum(y==i))}"
                         for i in range(N_CLASSES))
        print(f"  {name:5s}: {len(X):5d} windows  [{dist}]")
        splits.extend([X, y])

    return tuple(splits)   # X_tr, y_tr, X_val, y_val, X_te, y_te


# ─────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────

def run_preprocessing(dataset_dir: Path  = EEG_DIR,
                      ica_dir:     Path  = ICA_DIR,
                      set_file:    str   = SET_FILE,
                      window_sec:  float = 1.0,
                      stride_sec:  float = 0.25,
                      n_ics:       int   = 58,
) -> dict:
    """
    Full pipeline — returns everything the training script needs.

    Returns dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test  (numpy arrays)
        normalizer     : ICANormalizer  (fitted on train set only)
        meta           : EEG metadata
        ica            : ICA matrices + activations
        sample_labels  : (trials, pnts) int64 per-sample class labels
        task_types     : (trials,) int64 task type per epoch
        n_classes      : int  (always N_CLASSES = 6)
        class_names    : list of str
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE  —  6-class multiclass (per-sample labels)")
    print("=" * 60)

    # 1–3
    meta, epochs_raw = load_eeg(dataset_dir, set_file)
    epochs_filt      = fir_filter(epochs_raw)

    # 4–5
    ica = load_ica_and_compute_activations(epochs_filt, meta, ica_dir)

    # 6a — per-sample labels (using event latencies)
    print("\n[labels] building per-sample labels from event latencies:")
    sample_labels = _build_sample_labels(meta)

    # 6a.1 — task types for stratification
    print("\n[tasks]")
    task_types = _get_task_types(meta)

    # 6b + 7 — stratified epoch split, then window each subset
    print(f"\n[split+window] win={window_sec}s ({int(window_sec*meta['srate'])} pts)"
          f"  stride={stride_sec}s ({int(stride_sec*meta['srate'])} pts)"
          f"  n_ics={n_ics}")
    print("  stratified 70/15/15 at epoch level (by task type)")
    X_tr, y_tr, X_val, y_val, X_te, y_te = stratified_epoch_split(
        ica["ic_all"], sample_labels, task_types,
        srate=meta["srate"],
        window_sec=window_sec,
        stride_sec=stride_sec,
        n_ics=n_ics,
    )

    # 8 — normalise (fit on train only)
    norm  = ICANormalizer()
    X_tr  = norm.fit_transform(X_tr)
    X_val = norm.transform(X_val)
    X_te  = norm.transform(X_te)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  X_train {X_tr.shape}  y_train {y_tr.shape}")
    print(f"  X_val   {X_val.shape}  y_val   {y_val.shape}")
    print(f"  X_test  {X_te.shape}  y_test  {y_te.shape}")
    print(f"  n_classes={N_CLASSES}  classes={CLASS_NAMES}")
    print("=" * 60 + "\n")

    return dict(
        X_train=X_tr,  y_train=y_tr,
        X_val=X_val,   y_val=y_val,
        X_test=X_te,   y_test=y_te,
        normalizer=norm,
        meta=meta,
        ica=ica,
        sample_labels=sample_labels,
        task_types=task_types,
        n_classes=N_CLASSES,
        class_names=CLASS_NAMES,
    )


if __name__ == "__main__":
    data = run_preprocessing()
    print("X_train[0] shape:", data["X_train"][0].shape)
