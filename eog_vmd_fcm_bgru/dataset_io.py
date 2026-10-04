"""Shared preprocessing and grouped splits. No trial-boundary concatenation."""
import hashlib
from fractions import Fraction
import numpy as np
from scipy import signal
from sklearn.model_selection import GroupShuffleSplit
from .provenance import read_osf


def preprocess(values, native_fs=200):
    if native_fs <= 80:
        raise ValueError("Native Nyquist cannot support the 40-Hz contract")
    ratio = Fraction(200 / native_fs).limit_denominator(10000)
    result = signal.resample_poly(values, ratio.numerator, ratio.denominator, axis=-1) if native_fs != 200 else np.asarray(values)
    sos = signal.butter(4, [0.5, 40], fs=200, btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, result, axis=-1).astype(np.float32)


def klados_arrays(root):
    path = root / "klados"
    dirty = np.load(path / "klados_contaminated_eeg.npy", allow_pickle=False)
    clean = np.load(path / "klados_pure_eeg.npy", allow_pickle=False)
    eog = [np.load(path / f"klados_{name}.npy", allow_pickle=False) for name in ("heog", "veog")]
    reshaped = [value.reshape(len(dirty), -1, dirty.shape[-1]) for value in eog]
    if any(value.shape[1] != 1 for value in reshaped):
        raise ValueError("Klados EOG has multiple reference rows; an explicit mapping is required")
    references = np.stack([value[:, 0] for value in reshaped], axis=1)
    return preprocess(dirty), preprocess(clean), preprocess(references)


def split_records(clean, seed=42):
    # Shared clean targets imply shared underlying EEG: keep these variants in
    # one group even if their contaminations differ. Subject IDs remain unknown.
    groups = [hashlib.sha256(np.ascontiguousarray(record).tobytes()).hexdigest() for record in clean]
    indices = np.arange(len(clean))
    train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(indices, groups=groups))
    train_local, val_local = next(GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed).split(train_val, groups=np.asarray(groups)[train_val]))
    return {"train": train_val[train_local].tolist(), "val": train_val[val_local].tolist(), "test": test.tolist(),
            "group_unit": "duplicate-clean-target group; subject provenance unavailable"}


def osf_trials(path):
    item = read_osf(path)
    for trial in range(len(item["data"])):
        eeg = np.asarray(item["data"][trial, item["eeg_indices"]], dtype=np.float64)
        valid = np.isfinite(eeg).all(axis=-1) & (np.ptp(eeg, axis=-1) > 0)
        if not valid.any():
            continue
        eeg = preprocess(eeg[valid], item["fs"])
        eog = {item["names"][index]: preprocess(item["data"][trial, index], item["fs"]) for index in item["eog_indices"]}
        labels = None
        if item["sample_labels"] is not None:
            # Annotation resampling is nearest-neighbor; never low-pass labels.
            positions = np.minimum((np.arange(eeg.shape[-1]) * item["fs"] / 200).astype(int), item["sample_labels"].shape[-1] - 1)
            labels = item["sample_labels"][trial, positions]
        yield {"eeg": eeg, "eog": eog, "labels": labels, "trial": trial,
               "names": [name for name, keep in zip([item["names"][i] for i in item["eeg_indices"]], valid) if keep],
               "study": item["study"], "participant": item["participant"], "session": path.stem,
               "native_fs": item["fs"], "reference": item["reference"]}
