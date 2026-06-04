from __future__ import annotations

from pathlib import Path
from typing import Any

import hdf5storage
import mne
import numpy as np
import matplotlib.pyplot as plt

def fir_filter(epochs: mne.Epochs, l_freq: float = 0.5, h_freq: float = 40.0) -> mne.Epochs:
    return epochs.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose=False,
    )


def fit_mne_ica(
    epochs_filt: mne.Epochs,
    n_components: int | None = None,
    random_state: int = 42,
    method: str = "fastica",
    max_iter: int = 512,
):
    """Fit an ICA model with MNE and return the fitted estimator."""
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        method=method,
        max_iter=max_iter,
    )
    ica.fit(epochs_filt, verbose=False)
    return ica


def run_matlab_runica(data_uv: np.ndarray, n_components: int | None = None) -> dict[str, np.ndarray]:
    """
    Execute MATLAB's runica through the MATLAB engine when available.

    The function is intentionally lazy-imported so the notebooks can still
    open on machines without MATLAB. If the engine is unavailable, raise a
    clear error instead of silently faking the result.
    """
    try:
        import matlab.engine  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("MATLAB engine is not available in this environment") from exc

    eng = matlab.engine.start_matlab()
    try:
        if hasattr(eng, "runica"):
            icaweights, icasphere, icawinv = eng.runica(data_uv.tolist(), nargout=3)
        else:
            raise RuntimeError("MATLAB runica.m not found on MATLAB path")
    finally:
        try:
            eng.quit()
        except Exception:
            pass

    return {
        "icaweights": np.asarray(icaweights, dtype=float),
        "icasphere": np.asarray(icasphere, dtype=float),
        "icawinv": np.asarray(icawinv, dtype=float),
    }


def load_matlab_runica(ica_set_path: str | Path) -> dict[str, np.ndarray]:
    ica_mat = hdf5storage.loadmat(str(ica_set_path))
    icaweights = np.array(ica_mat["icaweights"], dtype=float)
    icasphere = np.array(ica_mat["icasphere"], dtype=float)
    icawinv = np.array(ica_mat["icawinv"], dtype=float)
    return {
        "icaweights": icaweights,
        "icasphere": icasphere,
        "icawinv": icawinv,
        "W_combined": icaweights @ icasphere,
    }


def build_ica_plot_legend():
    from matplotlib.patches import Patch

    return [
        Patch(facecolor="#DD0000", alpha=0.85, label="eye-l"),
        Patch(facecolor="#CC00CC", alpha=0.85, label="eye-r"),
        Patch(facecolor="#00AA00", alpha=0.85, label="eye-u"),
        Patch(facecolor="#0000DD", alpha=0.85, label="eye-d"),
        Patch(facecolor="#996600", alpha=0.85, label="blink"),
        Patch(facecolor="#888888", alpha=0.85, label="fixation"),
    ]


def plot_raw_epoch(epochs: mne.Epochs, epoch_1idx: int = 4, n_channels: int = 20, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(16, 7))
    ep = epoch_1idx - 1
    data = epochs.get_data()[ep, : min(n_channels, len(epochs.ch_names)), :]
    t = np.arange(data.shape[-1]) / epochs.info["sfreq"]
    for idx in range(data.shape[0]):
        trace = data[idx]
        scale = np.std(trace) or 1.0
        ax.plot(t, trace / scale + (data.shape[0] - idx) * 20.0, lw=0.6)
    ax.set_title(f"Raw EEG epoch {epoch_1idx}")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    return ax


def plot_event_markers(ax: plt.Axes, events: list[dict[str, Any]], epoch_1idx: int, srate: int, pnts: int) -> None:
    for ev in events:
        if ev["epoch"] != epoch_1idx:
            continue
        sample = int(ev["latency"]) - ((epoch_1idx - 1) * pnts + 1)
        if 0 <= sample < pnts:
            ax.axvline(sample / srate, color="gray", alpha=0.35, linestyle="--", linewidth=0.8)


def compute_ica_activations(epochs_filt: mne.Epochs, meta: dict[str, Any], ica_set_path: str | Path) -> dict[str, Any]:
    ica = load_matlab_runica(ica_set_path)
    filt_data = epochs_filt.get_data()
    trials, nbchan, pnts = filt_data.shape
    data_uv = np.concatenate([filt_data[i] for i in range(trials)], axis=1) * 1e6
    icaact = ica["W_combined"] @ data_uv
    ic_all = (
        icaact.reshape(meta["nbchan"], meta["trials"], meta["pnts"]).transpose(1, 0, 2)
    )
    ica.update({"icaact": icaact, "ic_all": ic_all, "data_uv": data_uv})
    return ica


def find_task_markers(meta: dict[str, Any]) -> np.ndarray:
    task_map = {"1": 0, "2": 1, "3": 2, "4": 3}
    epoch_task = {}
    for ev in meta["events"]:
        if ev["type"] in task_map and ev["epoch"] not in epoch_task:
            epoch_task[ev["epoch"]] = task_map[ev["type"]]
    return np.array([epoch_task.get(i + 1, 0) for i in range(meta["trials"])], dtype=np.int64)


def build_sample_labels(meta: dict[str, Any]) -> np.ndarray:
    classes = {"eye-l": 0, "eye-r": 1, "eye-u": 2, "eye-d": 3, "blink": 4, "fixation": 5}
    labels = np.full((meta["trials"], meta["pnts"]), classes["fixation"], dtype=np.int64)
    epoch_events: dict[int, list[dict[str, Any]]] = {}
    for ev in meta["events"]:
        epoch_events.setdefault(ev["epoch"], []).append(ev)

    for ep_1idx in range(1, meta["trials"] + 1):
        epoch_start = (ep_1idx - 1) * meta["pnts"] + 1
        recog = sorted([ev for ev in epoch_events.get(ep_1idx, []) if ev["type"] in classes], key=lambda e: e["latency"])
        for ev in recog:
            s = int(ev["latency"]) - epoch_start
            s = max(0, min(s, meta["pnts"] - 1))
            labels[ep_1idx - 1, s:] = classes[ev["type"]]
    return labels


def plot_ic_epoch(
    ic_all: np.ndarray,
    meta: dict[str, Any],
    epoch_1idx: int = 4,
    n_ics: int = 8,
    spacing: int = 50,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(18, 10))
    ep = epoch_1idx - 1
    epoch = ic_all[ep, : min(n_ics, ic_all.shape[1]), :]
    t = np.arange(meta["pnts"]) / meta["srate"]
    for i in range(epoch.shape[0]):
        s = np.std(epoch[i]) or 1.0
        ax.plot(t, epoch[i] / s + (epoch.shape[0] - i) * spacing, lw=0.6, color="black")
    ax.set_xlim(0, meta["pnts"] / meta["srate"])
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Epoch {epoch_1idx} IC stack")
    return ax
