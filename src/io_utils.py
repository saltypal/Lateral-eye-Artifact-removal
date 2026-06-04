from __future__ import annotations

from pathlib import Path
from typing import Any

import hdf5storage
import mne
import numpy as np


def _as_int(value: Any) -> int:
    return int(np.asarray(value).flat[0])


def _decode_scalar(value: Any) -> str:
    if hasattr(value, "__getitem__"):
        try:
            return str(value[0])
        except Exception:
            pass
    return str(value)


def _norm_mat_root(mat: dict[str, Any]) -> Any:
    if "EEG" in mat and hasattr(mat["EEG"], "dtype"):
        return mat["EEG"].flat[0]
    return mat


def load_eeg_set(set_path: str | Path) -> tuple[dict[str, Any], mne.Epochs]:
    set_path = Path(set_path)
    fdt_path = set_path.with_suffix(".fdt")

    mat = hdf5storage.loadmat(str(set_path))
    eeg = _norm_mat_root(mat)

    meta: dict[str, Any] = {
        "nbchan": _as_int(eeg["nbchan"]),
        "pnts": _as_int(eeg["pnts"]),
        "trials": _as_int(eeg["trials"]),
        "srate": _as_int(eeg["srate"]),
    }

    ch_names = []
    for ch in np.asarray(eeg["chanlocs"]).flat:
        ch_names.append(_decode_scalar(ch["labels"]))
    meta["ch_names"] = ch_names

    events = []
    for ev in np.asarray(eeg["event"]).flat:
        ev_type = ev["type"]
        events.append(
            {
                "latency": float(np.asarray(ev["latency"]).flat[0]),
                "type": str(ev_type[0]) if hasattr(ev_type, "__getitem__") else str(ev_type),
                "epoch": int(np.asarray(ev["epoch"]).flat[0]),
                "duration": float(np.asarray(ev["duration"]).flat[0]),
            }
        )
    meta["events"] = events
    meta["set_path"] = str(set_path)
    meta["fdt_path"] = str(fdt_path)

    flat = np.fromfile(str(fdt_path), dtype=np.float32)
    data = (
        flat.reshape((meta["nbchan"], meta["pnts"], meta["trials"]), order="F")
        .transpose(2, 0, 1)
        .astype(np.float64)
        / 1e6
    )

    info = mne.create_info(ch_names=ch_names, sfreq=meta["srate"], ch_types="eeg")
    epochs = mne.EpochsArray(data, info, tmin=0.0, verbose=False)
    return meta, epochs


def load_eeg_only_epoch_file(
    set_path: str | Path,
) -> tuple[dict[str, Any], mne.Epochs]:
    return load_eeg_set(set_path)

