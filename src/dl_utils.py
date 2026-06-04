from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def build_epoch_windows(ic_all: np.ndarray, meta: dict[str, Any], epoch_1idx: int, window_sec: float = 1.0, stride_sec: float = 0.25, n_ics: int = 58):
    srate = meta["srate"]
    win = int(window_sec * srate)
    stride = int(stride_sec * srate)
    ep = ic_all[epoch_1idx - 1, : min(n_ics, ic_all.shape[1]), :]
    windows = []
    ranges = []
    for start in range(0, meta["pnts"] - win + 1, stride):
        end = start + win
        windows.append(ep[:, start:end].T.astype(np.float32))
        ranges.append((start, end))
    return np.stack(windows), ranges


def locate_lateral_regions(predictions: np.ndarray, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(predictions) != len(ranges):
        raise ValueError("predictions/ranges length mismatch")
    out = []
    active = None
    for pred, (start, end) in zip(predictions, ranges):
        if int(pred) == 1 and active is None:
            active = [start, end]
        elif int(pred) == 1 and active is not None:
            active[1] = end
        elif int(pred) == 0 and active is not None:
            out.append((active[0], active[1]))
            active = None
    if active is not None:
        out.append((active[0], active[1]))
    return out

