from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MCAResult:
    trend: np.ndarray
    transient: np.ndarray
    residual_history: list[float]


def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def split_trend_transient(
    signal: np.ndarray,
    max_iter: int = 50,
    lam_start: float = 1.0,
    lam_end: float = 0.05,
) -> MCAResult:
    signal = np.asarray(signal, dtype=float)
    trend = np.zeros_like(signal)
    transient = np.zeros_like(signal)
    residual_history: list[float] = []
    residual = signal.copy()
    lam_values = np.linspace(lam_start, lam_end, max_iter)

    for lam in lam_values:
        transient_step = soft_threshold(residual, lam)
        trend_step = residual - transient_step
        trend += trend_step / max_iter
        transient += transient_step / max_iter
        residual = signal - trend - transient
        residual_history.append(float(np.linalg.norm(residual)))

    return MCAResult(trend=trend, transient=transient, residual_history=residual_history)


def reconstruct_ic_from_trend(ic: np.ndarray, trend: np.ndarray) -> np.ndarray:
    if ic.shape != trend.shape:
        raise ValueError(f"shape mismatch: {ic.shape} vs {trend.shape}")
    return trend

