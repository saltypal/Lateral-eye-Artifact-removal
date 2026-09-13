"""Signal filtering, VMD decomposition, and mode-level descriptors."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.stats import entropy, kurtosis
from vmdpy import VMD

from .config import VMDConfig


def bandpass(
    values: np.ndarray,
    sampling_hz: float = 200.0,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
) -> np.ndarray:
    """Apply the shared 0.5–40 Hz zero-phase EEG/EOG filter along time."""
    sos = signal.butter(
        4,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sampling_hz,
        output="sos",
    )
    return signal.sosfiltfilt(sos, values, axis=-1).astype(np.float32)


def vmd_one(values: np.ndarray, config: VMDConfig) -> np.ndarray:
    """Decompose one 1-D signal and preserve its original sample length.

    Some VMD implementations return one sample more or less than the input.
    This routine trims or edge-pads deterministically, preserving the notebook's
    5400/5401 alignment rule.
    """
    signal_1d = np.asarray(values, dtype=np.float64).reshape(-1)
    modes, _, _ = VMD(
        signal_1d,
        config.alpha,
        0,
        config.modes,
        0,
        1,
        config.tolerance,
    )

    aligned = np.zeros((config.modes, signal_1d.size), dtype=np.float32)
    copied = min(signal_1d.size, modes.shape[-1])
    aligned[:, :copied] = modes[:, :copied]
    if copied < signal_1d.size and copied > 0:
        aligned[:, copied:] = aligned[:, copied - 1 : copied]
    return aligned


def mode_descriptor(mode: np.ndarray, sampling_hz: float = 200.0) -> np.ndarray:
    """Compute the six FCM features for a single VMD mode.

    Feature order: spectral centroid, low-frequency fraction, energy,
    kurtosis, spectral entropy, and temporal concentration.
    """
    values = np.asarray(mode, dtype=np.float64).reshape(-1)
    frequencies, power = signal.welch(
        values,
        fs=sampling_hz,
        nperseg=min(512, values.size),
    )
    total_power = power.sum() + 1e-12
    spectral_centroid = float(np.sum(frequencies * power) / total_power)
    low_frequency_fraction = float(power[frequencies <= 5.0].sum() / total_power)
    probability = power / total_power
    temporal_concentration = float(
        np.max(np.abs(values)) / (np.sqrt(np.mean(values**2)) + 1e-8)
    )
    return np.asarray(
        [
            spectral_centroid,
            low_frequency_fraction,
            float(np.mean(values**2)),
            float(kurtosis(values, fisher=False)),
            float(entropy(probability + 1e-12)),
            temporal_concentration,
        ],
        dtype=np.float32,
    )
