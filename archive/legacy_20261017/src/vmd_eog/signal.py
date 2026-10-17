"""Trial-aware offline preprocessing and correction overlap-add."""
from fractions import Fraction
import numpy as np
from scipy import signal as scipy_signal

FS = 200


def preprocess(values, native_fs):
    values = np.asarray(values, dtype=np.float64)
    if native_fs <= 80 or values.shape[-1] < 64 or not np.isfinite(values).all():
        raise ValueError("Signal cannot support declared offline preprocessing")
    if native_fs != FS:
        ratio = Fraction(FS / float(native_fs)).limit_denominator(10000)
        values = scipy_signal.resample_poly(values, ratio.numerator, ratio.denominator, axis=-1)
    sos = scipy_signal.butter(4, [0.5, 40], fs=FS, btype="bandpass", output="sos")
    return scipy_signal.sosfiltfilt(sos, values, axis=-1).astype(np.float32)


def calibration_trials(trials, fs):
    lengths = [row.shape[-1] / fs for row in trials]
    required = min(120.0, max(10.0, 0.2 * sum(lengths)))
    if len(trials) == 1:
        stop = int(np.ceil(required * fs))
        if trials[0].shape[-1] - stop < 1024 * fs / FS:
            return [], [], "insufficient unscored calibration and scoring duration"
        return [trials[0][..., :stop]], [trials[0][..., stop:]], None
    elapsed, count = 0.0, 0
    for length in lengths:
        elapsed += length
        count += 1
        if elapsed >= required:
            break
    if count >= len(trials):
        return [], [], "insufficient whole trials after calibration"
    return trials[:count], trials[count:], None


def overlap_add_correction(eeg, estimator, window=1024, hop=512):
    """Return all original samples; combine corrections, not reconstructed signals."""
    eeg = np.asarray(eeg)
    if eeg.ndim != 2 or not 0 < hop <= window:
        raise ValueError("Expected channels by time and a valid hop")
    length = eeg.shape[-1]
    pad = window // 2
    padded = np.pad(eeg, ((0, 0), (pad, pad + window)), mode="reflect")
    artifact = np.zeros_like(padded, dtype=np.float64)
    denominator = np.zeros(padded.shape[-1], dtype=np.float64)
    weight = np.hanning(window + 2)[1:-1]
    for start in range(0, padded.shape[-1] - window + 1, hop):
        correction = np.asarray(estimator(padded[:, start:start + window]))
        if correction.shape != (eeg.shape[0], window) or not np.isfinite(correction).all():
            raise ValueError("Invalid correction window")
        artifact[:, start:start + window] += correction * weight
        denominator[start:start + window] += weight
    valid = slice(pad, pad + length)
    if np.any(denominator[valid] <= 0):
        raise RuntimeError("Uncovered original sample")
    correction = artifact[:, valid] / denominator[valid]
    return eeg - correction, correction
