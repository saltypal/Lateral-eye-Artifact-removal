"""Fixed VMD correction on explicitly selected frontal signals only."""
import time
import numpy as np
from .metrics import correlations
from .methods import window_regression
from .vmd import rolling_decompose


def window_starts(samples, window=1024, hop=512):
    if samples < window:
        raise ValueError("Trial shorter than declared VMD window")
    starts = list(range(0, samples-window+1, hop))
    if starts[-1] != samples-window:
        starts.append(samples-window)
    return starts


def overlap_add(windows, starts, samples):
    """Positive taper at both endpoints; one complete estimate for each sample."""
    window = windows.shape[-1]
    taper = np.hanning(window+2)[1:-1]
    result = np.zeros((*windows.shape[1:-1], samples), dtype=float)
    denominator = np.zeros(samples, dtype=float)
    for values, start in zip(windows, starts):
        result[..., start:start+window] += values*taper
        denominator[start:start+window] += taper
    if np.any(denominator <= 0):
        raise ValueError("Overlap-add left uncovered samples")
    return result/denominator


def estimate(values, eyes, config, names):
    """Return artifact, cached modes/residuals and every mode's H/V evidence.

    Values must already be a verified frontal subset. No electrode ordering is
    inferred here. A capped/invalid solver produces zero correction and an
    explicit diagnostic. Reference fitting uses the current window, so the
    matched direct-window regression has the same reference access.
    """
    if len(values) != len(names) or eyes.shape != (2, values.shape[-1]):
        raise ValueError("Frontal values, channel names and references do not align")
    starts = window_starts(values.shape[-1], config["window"], config["hop"])
    recipe = config["vmd"]
    artifacts, all_modes, all_residuals, diagnostics = [], [], [], []
    began = time.perf_counter()
    for start in starts:
        wave = values[:, start:start+config["window"]]
        references = eyes[:, start:start+config["window"]]
        correction = np.zeros_like(wave)
        modes = np.zeros((len(wave), recipe["K"], config["window"]), dtype=np.float32)
        residuals = wave.copy()
        for channel, row in enumerate(wave):
            failure = None
            try:
                vectors, residual, detail = rolling_decompose(row, recipe["K"], recipe["alpha"],
                    fs=config["sampling_rate"], tolerance=recipe["tolerance"], max_iterations=recipe["max_iterations"])
                modes[channel], residuals[channel] = vectors, residual
                valid = not detail["hit_iteration_limit"]
            except (ValueError, RuntimeError, FloatingPointError) as error:
                vectors = modes[channel]
                detail = {"centers_hz": [None]*recipe["K"], "iterations": 0,
                          "hit_iteration_limit": False, "runtime_s": None}
                failure = str(error)
                valid = False
            association = correlations(vectors, references)
            for mode, vector in enumerate(vectors):
                selected = valid and np.max(np.abs(association[mode])) >= config["mode_reference_threshold"]
                if selected:
                    correction[channel] += window_regression(vector[None], references, config["ridge"])[0]
                spectrum = np.abs(np.fft.rfft(vector))**2
                frequency = np.fft.rfftfreq(len(vector), 1/config["sampling_rate"])
                energy = spectrum.sum()
                measured_center = float(np.sum(frequency*spectrum)/energy) if energy > 0 else None
                bandwidth = float(np.sqrt(np.sum((frequency-measured_center)**2*spectrum)/energy)) if energy > 0 else None
                diagnostics.append({"channel": names[channel], "window_start": start, "mode": mode+1,
                    "center_hz": detail["centers_hz"][mode], "measured_center_hz": measured_center,
                    "bandwidth_hz": bandwidth, "heog_cc": float(association[mode, 0]),
                    "veog_cc": float(association[mode, 1]), "selected": bool(selected),
                    "valid": bool(valid), "iterations": detail["iterations"],
                    "hit_iteration_limit": detail["hit_iteration_limit"], "failure": failure,
                    "mode_energy": float(np.sum(vector.astype(float)**2)),
                    "residual_energy": float(np.sum(residuals[channel].astype(float)**2))})
        artifacts.append(correction)
        all_modes.append(modes)
        all_residuals.append(residuals)
    artifact = overlap_add(np.stack(artifacts), starts, values.shape[-1])
    cache = {"starts": np.asarray(starts), "modes": np.stack(all_modes),
             "residuals": np.stack(all_residuals), "artifact": artifact.astype(np.float32),
             "runtime_s": np.asarray(time.perf_counter()-began)}
    return artifact, cache, diagnostics


def combine(original, frontal_artifact, other_artifact, groups):
    """Disjoint masks, estimates relative to original EEG, exactly one subtraction."""
    if frontal_artifact.shape != original.shape or other_artifact.shape != original.shape:
        raise ValueError("Regional artifact axes differ from original EEG")
    groups = np.asarray(groups)
    frontal = groups == "frontal"
    nonfrontal = np.isin(groups, ["posterior", "central"])
    artifact = np.zeros_like(original)
    artifact[frontal] = frontal_artifact[frontal]
    artifact[nonfrontal] = other_artifact[nonfrontal]
    return original-artifact, artifact
