"""Source-mapped metrics. Keep paper equations separate from project gates.

Signals use [channels, samples]. See docs/EVALUATION_PROTOCOL.md for source
equations, paper inconsistencies, undefined cases and disclosed adaptations.
No clean-reference reconstruction metric is defined for native unpaired EEG.
"""
import numpy as np
from scipy.signal import welch


EEGOAR_BANDS = {
    "delta": (1., 4.), "theta": (4., 8.), "alpha": (8., 13.),
    "beta1": (13., 19.), "beta2": (19., 30.),
}


def signal_array(value):
    value = np.asarray(value, dtype=float)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2 or value.shape[-1] < 2 or not np.isfinite(value).all():
        raise ValueError("Expected finite [channels, samples] with at least two samples")
    return value


def matched_signals(first, second):
    first, second = signal_array(first), signal_array(second)
    if first.shape != second.shape:
        raise ValueError("Signal shapes differ")
    return first, second


def ratio(numerator, denominator):
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def pearson(first, second):
    """Ordinary zero-lag Pearson, preserving undefined constant-channel cases."""
    first, second = matched_signals(first, second)
    x = first - first.mean(axis=-1, keepdims=True)
    y = second - second.mean(axis=-1, keepdims=True)
    return ratio((x*y).sum(-1), np.sqrt((x*x).sum(-1)*(y*y).sum(-1)))


def paired_channels(cleaned, target, contaminated=None):
    """IVMD-SOBI equations 16/17 plus standard Pearson and labelled SNR.

    SNR here is the explicit energy-ratio project target, not that paper's
    linear input RMS ratio or its PSNR. Never pool channel energies first.
    The literal printed equation 15 is included for auditing its discrepancy.
    """
    cleaned, target = matched_signals(cleaned, target)
    error_power = ((target-cleaned)**2).mean(-1)
    target_power = (target**2).mean(-1)
    target_centered = target-target.mean(-1, keepdims=True)
    cleaned_centered = cleaned-cleaned.mean(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 10*np.log10(ratio(target_power, error_power))
    snr[(target_power > 0) & (error_power == 0)] = np.inf
    result = {
        "mse": error_power, "rmse": np.sqrt(error_power),
        "rrmse_time": np.sqrt(ratio(error_power, target_power)),
        "pearson_cc": pearson(target, cleaned), "snr_energy_db": snr,
        "ivmd_eq15_printed_cov_over_target_var": ratio(
            (target_centered*cleaned_centered).mean(-1),
            (target_centered**2).mean(-1)),
    }
    if contaminated is not None:
        contaminated, _ = matched_signals(contaminated, target)
        input_error_power = ((contaminated-target)**2).mean(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            input_snr = 10*np.log10(ratio(target_power, input_error_power))
            input_ratio = np.sqrt(ratio(target_power, input_error_power))
        input_snr[(target_power > 0) & (input_error_power == 0)] = np.inf
        result.update(input_snr_energy_db=input_snr,
                      snr_improvement_db=snr-input_snr,
                      ivmd_eq13_input_rms_ratio=input_ratio)
    return result


def psnr_channels(cleaned, target, *, peak, scale_description):
    """IVMD-SOBI Eq18: peak=255 only with an explicit justified signal scale.

    No implicit conversion of volts/microvolts into 8-bit image intensities.
    PSNR must never be compared with the 15/20 dB reconstruction-SNR target.
    """
    if not np.isfinite(peak) or peak <= 0 or not scale_description.strip():
        raise ValueError("PSNR requires a positive peak and explicit signal-scale description")
    cleaned, target = matched_signals(cleaned, target)
    mse = ((target-cleaned)**2).mean(-1)
    with np.errstate(divide="ignore"):
        return 10*np.log10(peak**2/mse)


def native_condition_channels(cleaned, original, references, condition):
    """EEGOAR-Net section2.4.1 and Kobler author demo's condition metrics."""
    cleaned, original = matched_signals(cleaned, original)
    references = signal_array(references)
    if references.shape != (2, cleaned.shape[-1]):
        raise ValueError("References must be [HEOG, VEOG] at matching samples")
    if condition == "rest":
        return {"rest_rmse": np.sqrt(((cleaned-original)**2).mean(-1))}
    index = {"lateral": 0, "vertical": 1, "blink": 1}.get(condition)
    if index is None:
        raise ValueError("Paper-native condition must be rest/lateral/vertical/blink")
    ref = np.broadcast_to(references[index], cleaned.shape)
    return {"eeg_eog_abs_r_before": np.abs(pearson(original, ref)),
            "eeg_eog_abs_r_after": np.abs(pearson(cleaned, ref))}


def resting_spectrum(cleaned, original, fs, *, normalization_band):
    """EEGOAR Welch durations/bands; explicit unspecified implementation choices.

    Hann, constant detrending, half-open bins and normalization_band are
    declared adaptations because the inspected paper does not specify them.
    Reject short windows instead of silently reducing the 2-second duration.
    """
    cleaned, original = matched_signals(cleaned, original)
    segment = int(round(2*fs))
    if fs <= 0 or cleaned.shape[-1] < segment:
        raise ValueError("Resting PSD requires at least two seconds")
    lo, hi = normalization_band
    if not 0 <= lo < hi <= fs/2:
        raise ValueError("Invalid relative-power normalization band")
    arguments = dict(fs=fs, window="hann", nperseg=segment,
                     noverlap=int(round(fs)), nfft=segment,
                     detrend="constant", scaling="density", axis=-1)
    frequencies, before = welch(original, **arguments)
    _, after = welch(cleaned, **arguments)
    total = (frequencies >= lo) & (frequencies < hi)
    result = {"frequencies_hz": frequencies, "psd_before": before,
              "psd_after": after, "ivmd_eq19_absolute_psd_change": np.abs(before-after)}
    for name, (lower, upper) in EEGOAR_BANDS.items():
        mask = (frequencies >= lower) & (frequencies < upper)
        first = ratio(before[:, mask].sum(-1), before[:, total].sum(-1))
        second = ratio(after[:, mask].sum(-1), after[:, total].sum(-1))
        result[name+"_relative_before"] = first
        result[name+"_relative_after"] = second
        result[name+"_relative_change"] = second-first
    return result


def paired_permutation(first, second, *, comparisons, seed=42, repetitions=10000):
    """EEGOAR's two-sided paired permutation with Bonferroni adjustment.

    One independent participant summary per element; Monte Carlo sign flips
    and the conservative +1 estimate are declared implementation choices.
    Missing values require an explicit upstream eligibility decision.
    """
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.ndim != 1 or first.shape != second.shape or len(first) < 2:
        raise ValueError("Need at least two matched participant summaries")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Do not silently exclude missing participants")
    if comparisons < 1 or repetitions < 1:
        raise ValueError("Declare positive comparison family and repetition count")
    difference = first-second
    observed = abs(difference.mean())
    generator = np.random.default_rng(seed)
    exceedances = 0
    for start in range(0, repetitions, 1000):
        count = min(1000, repetitions-start)
        signs = generator.choice((-1., 1.), size=(count, len(first)))
        null = np.abs((signs*difference).mean(-1))
        exceedances += int((null >= observed).sum())
    probability = (exceedances+1)/(repetitions+1)
    return {"mean_paired_difference": float(difference.mean()),
            "p_two_sided": probability, "p_bonferroni": min(1., probability*comparisons),
            "participants": len(first), "permutations": repetitions,
            "comparison_family_size": comparisons}
