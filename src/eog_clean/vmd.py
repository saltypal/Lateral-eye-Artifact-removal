"""RMS-normalized VMD with two spectral iterates instead of 500.

The spectral update follows vmdpy 0.2 (tau=0, DC=False, uniform centers).
Copyright (c) 2019 Vinicius Carvalho & Eduardo Mazoni; MIT license retained
in third_party/vmdpy_LICENSE.txt. The original algorithm is Dragomiretskiy
and Zosso, DOI 10.1109/TSP.2013.2288675.

Only positive frequencies need storage. Preserve vmdpy's penultimate-iterate
return convention for reproducible comparison; this is an offline solver.
"""
import time
import numpy as np


def rolling_decompose(values, modes=5, alpha=1000, fs=200,
                      tolerance=1e-6, max_iterations=500):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 32 or not np.isfinite(values).all():
        raise ValueError("VMD needs a finite continuous vector of at least 32 samples")
    if modes < 2 or int(modes) != modes or alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("VMD needs an integer K>=2 and a positive finite alpha")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 3:
        raise ValueError("Invalid VMD stopping rule")
    padded = np.pad(values, (0, values.size % 2), mode="edge")
    scale = float(np.sqrt(np.mean(padded ** 2)))
    if scale <= np.finfo(float).tiny:
        raise ValueError("RMS-normalized VMD requires a nonzero signal")
    normalized = padded / scale
    half = normalized.size // 2
    mirrored = np.concatenate([normalized[:half][::-1], normalized,
                               normalized[-half:][::-1]])
    width = mirrored.size
    # Use the reference's exact frequency expression, including rounding.
    frequencies = np.arange(1, width + 1) / width - 0.5 - 1 / width
    frequencies = frequencies[width // 2:]
    target_spectrum = np.fft.fftshift(np.fft.fft(mirrored))[width // 2:]
    previous = np.zeros((modes, frequencies.size), dtype=np.complex128)
    centers = (0.5 / modes) * np.arange(modes)
    accumulator = np.zeros(frequencies.size, dtype=np.complex128)
    difference = tolerance + np.spacing(1)
    iterations = 0
    started = time.perf_counter()
    while difference > tolerance and iterations < max_iterations - 1:
        current = np.empty_like(previous)
        current_centers = np.empty_like(centers)
        accumulator = previous[-1] + accumulator - previous[0]
        for mode in range(modes):
            if mode:
                accumulator = current[mode - 1] + accumulator - previous[mode]
            denominator = 1 + alpha * (frequencies - centers[mode]) ** 2
            current[mode] = (target_spectrum - accumulator) / denominator
            energy = np.abs(current[mode]) ** 2
            total_energy = float(energy.sum())
            if total_energy <= np.finfo(float).tiny:
                raise RuntimeError("Degenerate VMD mode spectrum")
            current_centers[mode] = np.dot(frequencies, energy) / total_energy
        delta = current - previous
        difference = float(np.spacing(1) + np.sum(np.abs(delta) ** 2) / width)
        if not np.isfinite(difference):
            raise RuntimeError("Nonfinite VMD update")
        # vmdpy returns n-1 after computing iterate n. Retain that contract.
        returned_spectrum, returned_centers = previous, centers
        previous, centers = current, current_centers
        iterations += 1
    full_spectrum = np.zeros((modes, width), dtype=np.complex128)
    full_spectrum[:, width // 2:] = returned_spectrum
    reflected_indices = np.arange(1, width // 2 + 1)[::-1]
    full_spectrum[:, reflected_indices] = np.conj(returned_spectrum)
    full_spectrum[:, 0] = np.conj(full_spectrum[:, -1])
    vectors = np.fft.ifft(np.fft.ifftshift(full_spectrum, axes=-1), axis=-1).real
    vectors = vectors[:, width // 4:3 * width // 4][:, :values.size] * scale
    order = np.argsort(returned_centers)
    vectors = vectors[order]
    residual = values - vectors.sum(axis=0)
    diagnostics = {"centers_hz": (returned_centers[order] * fs).tolist(),
        "iterations": iterations, "hit_iteration_limit": iterations >= max_iterations - 1,
        "max_iterations": max_iterations, "last_update_difference": difference,
        "runtime_s": time.perf_counter() - started, "solver_amplitude_scale": scale,
        "rms_normalized_stopping": True, "stopping_tolerance": tolerance,
        "iteration_history_stored": False, "iteration_spectra_buffers": 2,
        "spectral_iterate_storage_bytes": int(2 * previous.nbytes),
        "residual_ratio": float(np.linalg.norm(residual) / max(np.linalg.norm(values), np.finfo(float).tiny)),
        "return_convention": "vmdpy 0.2 penultimate iterate"}
    return vectors.astype(np.float32), residual.astype(np.float32), diagnostics
