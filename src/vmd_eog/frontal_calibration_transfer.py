"""Signed frontal calibration transfer with matched raw and VMD gain estimates.

Project extension, not a paper reproduction. Calibration estimates a spatial
ocular field; scoring EEG determines one bounded field gain. This particularly
matches the reference-generated controlled corpus, so native validation is
required before generalizing it to real ocular contamination.
"""
import numpy as np

from .frontal_methods import frontal_indices
from .posterior import boundary_valid
from .reference import lag_matrix, correlations


def calibration_transfer(eeg, references, calibration, names, regions, mask,
                         bundles=None, threshold=.4, lags=(-10,0,10), penalty=.01,
                         gain_mode="shared"):
    """Return correction in original units; clean targets are never accepted.

    A shared gain reduces independent per-channel projection of neural content.
    Raw field association decides intervention, equally for both matched arms.
    VMD only changes the signal used to estimate that gain. No scoring reference
    mean is removed from the transferred field; calibration defines its baseline.
    """
    original = np.asarray(eeg,dtype=np.float64)
    ids = frontal_indices(original,names,regions,mask)
    artifact = np.zeros_like(original)
    if not len(ids):
        return original.copy(),artifact,[]
    cal_eeg = np.asarray(calibration["eeg"],dtype=np.float64)
    cal_refs = np.asarray(calibration["references"],dtype=np.float64)
    references = np.asarray(references,dtype=np.float64)
    if (cal_eeg.ndim != 2 or cal_eeg.shape[0] != len(original) or
            cal_refs.shape != (2,cal_eeg.shape[-1]) or references.shape != (2,original.shape[-1]) or
            not np.isfinite(cal_eeg).all() or not np.isfinite(cal_refs).all() or
            not np.isfinite(references).all()):
        raise ValueError("Finite, same-channel calibration and aligned eye references required")
    if not 0 <= threshold <= 1 or penalty <= 0 or 0 not in lags:
        raise ValueError("Invalid declared calibration-transfer settings")
    if gain_mode not in ("shared", "axis"):
        raise ValueError("Gain mode must be shared or axis")
    design, valid = lag_matrix(cal_refs,lags)
    valid &= boundary_valid(cal_eeg.shape[-1],calibration.get("boundaries",()),lags)
    if valid.sum() < max(128,2*len(design)):
        raise ValueError("Insufficient calibration after trial-boundary exclusions")
    mean = design[:,valid].mean(-1,keepdims=True)
    scale = design[:,valid].std(-1,keepdims=True)
    active = scale[:,0] > 0
    safe_scale = np.where(active[:,None],scale,1.)
    standardized = (design-mean)/safe_scale
    standardized[~active] = 0
    cal_values = cal_eeg[ids] - cal_eeg[ids][:,valid].mean(-1,keepdims=True)
    covariance = standardized[:,valid] @ standardized[:,valid].T / valid.sum()
    cross = standardized[:,valid] @ cal_values[:,valid].T / valid.sum()
    coefficients = np.linalg.solve(covariance+penalty*np.eye(len(design)),cross)
    score_design, score_valid = lag_matrix(references,lags)
    score_design = (score_design-mean)/safe_scale
    score_design[~active] = 0
    field = coefficients.T @ score_design
    field[:,~score_valid] = 0
    original_valid = original[ids][:,score_valid]
    centered_input = original_valid-original_valid.mean(-1,keepdims=True)
    field_valid = field[:,score_valid]
    centered_field = field_valid-field_valid.mean(-1,keepdims=True)
    denominator = float(np.sum(centered_field**2))
    input_energy = float(np.sum(centered_input**2))
    association = float(np.sum(centered_input*centered_field) /
        np.sqrt(denominator*input_energy)) if denominator > 0 and input_energy > 0 else 0.
    selected = []
    failed = set()
    values = original[ids].copy()
    if bundles is not None:
        if set(bundles) != set(int(index) for index in ids):
            raise ValueError("Calibration hybrid must contain exactly the frontal mode rows")
        values[:] = 0
        for local,index in enumerate(ids):
            modes, residual, info = bundles[int(index)]
            if info.get("failure") or info.get("hit_iteration_limit"):
                failed.add(int(index))
                selected.append([])
                continue
            modes = np.asarray(modes,dtype=np.float64)
            if modes.shape != (3,original.shape[-1]) or not np.isfinite(modes).all():
                raise ValueError("Calibration hybrid requires finite fixed K3 vectors")
            # Mode association uses the existing declared 0.2 bank member.
            keep = np.max(np.abs(correlations(modes,references,20)),axis=1) >= .2
            values[local] = modes[keep].sum(0)
            selected.append(np.flatnonzero(keep).tolist())
    usable = np.asarray([int(index) not in failed for index in ids])
    observed = values[:,score_valid]-values[:,score_valid].mean(-1,keepdims=True)
    gain_denominator = float(np.sum(centered_field[usable]**2))
    gain = (float(np.sum(observed[usable]*centered_field[usable]))/gain_denominator
        if gain_denominator > 0 else 0.)
    # Fixed safety bound, not selected using the controlled target amplitudes.
    gain = float(np.clip(gain,0.,8.))
    # Decisions are saved as strict JSON; NumPy scalar booleans are not JSON values.
    triggered = bool(association >= threshold)
    axis_gains = None
    if gain_mode == "axis":
        # lag_matrix is lag-major: HEOG rows 0,2,4 and VEOG rows 1,3,5.
        # Calibration fixes each signed spatial field; scoring fits only two
        # montage-wide gains. This is not a per-channel target regression.
        axis_fields = np.stack([coefficients[axis::2].T @ score_design[axis::2]
                                for axis in range(2)])
        axis_fields[:,:,~score_valid] = 0
        centered_axes = axis_fields[:,:,score_valid]
        centered_axes -= centered_axes.mean(-1,keepdims=True)
        features = centered_axes[:,usable].reshape(2,-1)
        observation = observed[usable].reshape(-1)
        rms = np.sqrt(np.mean(features**2,axis=1)) if observation.size else np.zeros(2)
        active_axes = rms > 0
        axis_gains = np.zeros(2)
        if active_axes.any():
            normalized = features[active_axes]/rms[active_axes,None]
            gram = normalized @ normalized.T / observation.size
            rhs = normalized @ observation / observation.size
            solved = np.linalg.solve(gram+penalty*np.eye(len(gram)),rhs)
            axis_gains[active_axes] = np.clip(solved/rms[active_axes],0.,8.)
        estimated_field = np.einsum("a,act->ct",axis_gains,axis_fields)
    else:
        estimated_field = gain*field
    if triggered:
        artifact[ids[usable]] = estimated_field[usable]
    details = [{"channel":str(names[index]),"field_association":association,
        "gain":gain if axis_gains is None else None,"gain_mode":gain_mode,
        "axis_gains":axis_gains.tolist() if axis_gains is not None else None,
        "triggered":triggered,"lags":list(lags),"penalty":penalty,
        "selected_modes":selected[local] if bundles is not None else None,
        "failure":"scored_passthrough" if int(index) in failed else None,
        "inference_target_access":False} for local,index in enumerate(ids)]
    return original-artifact,artifact,details
