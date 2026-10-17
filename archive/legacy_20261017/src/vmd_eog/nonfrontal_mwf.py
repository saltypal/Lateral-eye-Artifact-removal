"""Two fixed GEVD-MWF controls using the validated posterior spatial solver."""
from dataclasses import dataclass
import json

import numpy as np

from .io import atomic_json
from .nonfrontal_challengers import balanced_gain, finite_matrix
from .nonfrontal_ridge import FrozenRidgeModel, fit_frozen_ridge
from .posterior import MWF, fit_mwf


METHODS = ("mwf_rank1", "mwf_rank2")


@dataclass
class MWFExperts:
    ids: np.ndarray
    field_model: object
    models: dict
    failures: dict


def fit_experts(calibration, policy):
    eeg = finite_matrix(calibration["eeg"], "calibration EEG")
    references = finite_matrix(calibration["references"], "calibration EOG")
    mask = np.asarray(calibration["mask"], dtype=bool)
    if mask.shape != (len(eeg),) or references.shape != (2, eeg.shape[-1]):
        raise ValueError("MWF calibration axes differ")
    ids = np.flatnonzero(mask)
    boundaries = tuple(int(x) for x in calibration["boundaries"])
    field_model = fit_frozen_ridge(eeg[ids], references, boundaries,
                                   tuple(policy["lags"]), policy["penalty"])
    models, failures = {}, {}
    for rank in (1, 2):
        method = "mwf_rank" + str(rank)
        try:
            models[method] = fit_mwf(eeg[ids], calibration["trial_types"],
                                     boundaries, lags=(0,), rank=rank)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            failures[method] = type(error).__name__ + ": " + str(error)
    return MWFExperts(ids, field_model, models, failures)


def estimate_fields(eeg, references, regions, mask, fitted, policy):
    eeg, references = finite_matrix(eeg, "EEG"), finite_matrix(references, "EOG")
    regions, mask = np.asarray(regions), np.asarray(mask, dtype=bool)
    if (regions.shape != (len(eeg),) or mask.shape != regions.shape
            or references.shape != (2, eeg.shape[-1])
            or not np.array_equal(np.flatnonzero(mask), fitted.ids)):
        raise ValueError("Scoring MWF montage differs")
    from .reference import lag_matrix
    _, valid = lag_matrix(references, fitted.field_model.lags)
    local = eeg[fitted.ids]
    field = fitted.field_model.artifact(references)
    artifacts, details = {}, {}
    for method in METHODS:
        artifacts[method] = np.zeros_like(eeg)
        details[method] = {"regions": {}}
        if method not in fitted.models:
            continue
        model = fitted.models[method]
        estimate = model.artifact(local, strength=1.)
        for region in (1, 2):
            rows = np.flatnonzero(regions[fitted.ids] == region)
            if not len(rows):
                continue
            _, association = balanced_gain(local[rows][:, valid], field[rows][:, valid], policy["gain_cap"])
            triggered = bool(association >= policy["association"])
            if triggered:
                artifacts[method][fitted.ids[rows]] = estimate[rows]
            details[method]["regions"][str(region)] = {
                "association": association, "triggered": triggered,
                "rank": model.rank, "lags": list(model.lags), "strength": 1.,
                "paired_target_access": False}
    return artifacts, details, dict(fitted.failures)


def save_experts(directory, fitted):
    directory.mkdir(parents=True, exist_ok=True)
    model = fitted.field_model
    arrays = {"ids": fitted.ids, "field_mean": model.reference_mean,
              "field_scale": model.reference_scale, "field_active": model.active_columns,
              "field_coefficients": model.coefficients}
    metadata = {"field": model.metadata, "models": {}, "failures": fitted.failures}
    for method, expert in fitted.models.items():
        for key in ("mean", "operator", "eigenvalues"):
            arrays[method + "_" + key] = getattr(expert, key)
        metadata["models"][method] = {"lags": list(expert.lags), "rank": expert.rank}
    np.savez_compressed(directory / "fitted_arrays.npz", **arrays)
    atomic_json(directory / "fit_status.json", metadata)


def load_experts(directory):
    arrays = dict(np.load(directory / "fitted_arrays.npz", allow_pickle=False))
    status = json.loads((directory / "fit_status.json").read_text())
    metadata = status["field"]
    field = FrozenRidgeModel(tuple(metadata["lags"]), metadata["penalty"],
        arrays["field_mean"], arrays["field_scale"], arrays["field_active"], arrays["field_coefficients"],
        metadata["calibration_valid_samples"], metadata["calibration_channel_count"], metadata["predictor_count"])
    models = {method: MWF(tuple(info["lags"]), arrays[method + "_mean"],
        arrays[method + "_operator"], arrays[method + "_eigenvalues"], info["rank"])
        for method, info in status["models"].items()}
    return MWFExperts(arrays["ids"], field, models, status["failures"])
