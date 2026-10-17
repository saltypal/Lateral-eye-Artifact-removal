"""Readable EOG regression, ICA component regression and regularized GEVD–MWF.

Every class returns an artifact estimate in the original channel space. The
caller performs one subtraction. None of these methods requires a paired target.
"""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from .metrics import correlations


@dataclass
class Ridge:
    coefficients: np.ndarray
    reference_mean: np.ndarray
    reference_scale: np.ndarray

    def artifact(self, values, references):
        standardized = (references - self.reference_mean) / self.reference_scale
        return self.coefficients @ standardized


def fit_ridge(values, references, penalty=0.001):
    values, references = np.asarray(values, float), np.asarray(references, float)
    if values.ndim != 2 or references.shape != (2, values.shape[-1]):
        raise ValueError("Expected C×T EEG and aligned 2×T HEOG/VEOG")
    mean = references.mean(axis=-1, keepdims=True)
    scale = np.std(references, axis=-1, keepdims=True)
    if np.any(scale <= 1e-12):
        raise ValueError("Both EOG axes need nonzero calibration variance")
    standardized = (references - mean) / scale
    centered = values - values.mean(axis=-1, keepdims=True)
    gram = standardized @ standardized.T / values.shape[-1]
    coefficients = np.linalg.solve(gram + penalty * np.eye(2),
                                   standardized @ centered.T / values.shape[-1]).T
    return Ridge(coefficients, mean, scale)


def window_regression(values, references, penalty=0.001):
    return fit_ridge(values, references, penalty).artifact(values, references)


@dataclass
class ICARegression:
    mean: np.ndarray
    unmixing: np.ndarray
    mixing: np.ndarray
    reference_fit: Ridge
    selected: np.ndarray
    associations: np.ndarray
    iterations: int

    def artifact(self, values, references, whole_components=False):
        sources = self.unmixing @ (values - self.mean[:, None])
        correction = sources if whole_components else self.reference_fit.artifact(sources, references)
        correction = correction * self.selected[:, None]
        return self.mixing @ correction


def fit_ica(values, references, threshold=0.3, penalty=0.001, seed=42, max_components=24):
    values = np.asarray(values, float)
    rank = np.linalg.matrix_rank(values - values.mean(axis=-1, keepdims=True))
    components = min(rank, values.shape[0], max_components)
    if components < 2 or values.shape[-1] < 20 * components:
        raise ValueError("Insufficient ICA calibration rank or duration")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = FastICA(n_components=components, whiten="unit-variance", random_state=seed,
                        max_iter=1000, tol=0.0001)
        sources = model.fit_transform(values.T).T
    if any(isinstance(item.message, ConvergenceWarning) for item in caught):
        raise RuntimeError("FastICA did not converge; retain failure and pass through")
    associations = correlations(sources, references)
    selected = np.max(np.abs(associations), axis=1) >= threshold
    return ICARegression(model.mean_, model.components_, model.mixing_,
                         fit_ridge(sources, references, penalty), selected, associations, model.n_iter_)


@dataclass
class MWF:
    mean: np.ndarray
    operator: np.ndarray
    eigenvalues: np.ndarray
    strength: float

    def artifact(self, values, references=None):
        return self.strength * self.operator @ (values - self.mean[:, None])


def fit_mwf(ocular, rest, rank=2, strength=0.5):
    """Instantaneous adaptation of Somers et al. 2018; no neural model involved.

    Generalized eigenvectors diagonalize ocular and resting covariance in a
    whitened coordinate system. The row-space correction is the transpose of
    V diag(max(lambda-1,0)/lambda) inv(V), not its untransposed matrix.
    """
    if ocular.shape[0] != rest.shape[0] or min(ocular.shape[-1], rest.shape[-1]) < max(64, 2 * len(rest)):
        raise ValueError("MWF needs sufficient separate ocular and resting calibration")
    mean = np.concatenate([ocular, rest], axis=-1).mean(axis=-1)
    ocular_cov = LedoitWolf(assume_centered=True).fit((ocular-mean[:, None]).T).covariance_
    rest_cov = LedoitWolf(assume_centered=True).fit((rest-mean[:, None]).T).covariance_
    jitter = max(np.trace(rest_cov) / len(rest_cov) * 1e-8, 1e-20)
    eigenvalues, vectors = eigh(ocular_cov + jitter*np.eye(len(rest)), rest_cov + jitter*np.eye(len(rest)))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    weights = np.maximum(eigenvalues - 1, 0) / eigenvalues
    weights[min(rank, len(weights)):] = 0
    operator = ((vectors * weights[None]) @ np.linalg.inv(vectors)).T
    return MWF(mean, operator, eigenvalues, strength)
