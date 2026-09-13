"""Train-fold-only fuzzy mode characterization for VMD features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler


@dataclass
class FCMState:
    """The fitted scaler and FCM centers needed for deterministic inference."""

    scaler: StandardScaler
    centers: np.ndarray
    artifact_cluster: int


def fit_fcm(
    descriptors: np.ndarray,
    train_records: np.ndarray,
    seed: int,
) -> FCMState:
    """Fit FCM using descriptors from training records only.

    ``descriptors`` must have shape ``records x channels x modes x features``.
    No validation, test, or OSF features participate in fitting.
    """
    descriptor_array = np.asarray(descriptors, dtype=np.float32)
    record_indices = np.asarray(train_records, dtype=int)
    if descriptor_array.ndim != 4:
        raise ValueError("Expected descriptors with shape records x channels x modes x features.")
    if record_indices.size == 0:
        raise ValueError("FCM requires at least one training record.")

    training_features = descriptor_array[record_indices].reshape(-1, descriptor_array.shape[-1])
    scaler = StandardScaler().fit(training_features)
    normalized_features = scaler.transform(training_features).T
    centers, _, _, _, _, _, _ = fuzz.cluster.cmeans(
        normalized_features,
        c=2,
        m=2.0,
        error=1e-5,
        maxiter=500,
        seed=seed,
    )

    # A high low-frequency fraction and low spectral centroid identifies the
    # ocular-likely fuzzy cluster without hard-deleting an entire VMD mode.
    artifact_cluster = int(np.argmax(centers[:, 1] - 0.20 * centers[:, 0]))
    return FCMState(scaler=scaler, centers=centers, artifact_cluster=artifact_cluster)


def predict_membership(descriptors: np.ndarray, state: FCMState) -> np.ndarray:
    """Return soft ocular-likely memberships with shape records x channels x modes."""
    descriptor_array = np.asarray(descriptors, dtype=np.float32)
    if descriptor_array.ndim < 2:
        raise ValueError("Expected a descriptor array ending in the feature dimension.")

    feature_matrix = state.scaler.transform(
        descriptor_array.reshape(-1, descriptor_array.shape[-1])
    ).T
    memberships, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        feature_matrix,
        state.centers,
        m=2.0,
        error=1e-5,
        maxiter=500,
    )
    ocular_membership = memberships[state.artifact_cluster].reshape(descriptor_array.shape[:-1])
    if not np.all(np.isfinite(ocular_membership)) or not np.all(
        (ocular_membership >= 0.0) & (ocular_membership <= 1.0)
    ):
        raise RuntimeError("FCM produced invalid soft memberships.")
    return ocular_membership.astype(np.float32)
