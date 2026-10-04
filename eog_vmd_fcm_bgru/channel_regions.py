"""Versioned exact-name anatomy. Unknown montage channels remain unknown."""
import numpy as np

DICTIONARY_VERSION = "standard-name-regions-v1"
FRONTAL = {"FP1", "FP2", "FPZ", "AFZ", "FZ"}
FRONTAL |= {f"{prefix}{number}" for prefix in ("AF", "F") for number in range(1, 11)}
POSTERIOR = {"PZ", "POZ", "OZ", "IZ", "O1", "O2"}
POSTERIOR |= {f"{prefix}{number}" for prefix in ("P", "PO") for number in range(1, 11)}
TRANSITIONAL = {"CZ", "CPZ", "FCZ", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"}
TRANSITIONAL |= {f"{prefix}{number}" for prefix in ("FC", "FT", "C", "CP", "TP") for number in range(1, 11)}
REGION_NAMES = ["frontal", "posterior", "central_temporal", "unknown"]


def region_ids(names: list[str]) -> np.ndarray:
    groups = []
    for name in names:
        key = name.strip().upper()
        groups.append(0 if key in FRONTAL else 1 if key in POSTERIOR else 2 if key in TRANSITIONAL else 3)
    return np.asarray(groups, dtype=np.int64)


def fuse_residuals(eeg, vmd_artifact, spatial_artifact, names, frontal=0.8, posterior=0.2, shared=0.5):
    if not (eeg.shape == vmd_artifact.shape == spatial_artifact.shape):
        raise ValueError("Both experts must estimate a residual against the same input")
    if len(names) != eeg.shape[0] or any(value < 0 or value > 1 for value in (frontal, posterior, shared)):
        raise ValueError("Invalid channel metadata or fusion coefficient")
    groups = region_ids(names)
    weights = np.full(len(names), shared, dtype=float)
    weights[groups == 0] = frontal
    weights[groups == 1] = posterior
    artifact = weights[:, None] * vmd_artifact + (1 - weights[:, None]) * spatial_artifact
    return eeg - artifact
