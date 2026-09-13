"""Reusable building blocks for VMD-FCM-Spatial-BiGRU EOG artifact removal.

The training notebook imports this package after cloning the project's GitHub
repository. Keeping the implementation here gives local and Kaggle runs the
same model and signal-processing contract.
"""

from .config import ModelConfig, VMDConfig
from .fuzzy import FCMState, fit_fcm, predict_membership
from .model import EfficientSpatialBiGRU
from .signal_features import bandpass, mode_descriptor, vmd_one

__all__ = [
    "EfficientSpatialBiGRU",
    "FCMState",
    "ModelConfig",
    "VMDConfig",
    "bandpass",
    "fit_fcm",
    "mode_descriptor",
    "predict_membership",
    "vmd_one",
]
