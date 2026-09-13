"""Explicit configuration objects shared by training and inference."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VMDConfig:
    """Fixed VMD and filter contract for the EOG-removal pipeline."""

    sampling_hz: float = 200.0
    low_hz: float = 0.5
    high_hz: float = 40.0
    modes: int = 5
    alpha: float = 1000.0
    tolerance: float = 1e-7


@dataclass(frozen=True)
class ModelConfig:
    """Compact order-agnostic spatial BiGRU architecture settings."""

    temporal_features: int = 32
    gru_hidden: int = 48
    bidirectional: bool = True
    use_spatial: bool = True
    use_gate: bool = True
    use_vmd: bool = True
    use_fcm: bool = True
