"""Allowlisted deployment-model registry.

Teacher models intentionally do not appear here because they require VMD and/or
EOG inputs and therefore cannot be exported as EEG-only deployment bundles.
"""
from __future__ import annotations

from typing import Any

from .regional_student import RegionalEEGStudentV2
from .student import SharedChannelStudent


LEGACY_ARCHITECTURE = "shared_channel_student_v1"
REGIONAL_ARCHITECTURE = RegionalEEGStudentV2.architecture_id
DEPLOYMENT_ARCHITECTURES = frozenset({LEGACY_ARCHITECTURE, REGIONAL_ARCHITECTURE})


def build_deployment_model(architecture_id: str, architecture: dict[str, Any]):
    if architecture_id == LEGACY_ARCHITECTURE:
        return SharedChannelStudent(**architecture)
    if architecture_id == REGIONAL_ARCHITECTURE:
        return RegionalEEGStudentV2(**architecture)
    raise ValueError(f"Unsupported deployment architecture: {architecture_id}")


def architecture_metadata(model) -> tuple[str, dict[str, Any], int]:
    if isinstance(model, SharedChannelStudent):
        return LEGACY_ARCHITECTURE, {"features": model.features, "hidden": model.gru.hidden_size}, 1
    if isinstance(model, RegionalEEGStudentV2):
        return REGIONAL_ARCHITECTURE, model.architecture_config(), 2
    raise ValueError("Only allowlisted EEG-only student models may be exported")
