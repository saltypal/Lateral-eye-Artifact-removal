"""
preprocessing package — EEG loading, filtering, ICA, windowing, and splitting.

Public API
----------
    run_preprocessing()   — full pipeline, returns train/val/test splits
    ICANormalizer         — per-IC z-score normaliser (fit on train only)
    CLASSES               — dict mapping class name → index
    CLASS_NAMES           — list of class names (ordered by index)
    N_CLASSES             — 6
"""

from .preprocess import (
    run_preprocessing,
    ICANormalizer,
    _build_sample_labels,
    _get_task_types,
    CLASSES,
    CLASS_NAMES,
    N_CLASSES,
)

__all__ = [
    "run_preprocessing",
    "ICANormalizer",
    "_build_sample_labels",
    "_get_task_types",
    "CLASSES",
    "CLASS_NAMES",
    "N_CLASSES",
]
