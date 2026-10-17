"""Kaggle numerical fixtures for the fresh data contract."""
import numpy as np

from vmd_eog.data import _hemisphere, _preprocess, _region


def test_unknown_klados_metadata_stays_unknown():
    assert _region("UNKNOWN_0") == 3
    assert _hemisphere("UNKNOWN_0") == 3


def test_named_regions_and_hemispheres_are_deterministic():
    assert _region("Fp1") == 0
    assert _region("O2") == 1
    assert _hemisphere("F3") == 0
    assert _hemisphere("F4") == 1
    assert _hemisphere("Fz") == 2


def test_complete_record_preprocess_preserves_channel_time_shape_at_200hz():
    values = np.random.default_rng(2).normal(size=(2, 4096))
    result = _preprocess(values, 200)
    assert result.shape == values.shape
    assert result.dtype == np.float32
