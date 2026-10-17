"""Numerical composition/source-contract fixtures: execute on Kaggle only."""
import numpy as np
import pytest

from vmd_eog.regional_integration import posterior_artifact, readiness, load_frontal
from vmd_eog.posterior_context import compose_correction


def test_original_input_composition_preserves_shared_rows():
    original = np.arange(30, dtype=float).reshape(3, 10)
    front = np.zeros_like(original)
    front[0] = 2
    cleaned, total = compose_correction(original, [1], np.ones((1, 10)) * 3, front)
    assert np.array_equal(cleaned[0], original[0] - 2)
    assert np.array_equal(cleaned[1], original[1] - 3)
    assert np.array_equal(cleaned[2], original[2])
    assert np.array_equal(cleaned + total, original)


def test_regional_readiness_rejects_autovmd_before_data_access(tmp_path):
    with pytest.raises(ValueError, match="stopped AutoVMD"):
        readiness(tmp_path, {"schema_version": 4, "autovmd_enabled": True})


def test_missing_front_archive_is_not_passed_as_success(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_frontal(tmp_path, {"example_id": "missing"}, {})


def test_inherited_identity_does_not_require_calibration_targets():
    data = {"eeg": np.ones((2, 16)), "regions": np.array([0, 1]),
            "mask": np.ones(2, bool), "channel_names": np.array(["Fp1", "P3"]),
            "hemispheres": np.array([-1, -1])}
    ids, artifact, failure = posterior_artifact(data, {}, {}, {"recipe": None}, {}, {})
    assert ids.tolist() == [1]
    assert not artifact.any() and failure is None
