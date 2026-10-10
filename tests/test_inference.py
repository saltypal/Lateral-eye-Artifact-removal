import pytest
torch = pytest.importorskip("torch")
import numpy as np
from vmd_eog.inference import EEGCleaner
from vmd_eog.neural import DeploymentStudent


def test_whole_record_identity_covers_trial_edges_and_unknown_metadata():
    values = np.random.default_rng(42).normal(size=(3,1537)).astype(np.float32)
    cleaner = EEGCleaner(DeploymentStudent(),{"fs":200,"window":1024,"hop":512})
    result = cleaner.denoise_record(values,["F3","F4","unknown"])
    np.testing.assert_array_equal(result["cleaned"],values)
    assert result["artifact"].shape == values.shape
    assert result["offline"] and not result["diagnostics"]
