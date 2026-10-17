"""Joint ocular association should survive reference polarity/collinearity."""
import numpy as np
from vmd_eog.metrics import ocular


def test_joint_ocular_metric_is_polarity_invariant_and_bounded():
    rng=np.random.default_rng(9)
    refs=rng.normal(size=(2,1024))
    eeg=np.stack([refs[0]+.5*refs[1],rng.normal(size=1024)])
    before=ocular(eeg,refs); flipped=ocular(eeg,refs*np.array([[-1],[1]]))
    assert 0<=before["joint_eog_r2"]<=1
    np.testing.assert_allclose(before["per_channel_joint_eog_r2"],flipped["per_channel_joint_eog_r2"],atol=1e-12)
    assert before["per_channel_joint_eog_r2"][0]>.999999


def test_joint_ocular_metric_handles_duplicated_reference():
    time=np.arange(1024)/200
    reference=np.sin(2*np.pi*2*time)
    result=ocular(np.stack([reference,np.cos(2*np.pi*20*time)]),np.stack([reference,reference]))
    assert result["per_channel_joint_eog_r2"][0]>.999999
    assert result["per_channel_joint_eog_r2"][1]<.01
