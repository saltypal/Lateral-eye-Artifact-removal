"""Metadata and region contracts without biological assumptions."""
import numpy as np
import pytest
from vmd_eog.data import freeze_groups, eog_axes, channel_metadata
from vmd_eog.experiments import support_input
from vmd_eog.frontal import FCMSelection


def test_participant_reservation_precedes_windows_and_is_deterministic():
    ids=[f"participant-{i}" for i in range(20)]
    partitions=freeze_groups(ids,42,.2,5)
    assert partitions==freeze_groups(ids[::-1],42,.2,5)
    assert sum(p["role"]=="confirmation" for p in partitions.values())==4
    assert {p["fold"] for p in partitions.values() if p["role"]=="development"}==set(range(5))


def test_unoriented_eog1_eog2_are_not_guessed():
    with pytest.raises(ValueError,match="orientation"):
        eog_axes({"names":["Fp1","EOG1","EOG2"],"eog_indices":[1,2]})
    assert eog_axes({"names":["Fp1","VEOG","HEOG"],"eog_indices":[1,2]})==[2,1]


def test_signed_posterior_support_does_not_change_channel_axis():
    names=["Fp1","Fp2","P3","P4"]
    signal=np.stack([np.ones(100),-np.ones(100),np.arange(100),np.arange(100)*2.])
    data={"eeg":signal,**channel_metadata(names)}
    raw,rows,ids=support_input(data,"raw_frontal")
    np.testing.assert_array_equal(ids,[2,3])
    np.testing.assert_array_equal(raw[rows],signal[ids])
    full,rows,ids=support_input(data,"full")
    np.testing.assert_array_equal(full[rows],signal[ids])


def test_fcm_membership_order_and_extreme_features():
    model=FCMSelection(np.array([[0.,0.],[2.,2.]]),np.zeros(2),np.ones(2),np.array([1]))
    probability=model.predict(np.array([[0.,0.],[2.,2.]]))
    assert probability[0]<1e-8 and probability[1]>.999999
