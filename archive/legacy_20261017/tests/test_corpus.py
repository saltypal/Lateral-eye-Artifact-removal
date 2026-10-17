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


def test_original_eeglab_trial_channel_axis_and_labels(tmp_path):
    from scipy.io import savemat
    from vmd_eog.osf_reader import read_osf
    values=np.zeros((5,1200,2))
    values[0,:,0]=11.; values[0,:,1]=22.
    values[1,:,0]=33.; values[1,:,1]=44.
    values[2,:,0]=55.; values[2,:,1]=66.
    values[3,:,0]=77.; values[3,:,1]=88.
    values[4,:,0]=5.; values[4,:,1]=6.
    locations=[{"labels":name,"type":kind} for name,kind in zip(
        ["Fp1","P3","HEOG","VEOG","artifactclasses"],["EEG","EEG","EOG","EOG","LABEL"])]
    path=tmp_path/"study01_p01_prep.set"
    savemat(path,{"EEG":{"nbchan":5,"pnts":1200,"trials":2,"srate":200.,"data":values,"chanlocs":locations}})
    source=read_osf(path)
    assert source["data"].shape==(2,5,1200)
    np.testing.assert_array_equal(source["data"][:,0,0],[11.,22.])
    np.testing.assert_array_equal(source["data"][:,2,0],[55.,66.])
    assert source["eeg_indices"]==[0,1] and source["eog_indices"]==[2,3]
    np.testing.assert_array_equal(source["sample_labels"][:,0],[5,6])


def test_train_only_fcm_fit_separates_feature_populations():
    rng=np.random.default_rng(15)
    neural=rng.normal([.1,.2,1.,.3],[.01,.01,.01,.01],size=(40,4))
    ocular=rng.normal([.9,.9,2.,.7],[.01,.01,.01,.01],size=(40,4))
    model=FCMSelection.fit(np.concatenate([neural,ocular]),clusters=2,seed=42)
    assert model.predict(neural).mean()<.1
    assert model.predict(ocular).mean()>.9
