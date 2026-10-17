import numpy as np
import pytest
from eog_clean.spatial_author import calibration_sample_labels,fit_sgeyesub


def test_sample_classes_never_filter_or_cross_trial_boundary():
    item={'fs':400,'sample_labels':np.array([[1,1,2,2,3,3],[4,4,5,5,6,6]])}
    trials=[{'index':1,'eeg':np.zeros((2,3))},{'index':0,'eeg':np.zeros((2,3))}]
    np.testing.assert_array_equal(calibration_sample_labels(item,trials),[4,5,6,1,2,3])
    with pytest.raises(ValueError,match='sample-level'):
        calibration_sample_labels({'sample_labels':None},trials)


def test_sgeyesub_rejects_trial_condition_labels_before_author_execution(tmp_path):
    with pytest.raises(ValueError,match='classes1'):
        fit_sgeyesub(np.zeros((3,10)),np.array([1,2,3,4,1,2,3,4,1,2]),tmp_path,tmp_path)


def test_uncertainty_pairs_cluster_id_instead_of_row_position():
    import pandas as pd
    from eog_clean.spatial_report import bootstrap_difference
    first=pd.Series([2.,4.,6.],index=['a','b','c'])
    second=pd.Series([5.,1.,3.],index=['c','a','b'])
    result=bootstrap_difference(first,second)
    assert result['clusters']==3
    assert result['mean_difference']==1.
    assert result['ci95_lower']==result['ci95_upper']==1.
    with pytest.raises(ValueError,match='clusters'):
        bootstrap_difference(first.iloc[:1],second)


def test_scored_input_uses_archived_bits_and_rejects_larger_changes():
    from eog_clean.frontal_spatial import canonical_scoring_input
    original=np.array([[1.,2.,3.],[4.,5.,6.]],dtype=np.float32)
    archived=original[[1]].copy()
    archived[0,0]=np.nextafter(archived[0,0],np.float32(np.inf))
    restored,diagnostic=canonical_scoring_input(original,[1],archived)
    np.testing.assert_array_equal(restored[1],archived[0])
    np.testing.assert_array_equal(restored[0],original[0])
    assert diagnostic['different_float32_values']==1
    changed=archived.copy();changed[0,0]=np.nextafter(changed[0,0],np.float32(np.inf))
    with pytest.raises(AssertionError):canonical_scoring_input(original,[1],changed)
