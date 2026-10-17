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
