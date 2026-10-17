import numpy as np
import pytest
from eog_clean.regional_comparison import compose
from eog_clean.metrics import paired


def test_combination_has_disjoint_corrections_and_untouched_middle():
    raw=np.arange(24,dtype=np.float32).reshape(4,6)
    first=raw[[0]]-1;last=raw[[3]]-2
    result=compose(raw,first,last,[0],[3])
    np.testing.assert_array_equal(result[[0]],first)
    np.testing.assert_array_equal(result[[3]],last)
    np.testing.assert_array_equal(result[[1,2]],raw[[1,2]])
    with pytest.raises(ValueError,match='disjoint'):
        compose(raw,first,last,[0],[0])


def test_whole_cap_snr_uses_pooled_error_instead_of_regional_db_sum():
    target=np.ones((2,8));estimate=np.array([[.5]*8,[0.]*8])
    result=paired(estimate,target)
    assert result['pooled_window_snr_db']==pytest.approx(10*np.log10(16/10))
    assert result['pooled_window_snr_db']!=pytest.approx(sum(result['snr_out_db']))
