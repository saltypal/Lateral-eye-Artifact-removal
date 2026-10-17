import numpy as np
import pytest
from eog_clean.central import channel_group, group_indices, predictions
from eog_clean.methods import fit_ridge


def test_central_temporal_and_other_labels_are_disjoint():
    names=['FP1','F3','FC1','C3','Cz','CP2','FT7','T3','T5','TP8','P3','O1','unknown']
    central=group_indices(names,'central');temporal=group_indices(names,'temporal')
    assert [names[index] for index in central]==['FC1','C3','Cz','CP2']
    assert [names[index] for index in temporal]==['FT7','T3','T5','TP8']
    assert not set(central)&set(temporal)
    assert channel_group('Fz') is None and channel_group('Pz') is None
    with pytest.raises(ValueError,match='Duplicate'):
        group_indices(['C3','c3'],'central')


def test_fixed_strengths_subtract_once_and_do_not_refit_scoring_data():
    time=np.arange(2000)/200
    eyes=np.stack([np.sin(2*np.pi*time),np.cos(4*np.pi*time)])
    neural=np.sin(34*np.pi*time)[None]
    raw=neural+np.array([[2.,-1.]])@eyes
    fit=fit_ridge(raw[:,:1000],eyes[:,:1000],1e-9)
    before=fit.coefficients.copy()
    output=predictions(raw[:,1000:],eyes[:,1000:],{'ridge':fit,'mwf':None})
    np.testing.assert_allclose(output['ridge_100'],neural[:,1000:],atol=1e-7)
    np.testing.assert_allclose(output['ridge_025'],raw[:,1000:]-.25*(raw[:,1000:]-output['ridge_100']))
    np.testing.assert_array_equal(fit.coefficients,before)
    np.testing.assert_array_equal(output['identity'],raw[:,1000:])
    np.testing.assert_array_equal(output['mwf_100'],raw[:,1000:])
