"""Calibration anchoring must not alter default projections or cross trials."""
import numpy as np
import pytest
from vmd_eog.reference import project, calibration_reference_baseline


def test_explicit_calibration_baseline_corrects_reference_mean_excursion():
    times=np.arange(1000)/200
    refs=np.stack([np.sin(2*np.pi*times)+3,np.cos(2*np.pi*times)])
    neural=np.sin(2*np.pi*11*times)
    contaminated=neural+2*refs[0]
    centered,_=project(contaminated,refs,penalty=1e-8)
    anchored,_=project(contaminated,refs,penalty=1e-8,reference_baseline=np.zeros(2))
    assert abs(centered.mean())<1e-12
    assert abs(anchored.mean()-6)<1e-6
    np.testing.assert_allclose(contaminated-anchored[0],neural,atol=1e-6)
    with pytest.raises(ValueError,match="lag-major reference rows"):
        project(contaminated,refs,reference_baseline=np.zeros(1))


def test_baseline_excludes_unavailable_samples_at_calibration_joins():
    refs=np.array([[1.,2.,3.,4.,90.,100.,110.,120.]])
    actual=calibration_reference_baseline(refs,[4],lags=(1,))
    np.testing.assert_allclose(actual,[(1+2+3+90+100+110)/6])
    with pytest.raises(ValueError,match="trial boundary"):
        calibration_reference_baseline(refs,[0],lags=(1,))


def test_projected_components_and_residual_close_to_direct_projection():
    generator=np.random.default_rng(10)
    components=generator.normal(size=(5,1024))
    refs=generator.normal(size=(2,1024))
    baseline=np.array([.2,-.3,.2,-.3,.2,-.3])
    for anchor in (None,baseline):
        projected,_=project(components,refs,(-10,0,10),.01,reference_baseline=anchor)
        direct,_=project(components.sum(0),refs,(-10,0,10),.01,reference_baseline=anchor)
        np.testing.assert_allclose(projected.sum(0),direct[0],atol=1e-12)
