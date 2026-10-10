"""Dimensionless ocular features must agree in volts and microvolts."""
import numpy as np
from vmd_eog.frontal import mode_features
from vmd_eog.reference import correlations, project


def test_mode_features_invariant_to_eeg_and_eog_units():
    generator=np.random.default_rng(11)
    modes=generator.normal(size=(4,1024))
    refs=np.stack([modes[0]+generator.normal(size=1024),modes[1]])
    expected=mode_features(modes,refs)
    for eeg_scale,eog_scale in ((1e-6,1.),(1e-6,1e-6),(1e-9,1e-9),(1e6,1e-6)):
        np.testing.assert_allclose(mode_features(modes*eeg_scale,refs*eog_scale),expected,rtol=1e-10,atol=1e-12)
    zero=mode_features(np.zeros((2,1024)),refs)
    np.testing.assert_array_equal(zero,np.zeros((2,4)))


def test_lag_correlation_invariant_at_small_physical_amplitudes():
    generator=np.random.default_rng(18)
    refs=generator.normal(size=(2,1024))
    eeg=np.stack([refs[0],-refs[1],refs[0]+.1*generator.normal(size=1024)])
    np.testing.assert_allclose(correlations(eeg*1e-9,refs*1e-9),correlations(eeg,refs),atol=1e-12)


def test_ridge_invariant_to_reference_units_with_baseline():
    generator=np.random.default_rng(28)
    refs=generator.normal(size=(2,1024))+np.array([[.3],[-.2]])
    eeg=refs[:1]+generator.normal(size=(1,1024))
    baseline=np.array([.1,-.1])
    expected,_=project(eeg,refs,penalty=.01,reference_baseline=baseline)
    actual,_=project(eeg,refs*1e-14,penalty=.01,reference_baseline=baseline*1e-14)
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)
