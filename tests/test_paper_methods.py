"""Method-specific numerical contracts; run on Kaggle, never locally."""
import numpy as np
import pytest
from eog_clean.paper_methods import (difference_adjoint, fuzzy_two_clusters,
                                     gmetv, higuchi_dimension)


def test_difference_adjoint_nonperiodic_boundaries():
    rng=np.random.default_rng(1)
    x,y=rng.normal(size=31),rng.normal(size=30)
    assert np.diff(x)@y == pytest.approx(x@difference_adjoint(y))


def test_fcm_memberships_normalized_and_clusters_separate():
    data=np.array([[-5.,-4.],[-4.,-5.],[4.,5.],[5.,4.]])
    membership,centers,detail=fuzzy_two_clusters(data)
    np.testing.assert_allclose(membership.sum(axis=1),1)
    assert detail['fcm_converged']
    assert np.argmax(membership[0])==np.argmax(membership[1])
    assert np.argmax(membership[0])!=np.argmax(membership[-1])


def test_higuchi_smooth_line_dimension():
    assert higuchi_dimension(np.arange(200.)) == pytest.approx(1.,abs=.02)


def test_gmetv_zero_gamma_is_classical_tv():
    from condat_tv import tv_denoise
    x=np.random.default_rng(2).normal(size=64)
    answer,detail=gmetv(x,.4,gamma=0)
    np.testing.assert_allclose(answer,tv_denoise(x,.4),atol=1e-12)
    assert detail['converged']


def test_gmetv_constant_and_translation_equivariance():
    x=np.r_[np.zeros(40),np.ones(40)*3]
    first,detail=gmetv(x,.5)
    second,_=gmetv(x+7,.5)
    np.testing.assert_allclose(second,first+7,atol=2e-5)
    constant,_=gmetv(np.ones(80)*4,.5)
    np.testing.assert_allclose(constant,4.,atol=1e-12)
    assert detail['convexity_bound']<1


def test_gmetv_unsafe_convexity_parameter_rejected():
    with pytest.raises(ValueError,match='convexity'):
        gmetv(np.ones(64),gamma=1.1)


def test_published_solver_classes_available_and_vme_reconstruction():
    from pysdkit import SVMD,VME
    t=np.arange(256)/200
    x=np.sin(2*np.pi*2*t)+.3*np.sin(2*np.pi*25*t)
    solver=VME(alpha=20000,omega_init=0,fs=200)
    mode=solver(x)
    assert mode.shape==x.shape and np.isfinite(mode).all()
    np.testing.assert_allclose(mode+(x-mode),x,atol=1e-12)
    assert 0<=solver.omega<=.5
    assert SVMD(max_modes=3).max_modes==3


def test_spectral_rrmse_uses_power_and_not_amplitude():
    from eog_clean.metrics import paired
    t=np.arange(256)/256
    target=np.sin(2*np.pi*8*t)[None]
    result=paired(2*target,target)
    assert result['rrmse_t'][0]==pytest.approx(1.)
    assert result['rrmse_s'][0]==pytest.approx(3.)


def test_gmetv_reduces_declared_convex_objective():
    x=np.r_[np.zeros(40),np.ones(40)*3]+np.random.default_rng(4).normal(0,.2,80)
    lam,gamma=.5,.8
    coefficient=gamma/(4*lam)
    def objective(estimate):
        gradient=np.diff(estimate)
        inner=np.sign(gradient)*np.maximum(np.abs(gradient)-1/coefficient,0)
        envelope=np.abs(inner).sum()+.5*coefficient*np.sum((gradient-inner)**2)
        return .5*np.sum((x-estimate)**2)+lam*(np.abs(gradient).sum()-envelope)
    answer,detail=gmetv(x,lam,gamma)
    assert detail['converged']
    assert objective(answer)<=objective(x)+1e-8


def test_native_montage_exact_labels_and_roundtrip():
    from eog_clean.author_models import pack_frontal_montage
    values=np.arange(20,dtype=float).reshape(2,10)
    padded,present,positions=pack_frontal_montage(values,['FZ','f3'],['FP1','FP2','F3','F4','F7','F8','Fz'])
    np.testing.assert_array_equal(padded[positions],values)
    np.testing.assert_array_equal(present,[False,False,True,False,False,False,True])
    assert np.all(padded[~present]==0)


def test_native_montage_never_aliases_fpz_to_fp1():
    from eog_clean.author_models import pack_frontal_montage
    with pytest.raises(ValueError):
        pack_frontal_montage(np.ones((1,10)),['FPz'],['FP1','FP2','F3'])


def test_mode_clean_heog_veog_column_order_on_orthogonal_fixture():
    from eog_clean.mode_audit import associations
    # Integer cycles make these vectors exactly orthogonal over the fixture.
    clean=np.sin(2*np.pi*31*np.arange(1024)/1024)
    eyes=np.stack([np.sin(2*np.pi*5*np.arange(1024)/1024),
                   np.cos(2*np.pi*9*np.arange(1024)/1024)])
    np.testing.assert_allclose(associations(clean,clean,eyes),[1,0,0],atol=1e-12)
    np.testing.assert_allclose(associations(eyes[0],clean,eyes),[0,1,0],atol=1e-12)
    np.testing.assert_allclose(associations(eyes[1],clean,eyes),[0,0,1],atol=1e-12)
