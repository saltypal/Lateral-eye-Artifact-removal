"""Independent analytic MWF fixtures and trial-join protection."""
import numpy as np
from vmd_eog.posterior import mwf_operator, boundary_valid, fit_mwf


def test_diagonal_mwf_reference_equation_and_rank():
    # Eigenvalues 9,3,1 imply artifact fraction 8/9,2/3,0.
    rest=np.diag([2.,4.,7.]); ocular=np.diag([18.,12.,7.])
    operator,eigenvalues=mwf_operator(ocular,rest,2)
    np.testing.assert_allclose(operator,np.diag([8/9,2/3,0]),atol=1e-12)
    one,_=mwf_operator(ocular,rest,1)
    np.testing.assert_allclose(one,np.diag([8/9,0,0]),atol=1e-12)


def test_mwf_nonorthogonal_generalized_eigenvectors():
    transform=np.array([[1.,.6],[.3,2.]])
    rnn=transform@transform.T
    ryy=transform@np.diag([5.,1.])@transform.T
    operator,_=mwf_operator(ryy,rnn,1)
    expected=np.linalg.inv(transform.T)@np.diag([.8,0])@transform.T
    np.testing.assert_allclose(operator,expected,atol=1e-12)


def test_lag_fit_excludes_trial_boundaries():
    valid=boundary_valid(100,[50],[-5,0,5])
    assert not valid[45:55].any()
    assert valid[:45].all() and valid[55:].all()


def test_mwf_requires_calibration_types():
    import pytest
    with pytest.raises(ValueError,match="ocular/rest"):
        fit_mwf(np.ones((3,300)),np.ones(300),rank=1)
