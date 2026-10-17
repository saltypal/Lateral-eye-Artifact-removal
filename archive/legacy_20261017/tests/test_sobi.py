"""Independent identifiable-source and entropy oracles, executed on Kaggle."""
import numpy as np
import pytest
from scipy.signal import lfilter
from vmd_eog.sobi import joint_diagonalize,fit_sobi,approximate_entropy


def test_joint_diagonalization_known_commuting_covariances():
    angle=.43
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    matrices=np.stack([rotation@np.diag(values)@rotation.T for values in ([.8,.1],[.6,-.3],[.4,.2])])
    fitted,diagnostic=joint_diagonalize(matrices)
    assert diagnostic["converged"]
    transformed=fitted.T@matrices@fitted
    assert np.max(np.abs(transformed[:,0,1]))<1e-10
    np.testing.assert_allclose(fitted.T@fitted,np.eye(2),atol=1e-12)
    assert diagnostic["off_diagonal_after"]<=diagnostic["off_diagonal_before"]


def test_sobi_recovers_identifiable_ar_sources_and_preserves_rank_residual():
    generator=np.random.default_rng(312)
    sources=np.stack([lfilter([1.],[1.,-coefficient],generator.normal(size=16384))
                      for coefficient in (.9,.2,-.6)])
    sources-=sources.mean(axis=-1,keepdims=True)
    sources/=sources.std(axis=-1,keepdims=True)
    mixing=np.array([[1.,.5,.2],[-.2,1.,.4],[.6,-.3,1.],[.3,.6,-.2]])
    values=mixing@sources
    expert=fit_sobi(values)
    assert expert.diagnostics["converged"]
    assert expert.diagnostics["rank"]==3
    similarity=np.abs(np.corrcoef(expert.sources,sources)[:3,3:])
    assert (similarity.max(axis=0)>.97).all()
    assert len(set(similarity.argmax(axis=0)))==3
    np.testing.assert_allclose(expert.mixing@expert.sources+expert.mean[:,None],values,atol=1e-10)
    smaller=fit_sobi(values*1e-6)
    np.testing.assert_allclose(smaller.sources,expert.sources,rtol=1e-5,atol=1e-5)
    with pytest.raises(ValueError,match="delays"):
        fit_sobi(values,lags=(0,))


def test_entropy_matches_direct_count_oracle_and_signal_units():
    values=np.array([0.,.1,.4,.8,.7,.2,.0,.1,.4,.8,.7,.2])
    radius=.15*values.std()
    phi=[]
    for dimension in (2,3):
        vectors=np.array([values[index:index+dimension] for index in range(len(values)-dimension+1)])
        distances=np.max(np.abs(vectors[:,None,:]-vectors[None,:,:]),axis=-1)
        phi.append(np.log((distances<radius).mean(axis=1)).mean())
    expected=phi[0]-phi[1]
    assert approximate_entropy(values)==pytest.approx(expected)
    assert approximate_entropy(values*1e-6)==pytest.approx(expected)
    assert approximate_entropy(np.zeros(32))==0.
