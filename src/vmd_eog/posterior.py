"""Calibration-only spatial experts, subtracting artifacts relative to raw input.

GEVD-MWF follows Somers et al.'s reference implementation: W=V Lambda^-1
Delta V^-1. Ryy is contaminated covariance; Rnn is non-ocular covariance.
Only positive selected generalized excess eigenvalues contribute.
"""
from dataclasses import dataclass
import numpy as np
import warnings
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf
from .reference import lag_matrix, correlations


def boundary_valid(length,boundaries,lags):
    valid=np.ones(length,bool)
    radius=max(abs(lag) for lag in lags)
    for edge in boundaries:
        valid[max(0,int(edge)-radius):min(length,int(edge)+radius)]=False
    return valid


@dataclass
class MWF:
    lags: tuple
    mean: np.ndarray
    operator: np.ndarray
    eigenvalues: np.ndarray
    rank: int

    def artifact(self,values,strength=1.):
        values=np.asarray(values,dtype=np.float64)
        embedded,valid=lag_matrix(values,self.lags)
        artifact=self.operator.T@(embedded-self.mean[:,None])
        artifact[:,~valid]=0
        return strength*artifact


def mwf_operator(ocular_cov,rest_cov,rank,mu=1.):
    """Dense independent formulation; no assumed Euclidean orthogonality of V."""
    eigenvalues,vectors=eigh(ocular_cov,rest_cov,check_finite=True)
    order=np.argsort(eigenvalues)[::-1]
    eigenvalues=eigenvalues[order]; vectors=vectors[:,order]
    excess=np.maximum(eigenvalues-1,0)
    excess[min(rank,len(excess)):]=0
    weights=excess/(eigenvalues+mu-1)
    operator=(vectors*weights[None,:])@np.linalg.inv(vectors)
    return operator,eigenvalues


def fit_mwf(values,trial_types,boundaries=(),lags=(0,),rank=2):
    values=np.asarray(values,dtype=np.float64)
    embedded,valid=lag_matrix(values,lags)
    valid &= boundary_valid(values.shape[-1],boundaries,lags)
    ocular=valid & np.isin(trial_types,[2,3,4])
    rest=valid & (np.asarray(trial_types)==1)
    if min(ocular.sum(),rest.sum())<max(128,2*len(embedded)):
        raise ValueError("MWF needs separate sufficient calibration ocular/rest samples")
    mean=embedded[:,valid].mean(axis=1)
    centered=embedded-mean[:,None]
    ryy=LedoitWolf(assume_centered=True).fit(centered[:,ocular].T).covariance_
    rnn=LedoitWolf(assume_centered=True).fit(centered[:,rest].T).covariance_
    jitter=max(np.trace(rnn)/len(rnn)*1e-8,1e-20)
    operator,eigenvalues=mwf_operator(ryy+np.eye(len(ryy))*jitter,rnn+np.eye(len(rnn))*jitter,rank)
    zero=lags.index(0)
    rows=np.arange(zero*len(values),(zero+1)*len(values))
    return MWF(tuple(lags),mean,operator[:,rows],eigenvalues,min(rank,len(eigenvalues)))


@dataclass
class ICAExpert:
    mean: np.ndarray
    unmixing: np.ndarray
    mixing: np.ndarray
    selected: np.ndarray
    associations: np.ndarray
    converged: bool

    def artifact(self,values,strength=1.,threshold=None):
        sources=self.unmixing@(values-self.mean[:,None])
        selected=self.selected if threshold is None else np.flatnonzero(np.max(np.abs(self.associations),axis=1)>=threshold)
        return strength*self.mixing[:,selected]@sources[selected]


def fit_ica(values,references,threshold=.6,seed=42,method="picard",*,boundaries=()):
    from picard import picard
    values=np.asarray(values,dtype=np.float64)
    mean=values.mean(axis=-1)
    centered=values-mean[:,None]
    covariance=centered@centered.T/values.shape[-1]
    spectrum=np.linalg.eigvalsh(covariance)
    components=int((spectrum>max(spectrum.max()*1e-7,1e-20)).sum())
    if components<2 or values.shape[-1]<20*components:
        raise ValueError("ICA calibration rank or duration insufficient")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        whiten,weights,sources=picard(centered,n_components=components,ortho=False,extended=True,
            whiten=True,max_iter=1000,tol=1e-7,random_state=seed,verbose=False)
    convergence_errors=[str(w.message) for w in caught if "converg" in str(w.message).lower()]
    if convergence_errors:
        raise RuntimeError("ICA convergence failed: "+"; ".join(convergence_errors))
    unmixing=weights@whiten
    mixing=np.linalg.pinv(unmixing)
    association=correlations(sources,references,20,boundaries=boundaries)
    selected=np.flatnonzero(np.max(np.abs(association),axis=1)>=threshold)
    # Picard emits a convergence warning when unsuccessful; flag the fit via
    # an explicit independence residual in experiment diagnostics as well.
    return ICAExpert(mean,unmixing,mixing,selected,association,True)
