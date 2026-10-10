"""Second-order separation of VMD modes; an explicit paper-inspired adaptation.

Whitening followed by Jacobi joint diagonalization of symmetric lagged
covariances implements the SOBI mechanism. Source means and rank residuals
stay in the input when selected artifacts are subtracted.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.spatial import cKDTree


def off_diagonal_energy(matrices):
    diagonal=np.diagonal(matrices,axis1=-2,axis2=-1)
    return float(np.sum(matrices**2)-np.sum(diagonal**2))


def joint_diagonalize(matrices,tolerance=1e-7,max_sweeps=100):
    matrices=np.asarray(matrices,float)
    if matrices.ndim!=3 or matrices.shape[1]!=matrices.shape[2] or not np.isfinite(matrices).all():
        raise ValueError("Finite square covariance matrices required")
    if not np.allclose(matrices,matrices.transpose(0,2,1),rtol=1e-10,atol=1e-12):
        raise ValueError("SOBI joint diagonalization requires symmetric matrices")
    if tolerance<=0 or max_sweeps<1:
        raise ValueError("Invalid joint diagonalization stopping rule")
    working=matrices.copy()
    size=working.shape[-1]
    rotation=np.eye(size)
    before=off_diagonal_energy(working)
    largest_angle=np.inf
    for sweep in range(max_sweeps):
        largest_angle=0.
        for first in range(size-1):
            for second in range(first+1,size):
                # Rotated off-diagonal terms are b*cos(2t)+(d-a)/2*sin(2t).
                features=np.column_stack((working[:,first,second],
                    .5*(working[:,second,second]-working[:,first,first])))
                gram=features.T@features
                if np.trace(gram)<=np.finfo(float).eps**2:
                    continue
                _,vectors=np.linalg.eigh(gram)
                direction=vectors[:,0]
                if direction[0]<0:
                    direction=-direction
                angle=.5*np.arctan2(direction[1],direction[0])
                largest_angle=max(largest_angle,abs(angle))
                if abs(angle)<=tolerance:
                    continue
                cosine,sine=np.cos(angle),np.sin(angle)
                givens=np.eye(size)
                givens[first,first]=givens[second,second]=cosine
                givens[first,second]=-sine
                givens[second,first]=sine
                working=givens.T@working@givens
                rotation=rotation@givens
        if largest_angle<=tolerance:
            break
    return rotation,{"sweeps":sweep+1,"converged":bool(largest_angle<=tolerance),
        "largest_rotation_rad":float(largest_angle),"off_diagonal_before":before,
        "off_diagonal_after":max(0.,off_diagonal_energy(working))}


@dataclass
class SOBIExpert:
    mean: np.ndarray
    unmixing: np.ndarray
    mixing: np.ndarray
    sources: np.ndarray
    diagnostics: dict


def fit_sobi(values,lags=(1,2,4,8,16,32),tolerance=1e-7,max_sweeps=100):
    values=np.asarray(values,float)
    if values.ndim!=2 or min(values.shape)<2 or not np.isfinite(values).all():
        raise ValueError("SOBI needs finite [component,time] input")
    if not lags or any(int(lag)!=lag or not 0<lag<values.shape[-1]//2 for lag in lags):
        raise ValueError("SOBI delays must be positive and shorter than half the segment")
    started=time.perf_counter()
    mean=values.mean(axis=-1)
    centered=values-mean[:,None]
    scale=float(np.sqrt(np.mean(centered**2)))
    if scale<=np.finfo(float).tiny:
        raise ValueError("SOBI has no nonconstant input")
    normalized=centered/scale
    eigenvalues,eigenvectors=np.linalg.eigh(normalized@normalized.T/values.shape[-1])
    retained=eigenvalues>eigenvalues.max()*1e-10
    rank=int(retained.sum())
    if rank<2 or values.shape[-1]<20*rank:
        raise ValueError("SOBI rank or duration insufficient")
    whitening=(eigenvectors[:,retained]/np.sqrt(eigenvalues[retained])).T
    whitened=whitening@normalized
    covariances=[]
    for lag in lags:
        lag=int(lag)
        covariance=whitened[:,lag:]@whitened[:,:-lag].T/(values.shape[-1]-lag)
        covariances.append(.5*(covariance+covariance.T))
    rotation,diagnostics=joint_diagonalize(np.stack(covariances),tolerance,max_sweeps)
    unmixing=rotation.T@whitening/scale
    mixing=np.linalg.pinv(unmixing)
    sources=unmixing@centered
    diagnostics.update(rank=rank,lags=list(lags),runtime_s=time.perf_counter()-started,
        amplitude_scale=scale,relative_rank_threshold=1e-10,
        rank_residual_relative=float(np.linalg.norm(centered-mixing@sources)/np.linalg.norm(centered)))
    return SOBIExpert(mean,unmixing,mixing,sources,diagnostics)


def approximate_entropy(values,embedding=2,radius_fraction=.15):
    """ApEn includes self matches; a constant signal has defined zero entropy."""
    values=np.asarray(values,float)
    if values.ndim!=1 or not np.isfinite(values).all() or len(values)<=embedding+1:
        raise ValueError("Finite sufficiently long source required for ApEn")
    if embedding<1 or int(embedding)!=embedding or radius_fraction<=0:
        raise ValueError("Invalid entropy embedding or radius")
    scale=float(values.std())
    if scale==0:
        return 0.
    normalized=(values-values.mean())/scale
    # nextafter makes the paper's strict '< r' convention explicit.
    radius=np.nextafter(float(radius_fraction),0.)
    frequencies=[]
    for dimension in (embedding,embedding+1):
        vectors=np.lib.stride_tricks.sliding_window_view(normalized,dimension).copy()
        counts=cKDTree(vectors).query_ball_point(vectors,radius,p=np.inf,return_length=True)
        frequencies.append(float(np.log(counts/len(vectors)).mean()))
    return frequencies[0]-frequencies[1]
