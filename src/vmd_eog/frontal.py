"""Mode diagnostics and selective VMD correction; residual is always retained."""
import numpy as np
from .reference import project, correlations
from .vmd import rolling_decompose


def decompose(values,k,alpha,config):
    grid=config["classical_grid"]
    modes,residual,info=rolling_decompose(values,k,alpha,config["fs"],grid["tolerance"],grid["max_iterations"])
    frequencies=np.fft.rfftfreq(modes.shape[-1],1/config["fs"])
    power=np.abs(np.fft.rfft(modes,axis=-1))**2
    energy=power.sum(axis=1)
    centers=np.asarray(info["centers_hz"])
    bandwidth=np.sqrt((power*(frequencies[None,:]-centers[:,None])**2).sum(axis=1)/np.maximum(energy,1e-20))
    normalized=power/np.maximum(energy[:,None],1e-20)
    overlap=np.minimum(normalized[:,None,:],normalized[None,:,:]).sum(axis=-1)
    info.update(bandwidth_hz=bandwidth.tolist(),power_overlap=overlap.tolist(),
        adjacent_center_distance_hz=np.diff(centers).tolist(),
        close_center_diagnostic=(np.diff(centers)<1.).tolist(),mode_energy=energy.tolist())
    return modes,residual,info


def mode_features(modes,refs,fs=200):
    modes=np.asarray(modes,dtype=float)
    associations=np.max(np.abs(correlations(modes,refs,20)),axis=1)
    power=np.abs(np.fft.rfft(modes,axis=-1))**2
    frequencies=np.fft.rfftfreq(modes.shape[-1],1/fs)
    energy=power.sum(axis=-1)
    low=np.divide(power[:,frequencies<8].sum(axis=1),energy,out=np.zeros_like(energy),where=energy>0)
    centered=modes-modes.mean(axis=-1,keepdims=True)
    scale=np.sqrt(np.mean(centered**2,axis=-1,keepdims=True))
    normalized=np.divide(centered,scale,out=np.zeros_like(centered),where=scale>0)
    kurtosis=np.mean(normalized**4,axis=-1)
    fractions=energy/energy.sum() if energy.sum()>0 else np.zeros_like(energy)
    return np.column_stack([associations,low,np.log1p(kurtosis),fractions])


def correct_modes(modes,refs,lags=(0,),penalty=.01,threshold=.6,strength=1.,kind="projected",fcm=None):
    features=mode_features(modes,refs)
    association=features[:,0]
    selected=association>=threshold
    if fcm is not None:
        selected &= fcm.predict(features)>=.5
    if kind=="whole":
        estimated=(modes*selected[:,None]).sum(axis=0)
    elif kind=="projected":
        estimates,_=project(modes,refs,lags,penalty)
        estimated=(estimates*selected[:,None]).sum(axis=0)
    else: raise ValueError("Unknown mode correction")
    return strength*estimated,{"associations":association.tolist(),"selected_modes":np.flatnonzero(selected).tolist()}


class FCMSelection:
    """Train-only feature clustering; ocular cluster selected by training association."""
    def __init__(self,centers,mean,scale,ocular_clusters):
        self.centers=centers; self.mean=mean; self.scale=scale; self.ocular_clusters=ocular_clusters

    @classmethod
    def fit(cls,features,clusters=2,seed=42):
        import skfuzzy
        mean=features.mean(axis=0); scale=np.maximum(features.std(axis=0),1e-8)
        centers,membership,*_=skfuzzy.cluster.cmeans(((features-mean)/scale).T,clusters,2.,error=1e-6,maxiter=1000,seed=seed)
        weighted=membership@features[:,0]/np.maximum(membership.sum(axis=1),1e-20)
        ocular=np.flatnonzero(weighted>=.5)
        return cls(centers,mean,scale,ocular)

    def predict(self,features):
        standardized=(features-self.mean)/self.scale
        squared=((standardized[:,None,:]-self.centers[None,:,:])**2).sum(axis=-1)
        inverse=1/np.maximum(squared,1e-12)
        membership=inverse/inverse.sum(axis=1,keepdims=True)
        return membership[:,self.ocular_clusters].sum(axis=1)
