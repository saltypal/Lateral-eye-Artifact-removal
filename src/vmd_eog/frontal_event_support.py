"""Rest-calibrated reference event support: a declared project adaptation.

Only unscored EOG and calibration trial labels fit this detector. No scoring
condition, EEG target, or reconstruction error is accepted by its interfaces.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def reference_features(values, center, scale, difference_scale):
    normalized = (values-center[:,None])/scale[:,None]
    amplitude = np.std(normalized,axis=1)
    peak = np.quantile(np.abs(normalized),.95,axis=1)
    velocity = np.std(np.diff(values,axis=1),axis=1)/difference_scale
    return np.concatenate([np.log(np.maximum(amplitude,1e-12)),peak,
                           np.log(np.maximum(velocity,1e-12))])


@dataclass
class ReferenceEventSupport:
    classifier: object
    center: np.ndarray
    scale: np.ndarray
    difference_scale: np.ndarray
    threshold: float
    calibration_counts: dict
    window: int = 256
    hop: int = 128

    def support(self,references):
        references = np.asarray(references,dtype=float)
        if references.ndim != 2 or references.shape[0] != 2 or not np.isfinite(references).all():
            raise ValueError("Finite named HEOG/VEOG required for event support")
        length = references.shape[-1]
        if length < self.window:
            raise ValueError("Event-support scoring segment is too short")
        starts = sorted(set([*range(0,length-self.window+1,self.hop),length-self.window]))
        features = np.stack([reference_features(references[:,start:start+self.window],
            self.center,self.scale,self.difference_scale) for start in starts])
        decisions = self.classifier.decision_function(features) > self.threshold
        support = np.zeros(length)
        for start,event in zip(starts,decisions):
            if event:
                support[start:start+self.window] = 1.
        # 100 ms extension accommodates delayed fields; no extension crosses a
        # trial because callers supply each independent trial/window separately.
        support = maximum_filter1d(support,size=41,mode="constant")
        kernel = np.hanning(43)[1:-1]
        kernel /= kernel.sum()
        support *= np.convolve(support,kernel,mode="same")
        return support,{"support_mean":float(support.mean()),
            "support_threshold_logit":self.threshold,"calibration_counts":self.calibration_counts,
            "support_policy":"reference-only logistic detector; threshold above all calibration rest windows and 0.9 probability",
            "scoring_condition_access":False,"support_window":self.window,"support_hop":self.hop}


def fit_event_support(calibration):
    references = np.asarray(calibration["references"],dtype=float)
    labels = np.asarray(calibration.get("trial_types",[]))
    if references.ndim != 2 or references.shape[0] != 2 or labels.shape != (references.shape[-1],):
        raise ValueError("Event support requires aligned calibration reference trial labels")
    if not np.isfinite(references).all():
        raise ValueError("Nonfinite event calibration")
    boundaries = np.asarray(calibration.get("boundaries",[]),dtype=int)
    if np.any(boundaries<=0) or np.any(boundaries>=len(labels)) or np.any(np.diff(boundaries)<=0):
        raise ValueError("Invalid event calibration trial boundaries")
    rest = labels == 1
    if rest.sum() < 512:
        raise ValueError("Insufficient labeled resting calibration for event support")
    center = np.median(references[:,rest],axis=1)
    scale = np.std(references[:,rest],axis=1)
    # Exclude derivatives across joins or label transitions.
    rest_pairs = rest[:-1] & rest[1:]
    rest_pairs[boundaries-1] = False
    difference_scale = np.std(np.diff(references,axis=1)[:,rest_pairs],axis=1)
    if np.any(scale<=0) or np.any(difference_scale<=0) or not np.isfinite(difference_scale).all():
        raise ValueError("Degenerate resting reference calibration")
    features,targets = [],[]
    edges = [0,*boundaries.tolist(),len(labels)]
    for left,right in zip(edges[:-1],edges[1:]):
        for start in range(left,right-255,128):
            local_labels = labels[start:start+256]
            if np.all(local_labels==1):
                target = 0
            elif np.all(np.isin(local_labels,[2,3,4])):
                target = 1
            else:
                continue
            features.append(reference_features(references[:,start:start+256],center,scale,difference_scale))
            targets.append(target)
    targets = np.asarray(targets)
    counts = {"rest":int(np.sum(targets==0)),"ocular":int(np.sum(targets==1))}
    if min(counts.values()) < 4:
        raise ValueError("Event calibration needs four windows per class")
    features = np.asarray(features)
    classifier = make_pipeline(StandardScaler(),LogisticRegression(C=1.,
        class_weight="balanced",max_iter=1000,random_state=42))
    classifier.fit(features,targets)
    threshold = float(max(np.log(9.),classifier.decision_function(features[targets==0]).max()))
    return ReferenceEventSupport(classifier,center,scale,difference_scale,threshold,counts)


def gate_artifact(original,artifact,references,gate,diagnostics,gain_scaled=False):
    detector_references = np.asarray(references,dtype=float)
    detector_gain = 1.
    if gain_scaled:
        # Controlled mixtures change EEG ocular strength without scaling the
        # donor references. Convert detector excursions to calibration-equivalent
        # strength using the already estimated montage-wide field gain.
        gains = [row.get("gain") for row in diagnostics]
        if not gains or any(value is None or not np.isfinite(value) or value < 0 for value in gains):
            raise ValueError("Gain-adjusted support requires finite shared field gains")
        if any(value != gains[0] for value in gains):
            raise ValueError("Gain-adjusted support requires one montage-wide gain")
        detector_gain = float(gains[0])
        detector_references = gate.center[:,None]+detector_gain*(detector_references-gate.center[:,None])
    support,metadata = gate.support(detector_references)
    metadata.update(gain_adjusted_support=bool(gain_scaled),detector_gain=detector_gain,
                    original_evaluation_references_unchanged=True)
    corrected_artifact = np.asarray(artifact)*support[None]
    details = [{**row,**metadata} for row in diagnostics]
    return np.asarray(original,dtype=float)-corrected_artifact,corrected_artifact,details


@dataclass
class AxisAmplitudeSupport:
    """Monotonic HEOG/VEOG detector fitted only on resting calibration."""
    center: np.ndarray
    thresholds: np.ndarray
    calibration_counts: dict
    window: int = 256
    hop: int = 128

    def support(self,references):
        values = np.asarray(references,dtype=float)
        if values.ndim != 2 or values.shape[0] != 2 or not np.isfinite(values).all():
            raise ValueError("Finite named HEOG/VEOG required for event support")
        length = values.shape[-1]
        if length < self.window:
            raise ValueError("Event-support scoring segment is too short")
        starts = sorted(set([*range(0,length-self.window+1,self.hop),length-self.window]))
        support = np.zeros(length)
        counts = [0,0]
        for start in starts:
            segment = values[:,start:start+self.window]-self.center[:,None]
            amplitude = np.stack([np.std(segment,axis=1),np.quantile(np.abs(segment),.95,axis=1)])
            # Axis-wise OR retains horizontal evidence even when VEOG is quiet.
            detected_axes = np.any(amplitude > self.thresholds,axis=0)
            counts = [count+int(event) for count,event in zip(counts,detected_axes)]
            if detected_axes.any():
                support[start:start+self.window] = 1.
        support = maximum_filter1d(support,size=41,mode="constant")
        kernel = np.hanning(43)[1:-1]
        kernel /= kernel.sum()
        support *= np.convolve(support,kernel,mode="same")
        return support,{"support_mean":float(support.mean()),
            "axis_thresholds":self.thresholds.tolist(),"axis_detected_windows":counts,
            "calibration_counts":self.calibration_counts,
            "support_policy":"monotonic per-axis STD or 95th amplitude exceeds all unscored calibration rest windows; HEOG OR VEOG",
            "scoring_condition_access":False,"support_window":self.window,"support_hop":self.hop}


def fit_axis_amplitude_support(calibration):
    references = np.asarray(calibration["references"],dtype=float)
    labels = np.asarray(calibration.get("trial_types",[]))
    if (references.ndim != 2 or references.shape[0] != 2 or
            labels.shape != (references.shape[-1],) or not np.isfinite(references).all()):
        raise ValueError("Aligned finite reference/condition calibration required")
    resting = labels == 1
    if resting.sum() < 512:
        raise ValueError("Insufficient resting event calibration")
    center = np.median(references[:,resting],axis=1)
    bounds = np.asarray(calibration.get("boundaries",[]),dtype=int)
    if np.any(bounds <= 0) or np.any(bounds >= len(labels)) or np.any(np.diff(bounds) <= 0):
        raise ValueError("Invalid event-calibration trial boundaries")
    features = []
    edges = [0,*bounds.tolist(),len(labels)]
    for left,right in zip(edges[:-1],edges[1:]):
        for start in range(left,right-255,128):
            if not np.all(labels[start:start+256] == 1):
                continue
            segment = references[:,start:start+256]-center[:,None]
            features.append(np.stack([np.std(segment,axis=1),np.quantile(np.abs(segment),.95,axis=1)]))
    if len(features) < 4:
        raise ValueError("Need four whole resting calibration windows")
    thresholds = np.max(features,axis=0)
    if not np.all(thresholds > 0):
        raise ValueError("Resting reference amplitude must be nonzero")
    return AxisAmplitudeSupport(center,thresholds,{"rest":len(features),"ocular_fitting":0})
