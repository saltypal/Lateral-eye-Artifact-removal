"""Explicit adaptations of SVMD–FCM and VME–GMETV, without EOG selectors.

The ocular papers' complete settings are unavailable. These are testable
adaptations, never advertised as exact reproductions. PySDKit supplies the
published decomposition algorithms; this file contains our declared selectors.
"""
import numpy as np
from scipy.stats import kurtosis


def higuchi_dimension(values, kmax=8):
    """Slope of log curve length versus log inverse sampling interval."""
    values = np.asarray(values, float)
    lengths, scales = [], []
    for step in range(1, min(kmax, len(values)//4)+1):
        curves = []
        for offset in range(step):
            sequence = values[offset::step]
            if len(sequence) > 1:
                curves.append(np.abs(np.diff(sequence)).sum() *
                              (len(values)-1)/((len(sequence)-1)*step*step))
        length = np.mean(curves)
        if length > 1e-15:
            lengths.append(np.log(length))
            scales.append(np.log(1/step))
    return float(np.polyfit(scales, lengths, 1)[0]) if len(scales) > 1 else 1.


def fuzzy_two_clusters(features, seed=42, tolerance=1e-6, maximum=200):
    """FCM, m=2; returns normalized memberships and convergence evidence."""
    features = np.asarray(features, float)
    if features.ndim != 2 or len(features) < 2 or not np.isfinite(features).all():
        raise ValueError('FCM needs at least two finite feature rows')
    rng = np.random.default_rng(seed)
    membership = rng.uniform(.1, 1, (len(features), 2))
    membership /= membership.sum(axis=1, keepdims=True)
    for iteration in range(maximum):
        weights = membership**2
        centers = weights.T@features / weights.sum(axis=0)[:, None]
        distances = np.maximum(((features[:, None]-centers)**2).sum(axis=-1), 1e-15)
        inverse = 1/distances
        updated = inverse/inverse.sum(axis=1, keepdims=True)
        difference = float(np.max(np.abs(updated-membership)))
        membership = updated
        if difference < tolerance:
            break
    return membership, centers, {'fcm_iterations':iteration+1,
                                 'fcm_converged':difference < tolerance}


def svmd_fcm(signal, fs=200, config=None):
    from pysdkit import SVMD
    config = config or {}
    original = np.asarray(signal, float)
    scale = max(float(np.sqrt(np.mean(original**2))), 1e-12)
    normalized = original/scale
    solver = SVMD(max_alpha=1000, tau=0, tol=1e-6, stopc=1,
                  max_iter=300, max_modes=config.get('max_modes', 16))
    modes, _, centers = solver.fit_transform(normalized, return_all=True)
    if modes.shape[-1] != len(original) or not np.isfinite(modes).all():
        raise ValueError('Invalid successive decomposition')
    window, hop = 200, 100
    starts = list(range(0, len(original)-window+1, hop))
    if starts[-1] != len(original)-window:
        starts.append(len(original)-window)
    features, indices = [], []
    for mode, vector in enumerate(modes):
        for start in starts:
            block = vector[start:start+window]
            features.append([kurtosis(block, fisher=False) if block.std()>1e-12 else 3.,
                             np.mean(np.abs(block)), block.std(), higuchi_dimension(block)])
            indices.append((mode, start))
    features = np.asarray(features)
    standardized = (features-features.mean(axis=0))/np.maximum(features.std(axis=0), 1e-8)
    memberships, cluster_centers, evidence = fuzzy_two_clusters(standardized)
    # Declared semantic heuristic: ocular activity tends to be large, smooth,
    # and transient. This rule is ours; the inaccessible paper rule is unknown.
    score = cluster_centers@np.array([.5, 1., 1., -1.])
    ocular_cluster = int(np.argmax(score))
    correction, coverage = np.zeros_like(modes), np.zeros_like(modes)
    taper = np.maximum(np.hanning(window), 1e-3)
    routing = []
    for row, (mode, start) in enumerate(indices):
        association = memberships[row, ocular_cluster]
        weight = association if association >= .8 and centers[mode]*fs <= 8 else 0.
        correction[mode, start:start+window] += weight*modes[mode, start:start+window]*taper
        coverage[mode, start:start+window] += taper
        routing.append([mode, start, association, weight])
    artifact = (correction/np.maximum(coverage, 1e-12)).sum(axis=0)*scale
    residual = original-modes.sum(axis=0)*scale
    return original-artifact, {'modes':modes*scale, 'residual':residual,
        'centers_hz':centers*fs, 'routing':np.asarray(routing),
        'mode_cap_reached':len(modes)>=solver.max_modes,
        'inner_convergence_exposed_by_library':False, **evidence}


def difference_adjoint(values):
    """D.T @ values for forward differences with nonperiodic boundaries."""
    return np.concatenate([-values[:1], values[:-1]-values[1:], values[-1:]])


def gmetv(signal, regularization=.5, gamma=.8, tolerance=1e-6, maximum=300):
    """Selesnick et al. 2020, Algorithm 3, equations 62a–c, C=cI.

    B=cD, c²=gamma/(4*lambda), so lambda*||B||² < gamma < 1.
    Inner L1 problem is exactly soft thresholding; classical TV uses Condat.
    This diagonal-C member is an adaptation, not the ocular paper's unknown B.
    """
    from condat_tv import tv_denoise
    if regularization <= 0 or not 0 <= gamma < 1:
        raise ValueError('Require lambda>0 and 0<=gamma<1 for convexity')
    y = np.ascontiguousarray(signal, dtype=np.float64)
    estimate = tv_denoise(y, regularization)
    coefficient = gamma/(4*regularization)
    if gamma == 0:
        return estimate, {'iterations':1, 'converged':True, 'convexity_bound':0.}
    for iteration in range(maximum):
        gradient = np.diff(estimate)
        inner = np.sign(gradient)*np.maximum(np.abs(gradient)-1/coefficient, 0)
        perturbation = coefficient*difference_adjoint(gradient-inner)
        updated = tv_denoise(np.ascontiguousarray(y+regularization*perturbation), regularization)
        difference = np.linalg.norm(updated-estimate)/max(np.linalg.norm(estimate), 1e-12)
        estimate = updated
        if difference < tolerance:
            break
    return estimate, {'iterations':iteration+1, 'converged':bool(difference<tolerance),
                      'convexity_bound':gamma}


def vme_gmetv(signal, fs=200, config=None):
    from pysdkit import VME
    config = config or {}
    original = np.asarray(signal, float)
    scale = max(float(np.sqrt(np.mean(original**2))), 1e-12)
    solver = VME(alpha=20000, omega_init=0, fs=fs, tau=0, tol=1e-7, max_iter=300)
    mode = solver.fit_transform(original/scale)
    artifact, evidence = gmetv(mode, regularization=config.get('lambda', .5))
    if not np.isfinite(artifact).all():
        raise ValueError('VME–GMETV produced nonfinite artifact')
    return original-artifact*scale, {'modes':mode[None]*scale,
        'residual':original-mode*scale, 'centers_hz':np.array([solver.omega*fs]),
        'vme_iterations':solver.n_iter, 'vme_iteration_cap_reached':solver.n_iter>=299,
        **evidence}
