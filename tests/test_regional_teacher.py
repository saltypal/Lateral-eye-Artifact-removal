"""Numerical-contract tests for the regional teacher; run in the Kaggle stage."""
import numpy as np
import pytest

from eog_vmd_fcm_bgru.posterior_experts import ArtifactEstimate, GEVDMWFExpert, SGEYESUBExpert
from eog_vmd_fcm_bgru.reference_guided import lagged_reference_matrix, signed_lagged_reference_projection
from eog_vmd_fcm_bgru.regional_teacher import (
    FrontalVMDConfig,
    compose_disjoint_teacher,
    direct_eog_ridge_estimate,
    frontal_context,
    frontal_vmd_estimate,
    legacy_convex_blend,
)
from eog_vmd_fcm_bgru.teacher_search_v2 import _grid


def signals(samples=256):
    time = np.arange(samples) / 200
    heog = np.sin(2 * np.pi * 1.1 * time)
    veog = np.sign(np.sin(2 * np.pi * 0.7 * time)) * 0.4
    neural = 0.25 * np.sin(2 * np.pi * 10 * time)
    return np.stack([heog, veog]), neural


def test_lagged_design_does_not_wrap_and_joint_projection_preserves_unrelated_signal():
    eyes, neural = signals()
    design = lagged_reference_matrix(eyes, (-3, 0, 5))
    assert design.shape == (6, eyes.shape[1])
    assert not design[0, :3].any()
    mixed = (eyes[0] + neural)[None]
    projected, details = signed_lagged_reference_projection(mixed, eyes, (0,), penalty=1e-8)
    remaining = mixed - projected
    assert details["window_adaptive"]
    np.testing.assert_allclose(np.dot(remaining[0], neural), np.dot(neural, neural), atol=2e-4)


def test_gevd_mwf_is_frozen_and_only_modifies_declared_outputs():
    eyes, neural = signals(512)
    raw = np.stack([neural + eyes[0], neural * 0.5 + eyes[1], neural])
    quiet = np.zeros(raw.shape[1], dtype=bool)
    quiet[::3] = True
    state = GEVDMWFExpert.fit(raw, quiet, output_indices=[0, 1], shrinkage=1e-3, excess_threshold=1.001)
    estimate = state.estimate(raw)
    assert estimate.method == "gevd_mwf"
    assert estimate.output_mask.tolist() == [True, True, False]
    np.testing.assert_array_equal(estimate.artifact[2], 0)
    assert estimate.diagnostics["frozen_calibration"]


def test_disjoint_teacher_rejects_overlap_and_reconstructs_once():
    raw = np.arange(3 * 64, dtype=float).reshape(3, 64)
    first = np.zeros_like(raw)
    second = np.zeros_like(raw)
    first[0] = 1
    second[1] = 2
    a = ArtifactEstimate(first, [True, False, False], [1, 0, 0], "front")
    b = ArtifactEstimate(second, [False, True, False], [0, 1, 0], "posterior")
    clean, artifact, details = compose_disjoint_teacher(raw, a, b)
    np.testing.assert_allclose(clean, raw - artifact)
    assert details["unrouted_channels"] == [2]
    with pytest.raises(ValueError, match="overlap"):
        compose_disjoint_teacher(raw, a, ArtifactEstimate(first, [True, False, False], [1, 0, 0], "bad"))


def test_frontal_context_keeps_signed_bilateral_lateral_difference():
    raw = np.zeros((4, 64))
    raw[0] = -1
    raw[1] = 2
    context = frontal_context(raw, ["Fp1", "Fp2", "O1", "unknown"])
    np.testing.assert_allclose(context.common, 0.5)
    np.testing.assert_allclose(context.lateral, 3)
    assert context.availability["bilateral"]


def test_vmd_route_does_not_modify_posterior_and_direct_baseline_aligns():
    eyes, neural = signals(257)
    raw = np.stack([neural + eyes[0], neural])
    names = ["Fp1", "O1"]
    config = FrontalVMDConfig(modes=3, alpha=1000, max_iterations=50)
    estimate = frontal_vmd_estimate(raw, eyes, names, config)
    assert estimate.output_mask.tolist() == [True, False]
    np.testing.assert_array_equal(estimate.artifact[1], 0)
    baseline = direct_eog_ridge_estimate(raw, eyes, [0], lags=(0,))
    assert baseline.artifact.shape == raw.shape
    np.testing.assert_array_equal(baseline.artifact[1], 0)


def test_legacy_convex_blend_remains_one_subtraction_baseline():
    raw = np.ones((2, 16))
    a = np.full_like(raw, 2)
    b = np.full_like(raw, 4)
    corrected = legacy_convex_blend(raw, a, b, ["Fp1", "O1"], frontal=0.5, posterior=0.5, shared=0.5)
    np.testing.assert_allclose(corrected, raw - 3)


def test_declared_default_grid_is_forty_candidates():
    assert len(_grid({})) == 40


def test_sgeyesub_never_silently_substitutes_another_method():
    with pytest.raises(RuntimeError, match="unavailable"):
        SGEYESUBExpert.fit(None)

