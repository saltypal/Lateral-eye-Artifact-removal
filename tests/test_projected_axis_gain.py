"""Signed-axis and subspace contracts for the lateral repair adaptation."""
import numpy as np

from vmd_eog.frontal_calibration_transfer import calibration_transfer


def fixture():
    rng = np.random.default_rng(2026)
    cal_refs = rng.normal(size=(2, 4096))
    refs = rng.normal(size=(2, 1024))
    cal_refs -= cal_refs.mean(-1, keepdims=True)
    refs -= refs.mean(-1, keepdims=True)
    spatial = np.array([[1., .3], [-1., .3], [.4, 1.], [0., 0.]])
    original = spatial @ (refs * np.array([1.8, .4])[:, None])
    cal = {"eeg": spatial @ cal_refs, "references": cal_refs}
    metadata = {"names": np.array(["F3", "F4", "Fz", "P3"]),
                "regions": np.array([0, 0, 0, 1]), "mask": np.ones(4, bool)}
    bundles = {i: (np.stack([1.8 * spatial[i,0] * refs[0], .4 * spatial[i,1] * refs[1],
                             np.zeros(1024)]), np.zeros(1024), {}) for i in range(3)}
    return original, refs, cal, metadata, bundles


def apply(original, refs, cal, metadata, bundles):
    return calibration_transfer(original, refs, cal, bundles=bundles, lags=(0,),
                                gain_mode="projected_axis", **metadata)


def test_projected_axis_recovers_distinct_gains_and_preserves_signed_polarity():
    original, refs, cal, meta, bundles = fixture()
    cleaned, artifact, details = apply(original, refs, cal, meta, bundles)
    assert np.linalg.norm(cleaned[:3]) < .05 * np.linalg.norm(original[:3])
    np.testing.assert_array_equal(cleaned[3], original[3])
    assert details[0]["axis_gains"][0] > details[0]["axis_gains"][1]
    flipped_cal = {**cal, "references": cal["references"] * np.array([-1., 1.])[:,None]}
    flipped = apply(original, refs * np.array([-1., 1.])[:,None], flipped_cal, meta, bundles)
    np.testing.assert_allclose(flipped[1], artifact, rtol=1e-10, atol=1e-12)


def test_mode_permutation_and_absent_axis_or_failed_modes_are_safe():
    original, refs, cal, meta, bundles = fixture()
    expected = apply(original, refs, cal, meta, bundles)[1]
    reordered = {i: (m[[2,0,1]], residual, info) for i,(m,residual,info) in bundles.items()}
    np.testing.assert_allclose(apply(original, refs, cal, meta, reordered)[1], expected, atol=1e-12)
    failed = {i: (m, residual, {"failure": "fixture"}) for i,(m,residual,info) in bundles.items()}
    np.testing.assert_array_equal(apply(original, refs, cal, meta, failed)[1], np.zeros_like(original))
    zero_axis = refs.copy()
    zero_axis[1] = 0.
    cal_zero = {**cal, "references": cal["references"].copy()}
    cal_zero["references"][1] = 0.
    result = apply(original, zero_axis, cal_zero, meta, bundles)
    assert np.isfinite(result[0]).all()
    assert result[2][0]["axis_gains"][1] == 0.
    empty = {i: (np.zeros_like(m), original[i], {}) for i,(m,residual,info) in bundles.items()}
    np.testing.assert_array_equal(apply(original, refs, cal, meta, empty)[1], np.zeros_like(original))


def test_projected_axis_respects_units_and_channel_permutation():
    original, refs, cal, meta, bundles = fixture()
    expected = apply(original, refs, cal, meta, bundles)[1]
    scale = 1e-6
    scaled_cal = {"eeg": cal["eeg"] * scale, "references": cal["references"] * scale}
    scaled_bundles = {i: (m * scale, r * scale, info) for i,(m,r,info) in bundles.items()}
    scaled = apply(original * scale, refs * scale, scaled_cal, meta, scaled_bundles)[1]
    np.testing.assert_allclose(scaled, expected * scale, rtol=1e-9, atol=1e-14)
    order = [2, 0, 3, 1]
    permuted_meta = {key: value[order] for key,value in meta.items()}
    permuted_cal = {**cal, "eeg": cal["eeg"][order]}
    permuted_bundles = {new: bundles[old] for new,old in enumerate(order) if old in bundles}
    actual = apply(original[order], refs, permuted_cal, permuted_meta, permuted_bundles)[1]
    np.testing.assert_allclose(actual, expected[order], rtol=1e-10, atol=1e-12)


def test_reference_supported_residual_repairs_missing_ocular_gain_and_anchor_limits_loss():
    original, refs, cal, meta, _ = fixture()
    original = np.array([[1., .3], [-1., .3], [.4, 1.], [0., 0.]]) @ refs
    bundles = {i: (np.stack([original[i] * .5, np.zeros(1024), np.zeros(1024)]),
                   original[i] * .5, {}) for i in range(3)}
    arguments = dict(eeg=original, references=refs, calibration=cal, bundles=bundles, lags=(0,), **meta)
    old = calibration_transfer(**arguments, gain_mode="shared")
    residual = calibration_transfer(**arguments, gain_mode="residual_shared")
    anchored = calibration_transfer(**arguments, gain_mode="anchored_shared")
    raw = calibration_transfer(original, refs, cal, lags=(0,), **meta)
    assert all(item["residual_selected"] for item in residual[2])
    assert np.linalg.norm(residual[0][:3]) < .35 * np.linalg.norm(old[0][:3])
    assert np.linalg.norm(anchored[0][:3] - raw[0][:3]) < .35 * np.linalg.norm(old[0][:3] - raw[0][:3])
