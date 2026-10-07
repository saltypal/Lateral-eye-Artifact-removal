"""Numerical contracts run on Kaggle, never on the local source workstation."""
import numpy as np
import pytest
from eog_vmd_fcm_bgru.reference_guided import (
    aligned_correlations, reference_projection, selected_mode_residual,
    frontal_support_indices, posterior_ica_indices, routed_ica_residual,
    ReferenceModeSelector)


def signals():
    time = np.arange(1024) / 200
    eyes = np.stack([np.sin(2 * np.pi * 2.34375 * time), np.cos(2 * np.pi * 1.171875 * time)])
    brain = np.sin(2 * np.pi * 12.5 * time)
    return eyes, brain


def test_signed_correlations_handle_constants_and_reject_misalignment():
    eyes, brain = signals()
    vectors = np.stack([eyes[0], -eyes[1], brain, np.ones_like(brain)])
    actual = aligned_correlations(vectors, eyes)
    np.testing.assert_allclose(actual[0, 0], 1, atol=1e-12)
    np.testing.assert_allclose(actual[1, 1], -1, atol=1e-12)
    np.testing.assert_allclose(actual[3], 0, atol=0)
    assert np.max(np.abs(actual[2])) < 1e-12
    with pytest.raises(ValueError, match="aligned"):
        aligned_correlations(vectors[:, :-1], eyes)
    with pytest.raises(ValueError, match="finite"):
        aligned_correlations(vectors * np.nan, eyes)


def test_reference_projection_preserves_neural_activity_inside_a_mixed_mode():
    eyes, brain = signals()
    mixed = (eyes[0] + brain)[None]
    removed, _ = selected_mode_residual(mixed, eyes, threshold=0, projection=True)
    remaining = mixed[0] - removed
    # Correction must not discard the independent alpha signal in this mode.
    np.testing.assert_allclose(np.dot(remaining, brain), np.dot(brain, brain), atol=1e-9)
    projected = reference_projection(mixed, eyes, penalty=1e-6)
    np.testing.assert_allclose(projected[0], eyes[0], atol=2e-6)
    np.testing.assert_allclose(reference_projection(mixed, eyes * 0), 0, atol=0)
    np.testing.assert_allclose(reference_projection(mixed, eyes * 1e-6),
                               reference_projection(mixed, eyes), atol=1e-12)


def test_correlation_selection_is_reference_sign_invariant_and_ignores_uncorrelated_mode():
    eyes, brain = signals()
    vectors = np.stack([eyes[0], brain])
    removed, details = selected_mode_residual(vectors, eyes, threshold=0.4)
    reversed_removed, _ = selected_mode_residual(vectors, -eyes, threshold=0.4)
    np.testing.assert_allclose(removed, reversed_removed, atol=0)
    np.testing.assert_allclose(removed, eyes[0], atol=1e-12)
    assert details["mode_weights"][1] == 0


def test_train_only_fuzzy_selector_reloads_without_refitting(tmp_path):
    import pickle
    eyes, brain = signals()
    examples = [np.stack([eyes[0] * factor, eyes[1] * factor, brain]) for factor in [0.5, 1, 2, 3]]
    selector = ReferenceModeSelector.fit(examples, [eyes] * len(examples), clusters=3)
    before = selector.centers.copy()
    actual = selector.evidence(examples[0], eyes)
    assert actual["membership"].shape == (3, 3)
    np.testing.assert_allclose(actual["membership"].sum(axis=1), 1, atol=1e-8)
    assert np.all((actual["cluster_weight"] >= 0) & (actual["cluster_weight"] <= 1))
    np.testing.assert_array_equal(before, selector.centers)
    path = tmp_path / "selector.pkl"
    path.write_bytes(pickle.dumps(selector))
    restored = pickle.loads(path.read_bytes())
    np.testing.assert_allclose(restored.evidence(examples[0], eyes)["cluster_weight"], actual["cluster_weight"])


def test_frontal_support_preserves_name_identity_across_permutations():
    names = ["O1", "Fp2", "Cz", "F8", "Fp1", "Pz", "F7", "F3"]
    fitting, targets, support = posterior_ica_indices(names)
    assert [names[index] for index in targets] == ["O1", "Pz"]
    assert [names[index] for index in support] == ["Fp1", "Fp2", "F7", "F8"]
    permuted = names[::-1]
    _, _, reordered_support = posterior_ica_indices(permuted)
    assert [permuted[index] for index in reordered_support] == [names[index] for index in support]
    assert set(fitting) == set(targets + support)
    with pytest.raises(ValueError, match="Duplicate"):
        frontal_support_indices(["Fp1", "FP1", "O1"])
    with pytest.raises(ValueError, match="Verified posterior"):
        posterior_ica_indices(["unknown"] * 3)


def test_posterior_correction_uses_raw_frontal_support_without_modifying_it():
    eeg = np.arange(5 * 128, dtype=np.float64).reshape(5, 128)
    fitting, targets = [3, 0, 4, 1], [3, 0]
    class FakeICA:
        def residual(self, values, threshold):
            np.testing.assert_array_equal(values, eeg[fitting])
            return values * 0.5
    actual = routed_ica_residual(eeg, fitting, targets, FakeICA(), strength=0.25)
    np.testing.assert_allclose(actual[targets], eeg[targets] * 0.125)
    np.testing.assert_array_equal(actual[[1, 2, 4]], 0)
    with pytest.raises(ValueError, match="include"):
        routed_ica_residual(eeg, [1, 4], [0], FakeICA())
