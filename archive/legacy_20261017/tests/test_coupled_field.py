"""Synthetic producer contracts; all numerical execution belongs on Kaggle."""
import numpy as np
import pytest

from vmd_eog.coupled_field import fit_donor_field, generated_pair


def donor_fixture():
    rng = np.random.default_rng(42)
    refs = rng.normal(size=(2, 512)) + np.array([3., -2.])[:, None]
    eeg = np.array([[1., 2.], [-1., .5], [.3, -.2]]) @ refs + rng.normal(size=(3, 512)) * .1
    return eeg, refs, np.array([256])


def test_forward_has_no_affine_offset_and_is_signed_homogeneous():
    eeg, refs, cuts = donor_fixture()
    model = fit_donor_field(eeg, refs, cuts)
    excursions = model.excursions(refs)
    base = model.forward(excursions, cuts)
    np.testing.assert_array_equal(model.forward(np.zeros_like(refs), cuts), np.zeros_like(eeg))
    for amplitude in (-3., -.25, 0., .25, 3.):
        np.testing.assert_allclose(model.forward(amplitude * excursions, cuts), amplitude * base, atol=1e-12)


def test_reference_unit_changes_preserve_the_forward_field():
    eeg, refs, cuts = donor_fixture()
    model = fit_donor_field(eeg, refs, cuts)
    changed = fit_donor_field(eeg, refs * 100., cuts)
    np.testing.assert_allclose(model.forward(model.excursions(refs), cuts),
                               changed.forward(changed.excursions(refs * 100.), cuts), atol=1e-11)


def test_driver_cannot_cross_original_trial_joins():
    eeg, refs, cuts = donor_fixture()
    model = fit_donor_field(eeg, refs, cuts)
    excursions = model.excursions(refs)
    changed = excursions.copy()
    changed[:, 256:] += 1000.
    first = model.forward(excursions, cuts)
    second = model.forward(changed, cuts)
    np.testing.assert_array_equal(first[:, :256], second[:, :256])
    np.testing.assert_array_equal(first[:, 246:266], np.zeros((3, 20)))
    with pytest.raises(ValueError, match="boundaries"):
        model.forward(excursions, [256, 128])


def test_generated_reference_and_artifact_share_exactly_one_multiplier():
    eeg, refs, cuts = donor_fixture()
    model = fit_donor_field(eeg, refs, cuts)
    target = np.random.default_rng(2026).normal(size=eeg.shape)
    for level in (-5., 0., 5.):
        record = generated_pair(target, model.excursions(refs), model, level, cuts)
        np.testing.assert_allclose(record["eeg"] - target, record["true_artifact"], atol=1e-12)
        np.testing.assert_allclose(model.forward(record["references"], cuts), record["true_artifact"], atol=1e-12)
        measured = 10 * np.log10(np.sum(target ** 2) / np.sum(record["true_artifact"] ** 2))
        assert measured == pytest.approx(level)


def test_clean_injection_is_explicitly_idealized_zero_driver():
    eeg, refs, cuts = donor_fixture()
    model = fit_donor_field(eeg, refs, cuts)
    clean = generated_pair(eeg, model.excursions(refs), model, None, cuts)
    np.testing.assert_array_equal(clean["eeg"], eeg)
    np.testing.assert_array_equal(clean["references"], np.zeros_like(refs))
    np.testing.assert_array_equal(clean["true_artifact"], np.zeros_like(eeg))


def test_invalid_donor_or_zero_energy_mixture_fails_explicitly():
    eeg, refs, cuts = donor_fixture()
    with pytest.raises(ValueError, match="ocular driver"):
        fit_donor_field(eeg, np.ones_like(refs), cuts)
    model = fit_donor_field(eeg, refs, cuts)
    with pytest.raises(ValueError, match="Zero energy"):
        generated_pair(eeg, np.zeros_like(refs), model, 0., cuts)
