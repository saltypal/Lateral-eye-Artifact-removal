"""Frontal scope, entropy, correction and source-isolation checks on Kaggle."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from vmd_eog.frontal_campaign import validate_policy
from vmd_eog.frontal_methods import (frontal_indices, fuzzy_entropy,
    estimate_components, reference_intervals)
from vmd_eog.frontal_replication import candidates, fit_detectors


def configuration():
    return json.loads((Path(__file__).parents[1] / "configs/frontal_replication.json").read_text())


def test_frontal_policy_blocks_autovmd_neural_and_arbitrary_recipes():
    config = configuration()
    validate_policy(config, "frontal-fixture")
    for stage in ("autovmd-cache", "autovmd-search", "router-train", "student-paired", "regional"):
        with pytest.raises(RuntimeError, match="checkpoint"):
            validate_policy(config, stage)
    for key, value in (("autovmd_enabled", True), ("scope", "all")):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_policy(changed, "frontal-fixture")
    changed = copy.deepcopy(config)
    changed["fixed_recipes"][0]["K"] = 5
    with pytest.raises(ValueError, match="fixed recipes"):
        validate_policy(changed, "frontal-fixture")


def test_verified_frontal_names_masks_and_permutation():
    values = np.zeros((5, 64))
    names = np.array(["F3", "FP2", "P3", "FC1", "E1"])
    regions = np.array([0, 0, 1, 2, 3])
    assert frontal_indices(values, names, regions).tolist() == [0, 1]
    assert frontal_indices(values, names, regions, [True, False, True, True, True]).tolist() == [0]
    order = np.array([4, 2, 1, 0, 3])
    indices = frontal_indices(values[order], names[order], regions[order])
    assert set(names[order][indices]) == {"F3", "FP2"}
    with pytest.raises(ValueError, match="verified"):
        frontal_indices(values, names, [0, 0, 0, 2, 3])


def test_fuzzy_entropy_against_scalar_pair_oracle_and_units():
    values = np.array([.1, .4, -.1, .2, .6, -.2, .3, .9, -.5, .4])
    unit = (values - values.mean()) / values.std()
    phi = []
    for dimension in (1, 2):
        vectors = [unit[i:i+dimension] - unit[i:i+dimension].mean()
                   for i in range(len(unit)-dimension+1)]
        weights = [np.exp(-max(abs(first-second))**2/.2)
                   for i, first in enumerate(vectors) for j, second in enumerate(vectors) if i != j]
        phi.append(np.mean(weights))
    expected = np.log(phi[0]) - np.log(phi[1])
    assert fuzzy_entropy(values, block=3) == pytest.approx(expected)
    assert fuzzy_entropy(values * 1e-6) == pytest.approx(expected)
    assert fuzzy_entropy(np.zeros(32)) == 0.


def test_interval_calibration_axes_no_target_and_no_boundary_leakage():
    generator = np.random.default_rng(25)
    calibration = generator.normal(size=(2, 4000))
    references = np.zeros((2, 128))
    references[0, 40:50] = 8.
    references[1, 85:95] = -8.
    support = reference_intervals(references, calibration, fs=200)
    assert support[0, 45] > .99 and support[1, 90] > .99
    assert support[:, :15].sum() == 0 and support[:, -10:].sum() == 0
    np.testing.assert_allclose(reference_intervals(references*1e-6, calibration*1e-6), support)
    np.testing.assert_allclose(reference_intervals(references[::-1], calibration[::-1]), support[::-1])


def test_mode_projection_permutation_and_preserved_residual():
    times = np.arange(1024)/200
    references = np.stack([np.sin(2*np.pi*2*times), np.cos(2*np.pi*3*times)])
    brain = .3*np.sin(2*np.pi*12*times)
    modes = np.stack([2*references[0], references[1], brain])
    residual = .1*np.sin(2*np.pi*35*times)
    original = modes.sum(0)+residual
    spec = dict(kind="projected", component_threshold=.6, lags=[0], penalty=.01, strength=1.)
    artifact, detail = estimate_components(modes, references, spec, references)
    reordered, _ = estimate_components(modes[[2,0,1]], references, spec, references)
    np.testing.assert_allclose(artifact, reordered, atol=1e-12)
    assert set(detail["selected"]) == {0,1}
    # Input-minus-artifact necessarily retains residual, unlike summing only
    # retained VMD modes. Compare its known 35-Hz contribution explicitly.
    np.testing.assert_allclose((original-artifact)-(modes.sum(0)-artifact), residual, atol=1e-12)


def test_fixed_candidate_bank_has_no_all_channel_or_adaptive_vmd():
    specs = candidates(configuration(), "full")
    assert {spec["K"] for spec in specs.values()} == {None, 3, 4}
    assert {spec["alpha"] for spec in specs.values()} == {None, 2000}
    svm = [spec for spec in specs.values() if spec["family"] == "svm_sobi_apen_fixed_adaptation"]
    assert len(svm) == 1 and svm[0]["K"] == 3


def test_detector_refuses_donor_crossfold_leakage(tmp_path):
    rows = []
    generator = np.random.default_rng(10)
    for fold in range(2):
        for condition in ("clean", "blink"):
            name = f"{fold}-{condition}.npz"
            np.savez(tmp_path/name, eeg=generator.normal(size=(1,1024)),
                channel_names=np.array(["F3"]), regions=np.array([0]), mask=np.array([True]))
            rows.append(dict(array_path=name, recipient=f"r{fold}", donor="same-donor",
                             partition={"fold":fold}, condition=condition))
    with pytest.raises(ValueError, match="leaks donor"):
        fit_detectors(tmp_path, rows, configuration())
