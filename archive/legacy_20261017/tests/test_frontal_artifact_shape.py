"""Artifact projection invariants; numerical execution is Kaggle-only."""
import numpy as np
import pandas as pd
import pytest

import vmd_eog.frontal_artifact_shape as module
from vmd_eog.frontal_artifact_shape import BASELINES, dc_projection, reconcile_metrics, shape_correction
from vmd_eog.posterior_context import PAPER_KEYS


def span_fixture():
    time = np.arange(256)
    modes = np.stack([np.sin(2 * np.pi * time / 32), np.cos(2 * np.pi * time / 16), np.ones(256)])
    valid = np.ones(256, bool); valid[:10] = False; valid[-10:] = False
    artifact = np.zeros(256)
    artifact[valid] = 3 + 2 * modes[0, valid] - .5 * modes[1, valid]
    return artifact, modes, valid


def test_in_span_shape_and_dc_are_retained_with_zero_invalid_edges():
    artifact, modes, valid = span_fixture()
    result, details = dc_projection(artifact, modes, [0, 1], valid)
    np.testing.assert_allclose(result, artifact, rtol=1e-12, atol=1e-12)
    assert details["rank"] == 2 and details["valid_samples"] == 236
    assert result[valid].mean() == pytest.approx(artifact[valid].mean(), abs=1e-12)
    assert not result[~valid].any()


def test_projection_is_idempotent_and_discarded_variation_is_orthogonal():
    artifact, modes, valid = span_fixture()
    artifact[valid] += np.random.default_rng(42).normal(size=valid.sum())
    result, _ = dc_projection(artifact, modes, [0, 1], valid)
    twice, _ = dc_projection(result, modes, [0, 1], valid)
    np.testing.assert_allclose(twice, result, rtol=1e-12, atol=1e-12)
    centered = modes[:2, valid] - modes[:2, valid].mean(-1, keepdims=True)
    np.testing.assert_allclose(centered @ (artifact[valid] - result[valid]), np.zeros(2), atol=1e-11)


def test_rank_deficiency_constant_and_empty_spans_preserve_dc():
    artifact, modes, valid = span_fixture()
    modes[1] = modes[0]
    result, details = dc_projection(artifact, modes, [0, 1], valid)
    assert details["rank"] == 1
    for selection in ([], [2]):
        result, details = dc_projection(artifact, modes, selection, valid)
        np.testing.assert_allclose(result[valid], artifact[valid].mean())
        assert details["empty_or_degenerate_span"] and not result[~valid].any()


def test_mode_permutation_and_signal_unit_scaling_do_not_change_operator():
    artifact, modes, valid = span_fixture()
    baseline, _ = dc_projection(artifact, modes, [0, 1], valid)
    permuted, _ = dc_projection(artifact, modes[[1, 2, 0]], [2, 0], valid)
    np.testing.assert_allclose(permuted, baseline, rtol=1e-12, atol=1e-12)
    for scale in (1e-6, 1e6, -2.):
        scaled, _ = dc_projection(artifact * scale, modes * scale, [0, 1], valid)
        np.testing.assert_allclose(scaled / scale, baseline, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("selection", [[0, 0], [3], [0.5]])
def test_invalid_mode_selection_fails_closed(selection):
    artifact, modes, valid = span_fixture()
    with pytest.raises(ValueError):
        dc_projection(artifact, modes, selection, valid)


def test_nonfinite_shape_and_nonzero_invalid_edges_are_rejected():
    artifact, modes, valid = span_fixture()
    artifact[0] = 1.
    with pytest.raises(ValueError, match="lag edges"):
        dc_projection(artifact, modes, [0], valid)
    artifact[0] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        dc_projection(artifact, modes, [0], valid)


def wrapper_fixture(monkeypatch):
    artifact, modes, valid = span_fixture()
    original = np.tile(10 + modes.sum(0), (4, 1))
    fields = np.zeros_like(original); fields[0] = artifact
    names = np.array(["F3", "F4", "Pz", "F8"])
    regions, mask = np.array([0, 0, 1, 0]), np.array([True, False, True, True])
    bundles = {0: (modes, original[0] - modes.sum(0), {}),
               3: (np.zeros_like(modes), original[3].copy(), dict(failure="solver failure"))}
    common = dict(gain=1., field_association=.7, triggered=True, lags=[-10, 0, 10], penalty=.01, gain_mode="shared")
    details = [{**common, "channel": "F3", "selected_modes": [0, 1], "failure": None},
               {**common, "channel": "F8", "selected_modes": [], "failure": "scored_passthrough"}]
    monkeypatch.setattr(module, "correlations", lambda modes, references, lag: np.array([[.5, .1], [.1, .5], [.1, .1]]))
    return original, np.zeros((2, 256)), names, regions, mask, bundles, fields, details


def test_failed_masked_nonfrontal_rows_pass_through_and_subtraction_occurs_once(monkeypatch):
    inputs = wrapper_fixture(monkeypatch)
    cleaned, fields, diagnostics = shape_correction(*inputs)
    np.testing.assert_array_equal(cleaned[1:], inputs[0][1:])
    np.testing.assert_allclose(cleaned, inputs[0] - fields)
    assert diagnostics[1]["shape_failure"] == "inherited_scored_passthrough"
    assert diagnostics[0]["gain"] == 1. and diagnostics[0]["gain_refitted"] is False


def test_archived_mode_selection_drift_and_unknown_artifact_rows_stop(monkeypatch):
    inputs = list(wrapper_fixture(monkeypatch))
    inputs[-1][0]["selected_modes"] = [0]
    with pytest.raises(ValueError, match="selection differs"):
        shape_correction(*inputs)
    inputs = list(wrapper_fixture(monkeypatch)); inputs[-2][2, 20] = 1.
    with pytest.raises(ValueError, match="Nonfrontal"):
        shape_correction(*inputs)


def test_metric_parity_uses_underlying_metrics_and_normalizes_legacy_nulls_only():
    rows = []
    for arm in BASELINES:
        rows.append(dict(example_id="e", channel="F3", evaluated_family=arm,
            recipient="r", donor=None, condition="clean", input_snr_db=None,
            **{field: (np.inf if field == "snr_energy_db" else 0.) for field in PAPER_KEYS},
            alpha_db=0., beta_db=0., covariance_error=0., clean_change=0., cc_deterioration=0.))
    current = pd.DataFrame(rows)
    previous = current.copy(); previous["snr_energy_db"] = np.nan
    previous["cc_deterioration"] = 100.  # The obsolete comparator is intentionally excluded.
    assert reconcile_metrics(current, previous) == len(rows) * (len(PAPER_KEYS) + 4)
    previous.loc[0, "mse"] = .1
    with pytest.raises(AssertionError):
        reconcile_metrics(current, previous)
