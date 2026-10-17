"""Frozen error attribution contracts; numerical execution on Kaggle."""
import numpy as np
import pytest
from vmd_eog.regional_diagnostic import groups_for, energy_rows, energy_snr, readiness, validate_rows, assert_parity
from vmd_eog.contracts import require_approval


def sample():
    original = np.arange(1, 81, dtype=float).reshape(5, 16)
    return {"original": original, "target": original * .8,
            "names": np.array(["Fp1", "P3", "C3", "UNKNOWN", "P4"]),
            "regions": np.array([0, 1, 2, 3, 1]), "mask": np.array([True, True, True, True, False])}


def test_energy_partition_excludes_missing_channels_and_reconciles():
    saved = sample()
    available, groups = groups_for(saved)
    corrected = saved["original"].copy()
    corrected[0] *= .9
    rows = energy_rows(saved["original"], saved["target"], corrected, available, groups)
    assert rows[-1]["channels"] == 4
    assert sum(row["channels"] for row in rows[:-1]) == 4
    assert sum(row["output_error_share"] for row in rows[:-1]) == pytest.approx(1)
    assert rows[-1]["target_energy"] == pytest.approx(np.sum(saved["target"][:4] ** 2))


def test_unknown_correction_is_rejected():
    saved = sample()
    available, groups = groups_for(saved)
    corrected = saved["original"].copy()
    corrected[3, 0] += 1
    with pytest.raises(ValueError, match="pass-through"):
        energy_rows(saved["original"], saved["target"], corrected, available, groups)


def test_metadata_cannot_guess_a_frontal_channel():
    saved = sample()
    saved["regions"][3] = 0
    with pytest.raises(ValueError):
        groups_for(saved)


def test_identity_energy_and_undefined_statuses_remain_distinct():
    saved = sample()
    available, groups = groups_for(saved)
    rows = energy_rows(saved["original"], saved["target"], saved["original"], available, groups)
    assert rows[-1]["modification_energy"] == 0
    assert energy_snr(0, 0)[1] == "zero_target_energy"
    assert energy_snr(1, 0)[1] == "perfect_reconstruction"
    assert energy_snr(10, 1)[0] == pytest.approx(10)
    assert_parity(np.array([np.nan, np.inf]), np.array([np.nan, np.inf]), "snr_energy_db")


def test_diagnostic_rejects_reserved_or_duplicate_examples():
    row = {"example_id": "one", "recipient": "p1", "partition": {"role": "confirmation", "fold": 0}}
    with pytest.raises(ValueError, match="Reserved"):
        validate_rows([row], 1, 1)
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows([row, row], 2, 1)


def test_diagnostic_never_authorizes_autovmd():
    with pytest.raises(ValueError, match="stopped AutoVMD"):
        readiness("missing", {"schema_version": 4, "autovmd_enabled": True})


def test_diagnostic_manifest_mismatch_rejects_before_waveform_access(tmp_path):
    (tmp_path / "regional_summary.json").write_text("{}")
    (tmp_path / "artifact_manifest.json").write_text("{}")
    config = {"schema_version": 4, "autovmd_enabled": False,
              "scope": "frozen_regional_error_diagnostic", "reserved_access": "closed until verified freeze",
              "diagnostic_parent": {"manifest_sha256": "wrong"}}
    with pytest.raises(ValueError, match="manifest differs"):
        readiness(tmp_path, config)


def test_diagnostic_schema_does_not_open_models(tmp_path):
    with pytest.raises(RuntimeError, match="does not authorize models"):
        require_approval("student-paired", tmp_path, {"schema_version": 4})
