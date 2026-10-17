"""Repair scope and identity checks, executed on Kaggle with other contracts."""
import hashlib
import json

import pytest

from vmd_eog import frontal_axis_native


def write_parent(directory, summary_name, summary):
    directory.mkdir()
    values = {
        summary_name: summary,
        "run_spec.json": {"run_id": "fixture", "git_sha": "fixture-source"},
    }
    manifest = {}
    for name, value in values.items():
        data = json.dumps(value).encode()
        (directory / name).write_bytes(data)
        manifest[name] = hashlib.sha256(data).hexdigest()
    (directory / "execution_state.json").write_text(json.dumps({"status": "complete"}))
    data = json.dumps(manifest).encode()
    (directory / "artifact_manifest.json").write_bytes(data)
    return {"run_id": "fixture", "git_sha": "fixture-source", "manifest": hashlib.sha256(data).hexdigest()}


def base_summary():
    return dict(scope="verified_frontal_only", reserved_confirmation_opened=False,
                autovmd_executed=False, calibration_hybrid=True, event_hybrid=True)


def test_repair_parent_requires_source_checksum_scope_and_completion(tmp_path):
    parent = tmp_path / "parent"
    expected = write_parent(parent, "summary.json", base_summary())
    result, _ = frontal_axis_native.checked_parent(tmp_path, "summary.json", expected)
    assert result == parent
    with pytest.raises(ValueError, match="manifest differs"):
        frontal_axis_native.checked_parent(tmp_path, "summary.json", {**expected, "manifest": "wrong"})
    with pytest.raises(ValueError, match="source identity"):
        frontal_axis_native.checked_parent(tmp_path, "summary.json", {**expected, "git_sha": "wrong"})
    (parent / "execution_state.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="incomplete"):
        frontal_axis_native.checked_parent(tmp_path, "summary.json", expected)


@pytest.mark.parametrize("field,value", [("reserved_confirmation_opened", True), ("autovmd_executed", True),
                                       ("calibration_hybrid", False), ("scope", "all_channels")])
def test_repair_parent_rejects_scope_drift(tmp_path, field, value):
    expected = write_parent(tmp_path / "parent", "summary.json", {**base_summary(), field: value})
    with pytest.raises(ValueError, match="scope differs"):
        frontal_axis_native.checked_parent(tmp_path, "summary.json", expected)


def test_exploratory_repair_accepts_unqualified_evidence_without_qualifying_it(tmp_path, monkeypatch):
    controlled = {**base_summary(), "K": 3, "alpha": 2000, "monotonic_events": True,
                  "weights_trained": False, "qualification": False}
    native = {**base_summary(), "profile": "pilot", "recordings": 2,
              "teacher_qualified": False, "native_parameter_retuning": False}

    def checked(root, name, expected):
        return (tmp_path / "controlled", controlled) if name == "frontal_transfer_summary.json" else (tmp_path / "native", native)

    monkeypatch.setattr(frontal_axis_native, "checked_parent", checked)
    assert frontal_axis_native.exploratory_parent(tmp_path) == (tmp_path / "controlled", tmp_path / "native")
    controlled["qualification"] = True
    with pytest.raises(ValueError, match="unqualified"):
        frontal_axis_native.exploratory_parent(tmp_path)
    controlled["qualification"] = False
    native["native_parameter_retuning"] = True
    with pytest.raises(ValueError, match="tuning scope"):
        frontal_axis_native.exploratory_parent(tmp_path)
