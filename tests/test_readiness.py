import json
import pytest
from vmd_eog.readiness import verify_source_ledger
from vmd_eog.io import sha256_file
from vmd_eog.contracts import require_approval


def test_missing_source_fails_preflight_before_trial_loading(tmp_path):
    record = {"record_id":"missing", "path":"missing.set", "source_sha256":"a"*64,
              "partition":{"role":"development"}}
    with pytest.raises(ValueError, match="preflight"):
        verify_source_ledger(tmp_path, [record], tmp_path)
    assert json.loads((tmp_path/"source_preflight.json").read_text())["passed"] is False


def test_reserved_source_cannot_enter_readiness(tmp_path):
    path = tmp_path/"source.set"
    path.write_bytes(b"metadata")
    record = {"record_id":"reserved", "path":path.name, "source_sha256":sha256_file(path),
              "partition":{"role":"confirmation"}}
    with pytest.raises(ValueError, match="reserved"):
        verify_source_ledger(tmp_path, [record], tmp_path)


def test_parallel_learning_still_requires_readiness(tmp_path):
    config = {"schema_version":2,"model_authorization":"Parallel paired learning after verified research readiness"}
    with pytest.raises(RuntimeError, match="readiness"):
        require_approval("student-paired", tmp_path, config)
