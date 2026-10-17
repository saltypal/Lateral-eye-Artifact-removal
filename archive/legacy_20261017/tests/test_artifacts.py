import json
import pytest
from vmd_eog.artifacts import verify_parent
from vmd_eog.io import sha256_file


def test_parent_corruption_and_incomplete_runs_are_rejected(tmp_path):
    payload = tmp_path / "metrics.json"
    payload.write_text('{"snr": 15}')
    (tmp_path / "artifact_manifest.json").write_text(json.dumps({"metrics.json": sha256_file(payload)}))
    state = tmp_path / "execution_state.json"
    state.write_text('{"status": "complete"}')
    verify_parent(tmp_path, ("metrics.json",))
    payload.write_text('{"snr": 20}')
    with pytest.raises(ValueError, match="checksum"):
        verify_parent(tmp_path, ("metrics.json",))
    state.write_text('{"status": "failed"}')
    with pytest.raises(ValueError, match="incomplete"):
        verify_parent(tmp_path)
