"""Archive packing retains the complete source manifest and original bytes."""
import json
import zipfile

from vmd_eog import artifact_bundle
from vmd_eog.io import atomic_json, sha256_file


def test_verified_bundle_retains_all_original_source_bytes(tmp_path, monkeypatch):
    parent, output = tmp_path / "parent", tmp_path / "out"
    parent.mkdir()
    output.mkdir()
    (parent / "first.txt").write_bytes(b"original first payload\n")
    (parent / "second.txt").write_bytes(b"original second payload\n")
    expected = {name: sha256_file(parent / name) for name in ("first.txt", "second.txt")}
    atomic_json(parent / "artifact_manifest.json", expected)
    atomic_json(parent / "execution_state.json", {"status": "complete"})
    monkeypatch.setattr(artifact_bundle, "readiness", lambda *_: parent)
    artifact_bundle.run(tmp_path, output, {"bundle_parent": {"run_id": "unit-fixture"}}, "full")
    index = json.loads((output / "bundle_index.json").read_text())
    manifest = json.loads((output / "source_payload_manifest.json").read_text())
    assert set(manifest) == {"first.txt", "second.txt", "artifact_manifest.json", "execution_state.json"}
    restored = {}
    for shard in index["shards"]:
        assert sha256_file(output / shard["path"]) == shard["sha256"]
        with zipfile.ZipFile(output / shard["path"]) as archive:
            assert archive.namelist() == shard["entries"]
            restored.update({name: archive.read(name) for name in archive.namelist()})
    assert set(restored) == set(manifest)
    for name, contents in restored.items():
        assert contents == (parent / name).read_bytes()
