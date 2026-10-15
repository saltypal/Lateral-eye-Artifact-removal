"""Package verified source bytes into bounded archives; no signal processing."""
import json
from pathlib import Path
import zipfile

from .artifacts import verify_parent
from .broader_coupled_corpus import declared_parent
from .io import atomic_json, sha256_file


def readiness(input_root, config):
    if (config.get("scope") != "verified_original_artifact_bundle"
            or config.get("autovmd_enabled") is not False
            or config.get("reserved_access") != "closed until verified freeze"):
        raise ValueError("Artifact bundle policy differs")
    parent = declared_parent(input_root, "broader_frontal_summary.json", config["bundle_parent"])
    summary = json.loads((parent / "broader_frontal_summary.json").read_text())
    if summary.get("reserved_confirmation_opened") is not False:
        raise ValueError("Bundle source has opened reserved data")
    return parent


def run(input_root, output, config, profile):
    if profile != "full":
        raise ValueError("Byte bundle requires the complete original parent")
    parent = readiness(input_root, config)
    expected = verify_parent(parent)
    expected = {**expected, **{name: sha256_file(parent / name)
                              for name in ("artifact_manifest.json", "execution_state.json")}}
    (output / "shards").mkdir()
    shards, archive, names, byte_count = [], None, [], 0
    limit = 50 * 1024 * 1024

    def finish():
        if archive is not None:
            archive.close()
            path = output / "shards" / f"part-{len(shards):03d}.zip"
            shards.append({"path": path.relative_to(output).as_posix(), "sha256": sha256_file(path),
                           "bytes": path.stat().st_size, "entries": names.copy()})
            print("VERIFIED_BYTE_SHARD_SAVED", len(shards), len(names), flush=True)

    for relative in sorted(expected):
        source = (parent / relative).resolve()
        if not source.is_relative_to(parent.resolve()) or not source.is_file():
            raise ValueError("Bundle source path escapes or is unavailable")
        size = source.stat().st_size
        if archive is None or (names and byte_count + size > limit):
            finish()
            archive = zipfile.ZipFile(output / "shards" / f"part-{len(shards):03d}.zip", "w",
                                      compression=zipfile.ZIP_STORED, allowZip64=True)
            names, byte_count = [], 0
        archive.write(source, arcname=relative)
        names.append(relative)
        byte_count += size
    finish()
    atomic_json(output / "source_payload_manifest.json", expected)
    atomic_json(output / "bundle_index.json", {"source": config["bundle_parent"], "shards": shards,
        "source_files": len(expected), "source_bytes_verified_before_packaging": True,
        "archive_format": "ZIP_STORED; each original file retains its original SHA256"})
    atomic_json(output / "bundle_summary.json", {"source_files": len(expected), "shards": len(shards),
        "source_run_id": config["bundle_parent"]["run_id"], "signal_processing": False,
        "reserved_confirmation_opened": False, "qualification": False})
