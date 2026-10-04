"""Verify and merge a persisted source supplement without numerical imports."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_supplement(root, manifest_path, output):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    container = manifest["archive"]
    archive_path = manifest_path.parent / container["name"]
    if archive_path.stat().st_size != container["bytes"] or digest_file(archive_path) != container["sha256"]:
        raise ValueError("Source supplement container SHA256/size mismatch")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    if len(entries) != len(manifest["files"]):
        raise ValueError("Duplicate supplement source paths")
    with zipfile.ZipFile(archive_path) as archive:
        if set(archive.namelist()) != set(entries) or len(archive.namelist()) != len(entries):
            raise ValueError("Supplement ZIP inventory differs from verified manifest")
        for name, entry in entries.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or not name.startswith("Dataset1_OSF/study04/"):
                raise ValueError("Invalid supplement source path")
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, path.open("xb") as destination:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        destination.write(block)
            if path.stat().st_size != entry["bytes"] or digest_file(path) != entry["sha256"]:
                raise ValueError("Supplement file mismatch; existing files were preserved: " + name)
    output.write_text(json.dumps({"manifest": str(manifest_path), "manifest_sha256": digest_file(manifest_path),
                                 "archive_sha256": container["sha256"], "verified_added_files": len(entries),
                                 "source": manifest["source"]}, indent=2), encoding="utf-8")
    print("Verified and merged source supplement", len(entries), "files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    merge_supplement(arguments.root, arguments.manifest, arguments.output)
