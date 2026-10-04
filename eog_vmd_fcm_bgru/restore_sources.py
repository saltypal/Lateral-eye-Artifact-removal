"""Restore missing original OSF sources on Kaggle with publisher SHA256 checks."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
from .provenance import save_json, sha256_file


def restore_study04(root, catalog_path, output):
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    restored, failures = [], []
    for entry in catalog["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not entry["path"].startswith("Dataset1_OSF/study04/"):
            raise ValueError("Invalid upstream source path")
        existed = path.is_file()
        try:
            if not existed:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".download")
                errors = []
                for url in [entry["download"], entry["provider_download"]]:
                    try:
                        request = urllib.request.Request(url, headers={"User-Agent": "EOGResearch/1.0"})
                        with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as handle:
                            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                                handle.write(block)
                        if temporary.stat().st_size != entry["bytes"] or sha256_file(temporary) != entry["sha256"]:
                            raise ValueError("Downloaded source size/SHA256 differs from the locked publisher catalog")
                        temporary.replace(path)
                        break
                    except Exception as error:
                        errors.append(repr(error))
                if not path.is_file():
                    raise RuntimeError("All publisher download endpoints failed: " + "; ".join(errors))
            if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise ValueError("Existing local source differs from upstream; it was preserved without overwrite")
            restored.append({**entry, "downloaded": not existed})
            print("Verified OSF upstream source", entry["path"], "downloaded", not existed, flush=True)
        except Exception as error:
            failures.append({"path": entry["path"], "error": repr(error)})
            print("OSF restoration failure", entry["path"], repr(error), flush=True)
        save_json(output / "restoration_progress.json", {"verified": restored, "failures": failures})
    if failures:
        raise RuntimeError("OSF restoration incomplete; inspect saved publisher download/hash errors")
    added = [entry for entry in restored if entry["downloaded"]]
    bundle = output / "osf_study04.eogbundle"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for entry in added:
            archive.write(root / entry["path"], arcname=entry["path"])
    manifest = {"source": catalog["source"], "catalog_sha256": sha256_file(catalog_path), "files": added,
                "all_study04_publisher_files_verified": len(restored),
                "archive": {"name": bundle.name, "bytes": bundle.stat().st_size, "sha256": sha256_file(bundle)}}
    save_json(output / "supplement_manifest.json", manifest)
    print("Restored source bundle persisted", manifest["archive"], flush=True)
    return manifest
