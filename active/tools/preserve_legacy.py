"""Freeze tracked legacy source without moving live files or deleting results."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    root = Path(__file__).resolve().parents[2]
    archive = root / "archive"
    archive.mkdir(exist_ok=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    destination = archive / f"legacy_20261017_{sha[:12]}.zip"
    if destination.exists():
        raise FileExistsError(destination)
    subprocess.run(["git", "archive", "--format=zip", "--output", str(destination), sha], cwd=root, check=True)
    with zipfile.ZipFile(destination) as stream:
        if stream.testzip() is not None:
            raise ValueError("Legacy archive CRC failed")
        names = stream.namelist()
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    autovmd = root / "src/vmd_eog/autovmd.py"
    declaration = {
        "source_git_sha": sha, "archive": destination.name, "archive_sha256": digest,
        "tracked_entries": len(names),
        "autovmd_sha256": hashlib.sha256(autovmd.read_bytes()).hexdigest(),
        "preserved_results_location": "results/ (ignored by Git, retained in place)",
        "untracked_user_documents": "retained in place; not silently added or moved",
        "legacy_files_moved": False,
        "reason": "Preserve other chats' open files and live-path compatibility; active/ is isolated and imports no legacy package.",
    }
    (archive / "LEGACY_MANIFEST.json").write_text(json.dumps(declaration, indent=2), encoding="utf-8")
    print(json.dumps(declaration, indent=2))


if __name__ == "__main__":
    main()
