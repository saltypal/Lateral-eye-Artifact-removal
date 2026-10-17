"""One-time reversible relocation of tracked legacy files; no result deletion."""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil


def main():
    root = Path(__file__).resolve().parents[2]
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    relocated = []

    def move(relative, target_relative):
        source, target = (root/relative).resolve(), (root/target_relative).resolve()
        if not source.is_relative_to(root) or not target.is_relative_to(root):
            raise ValueError("Relocation escapes project")
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.move(str(source), str(target))
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("Relocation changed file bytes")
        relocated.append({"from": relative, "to": target_relative, "sha256": digest})

    for name in tracked:
        if not name or name.startswith(("active/", "archive/", "docs/mentor/")) or name == ".gitignore":
            continue
        move(name, "archive/legacy_20261017/"+name)
    mapping = {
        "active/README.md": "README.md", "active/PLAN.md": "PLAN.md",
        "active/config.json": "configs/experiment.json", "active/pyproject.toml": "pyproject.toml",
        "active/REFERENCE_AUDIT.md": "docs/PAPERS.md", "active/WORK_LOG.md": "docs/WORK_LOG.md",
        "active/notebooks/Klados_OSF_Comparison.ipynb": "notebooks/Frontal_Comparison.ipynb",
        "active/tools/kaggle.py": "tools/kaggle.py",
        "active/tools/preserve_legacy.py": "archive/preserve_legacy.py",
    }
    for path in list((root/"active/src").rglob("*.py")):
        relative = str(path.relative_to(root)).replace("\\", "/")
        mapping[relative] = relative.replace("active/src/", "src/", 1)
    for path in (root/"active/tests").glob("*.py"):
        mapping[str(path.relative_to(root)).replace("\\", "/")] = "tests/"+path.name
    for path in (root/"active/provenance").glob("*"):
        mapping[str(path.relative_to(root)).replace("\\", "/")] = "docs/provenance/"+path.name
    for source, target in mapping.items():
        move(source, target)
    move("active/tools/simplify_tree.py", "archive/simplify_tree.py")
    # Only empty directories are removed. Other chats' untracked files survive.
    for directory in sorted((root/"active").rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    if not any((root/"active").iterdir()):
        (root/"active").rmdir()
    (root/"archive/RELOCATION_MANIFEST.json").write_text(json.dumps(relocated, indent=2), encoding="utf-8")
    print("Verified reversible file relocations:", len(relocated))


if __name__ == "__main__":
    main()
