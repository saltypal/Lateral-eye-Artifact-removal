"""Checksummed source loading and persistent JSON artifacts; no numerical imports."""
import hashlib
import json
import os
from pathlib import Path
import zipfile


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
    os.replace(temporary, path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _extract_checked(archive_path, root):
    root = Path(root).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive paths')
        for name in names:
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError('Archive path escapes extraction root')
        archive.extractall(root)


def unpack_source(input_root, temporary_root, output):
    manifests = list(Path(input_root).rglob('source_manifest.json'))
    if len(manifests) != 1:
        raise ValueError('Attach exactly one original source bundle')
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    container = manifest['archive']
    archive = manifest_path.parent / container['name']
    if archive.stat().st_size != container['bytes'] or sha256_file(archive) != container['sha256']:
        raise ValueError('Source archive checksum mismatch')
    extraction = Path(temporary_root) / 'source'
    extraction.mkdir(parents=True, exist_ok=False)
    _extract_checked(archive, extraction)
    candidates = list(extraction.rglob('klados_contaminated_eeg.npy'))
    if len(candidates) != 1:
        raise ValueError('Expected one Klados source tree')
    root = candidates[0].parent.parent
    for item in manifest['files']:
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Invalid source path: ' + item['path'])
        if path.stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']:
            raise ValueError('Source file checksum mismatch: ' + item['path'])
    supplements = list(Path(input_root).rglob('supplement_manifest.json'))
    if len(supplements) > 1:
        raise ValueError('Ambiguous source supplement')
    for supplement in supplements:
        declaration = json.loads(supplement.read_text(encoding='utf-8'))
        item = declaration['archive']
        archive = supplement.parent / item['name']
        if archive.stat().st_size != item['bytes'] or sha256_file(archive) != item['sha256']:
            raise ValueError('Supplement checksum mismatch')
        with zipfile.ZipFile(archive) as stream:
            expected = {entry['path']: entry for entry in declaration['files']}
            if len(expected) != len(declaration['files']) or set(stream.namelist()) != set(expected):
                raise ValueError('Supplement inventory mismatch')
            for name, entry in expected.items():
                target = (root / name).resolve()
                if not target.is_relative_to(root.resolve()) or not name.startswith('Dataset1_OSF/study04/'):
                    raise ValueError('Invalid supplement path')
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with stream.open(name) as source, target.open('xb') as destination:
                        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                            destination.write(chunk)
                if target.stat().st_size != entry['bytes'] or sha256_file(target) != entry['sha256']:
                    raise ValueError('Supplement file mismatch')
    atomic_json(Path(output) / 'source_verification.json', {
        'manifest_sha256': sha256_file(manifest_path), 'archive_sha256': container['sha256'],
        'verified_original_files': len(manifest['files']), 'verified_supplements': len(supplements)})
    return root
