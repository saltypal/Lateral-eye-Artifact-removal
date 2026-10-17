"""Kaggle source-recovery contracts; no downloaded human EEG in fixtures."""
import copy
import hashlib
import tarfile

import pytest

from vmd_eog import corpus
from vmd_eog.reference_coupling_audit import choose_pairs


def metadata_fixture():
    recipients, rows = [], []
    split = {"partitions": {"lemon": {}, "osf": {}}}
    for index in range(2):
        recipient = f"lemon:sub-00000{index}"
        donor = f"osf:p{index}"
        partition = {"role": "development", "fold": index}
        recipients.append({"participant": recipient, "partition": partition,
                           "archive_sha256": "a" * 64, "source_header_sha256": "b" * 64})
        split["partitions"]["lemon"][recipient] = partition
        split["partitions"]["osf"][donor] = partition
        for start in (512, 0):
            rows.append({"recipient": recipient, "donor": donor, "partition": partition,
                         "example_id": f"sample{index}-{start}",
                         "calibration": f"calibration/sub-00000{index}-study01_p{index}_prep/calibration.npz",
                         "recipe": {"recipient_start": start}})
    return rows, recipients, split


def test_probe_selection_is_source_only_and_order_invariant():
    rows, recipients, split = metadata_fixture()
    chosen = choose_pairs(rows, recipients, split)
    assert chosen == choose_pairs(rows[::-1], recipients[::-1], split)
    assert len({pair["donor"] for pair in chosen}) == 2
    assert all(pair["recipient_start"] == 0 for pair in chosen)


def test_probe_rejects_reserved_or_mismatched_partitions():
    rows, recipients, split = metadata_fixture()
    altered = copy.deepcopy(split)
    altered["partitions"]["lemon"][rows[0]["recipient"]]["role"] = "confirmation"
    with pytest.raises(ValueError, match="reserved"):
        choose_pairs(rows, recipients, altered)
    altered = copy.deepcopy(split)
    altered["partitions"]["osf"][rows[0]["donor"]]["fold"] = 4
    with pytest.raises(ValueError, match="partition"):
        choose_pairs(rows, recipients, altered)


def test_probe_rejects_missing_hash_or_guessed_donor_identity():
    rows, recipients, split = metadata_fixture()
    recipients[0].pop("archive_sha256")
    with pytest.raises(ValueError, match="checksum"):
        choose_pairs(rows, recipients, split)
    rows, recipients, split = metadata_fixture()
    rows[1]["calibration"] = "calibration/sub-000000-study01_p9_prep/calibration.npz"
    with pytest.raises(ValueError, match="disagrees"):
        choose_pairs(rows, recipients, split)


def test_probe_rejects_duplicate_sources_and_unbounded_count():
    rows, recipients, split = metadata_fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        choose_pairs(rows, recipients + recipients[:1], split)
    with pytest.raises(ValueError, match="bounded"):
        choose_pairs(rows, recipients, split, count=3)


def test_expected_archive_hash_is_checked_before_extraction(tmp_path, monkeypatch):
    def download(url, path):
        path.write_bytes(b"changed source; not a tar archive")
    monkeypatch.setattr(corpus, "safe_download", download)
    with pytest.raises(ValueError, match="archive checksum"):
        corpus.prepare_lemon({"participant": "sub-000001", "url": "fixture",
                              "expected_archive_sha256": "0" * 64}, tmp_path,
                             {"window": 1024, "hop": 512}, "pilot")


def test_expected_header_hash_is_checked_before_signal_loading(tmp_path, monkeypatch):
    source = tmp_path / "original.vhdr"
    source.write_bytes(b"changed header; not a BrainVision recording")
    archive = tmp_path / "original.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(source, arcname="original.vhdr")
    payload = archive.read_bytes()
    monkeypatch.setattr(corpus, "safe_download", lambda url, path: path.write_bytes(payload))
    with pytest.raises(ValueError, match="header checksum"):
        corpus.prepare_lemon({"participant": "sub-000001", "url": "fixture",
                              "expected_archive_sha256": hashlib.sha256(payload).hexdigest(),
                              "expected_header_sha256": "0" * 64}, tmp_path,
                             {"window": 1024, "hop": 512}, "pilot")
