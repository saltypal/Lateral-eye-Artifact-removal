"""Publisher link repairs preserve source bytes and reject ambiguity."""
import pytest
from vmd_eog.brainvision import repair_links


def test_stale_header_links_resolve_unique_renamed_payload(tmp_path):
    header=tmp_path/"sub-032305.vhdr"
    original=b"Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nDataFile=sub-010006.eeg\nMarkerFile=sub-010006.vmrk\n"
    header.write_bytes(original)
    (tmp_path/"sub-032305.eeg").write_bytes(b"original EEG payload")
    (tmp_path/"sub-032305.vmrk").write_bytes(b"original marker payload")
    resolved,report=repair_links(header)
    assert header.read_bytes()==original
    text=resolved.read_text(encoding="utf-8-sig")
    assert "DataFile=sub-032305.eeg" in text and "MarkerFile=sub-032305.vmrk" in text
    assert len(report["changes"])==2
    assert (tmp_path/"sub-032305.eeg").read_bytes()==b"original EEG payload"


def test_stale_header_rejects_ambiguous_companions(tmp_path):
    header=tmp_path/"participant.vhdr"
    header.write_text("DataFile=absent.eeg\nMarkerFile=absent.vmrk\n")
    (tmp_path/"one.eeg").write_bytes(b"one")
    (tmp_path/"two.eeg").write_bytes(b"two")
    with pytest.raises(FileNotFoundError,match="uniquely"):
        repair_links(header)
