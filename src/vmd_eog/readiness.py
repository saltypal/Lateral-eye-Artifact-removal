"""Real-data readiness is separate from scientific model qualification."""
import json
from pathlib import Path
import tempfile
import numpy as np
from .contracts import canonical_hash
from .io import atomic_json, sha256_file, unpack_source
from .experiments import corpus_parent


def verify_source_ledger(source, records, output):
    source = Path(source).resolve()
    failures = []
    for record in records:
        path = (source/record["path"]).resolve()
        if record["partition"]["role"] != "development":
            failures.append({"record_id":record["record_id"], "reason":"reserved source"})
        elif not path.is_relative_to(source) or not path.is_file():
            failures.append({"record_id":record["record_id"], "reason":"missing frozen source"})
        elif sha256_file(path) != record["source_sha256"]:
            failures.append({"record_id":record["record_id"], "reason":"source checksum differs"})
    atomic_json(Path(output)/"source_preflight.json", {
        "passed":not failures, "recordings":len(records), "failures":failures})
    if failures:
        raise ValueError("Frozen source preflight failed: " + json.dumps(failures))


def run_readiness(input_root, output, config):
    parent, rows = corpus_parent(input_root)
    source = unpack_source(input_root, tempfile.mkdtemp(prefix="eog-ready-"), output,
                           config["required_supplement_sha256"])
    records = [json.loads(path.read_text()) for path in (parent/"calibration").glob("*/record.json")]
    records = [record for record in records if record["dataset"] == "osf"]
    if not records:
        raise ValueError("Frozen OSF development ledger is absent")
    verify_source_ledger(source, records, output)
    # Explicit focused reproduction of the restored session, even if it belongs
    # to the closed confirmation partition: here read only development records.
    from .osf_reader import read_osf
    restored = next((record for record in records if record["record_id"] == "study04_p03_prep"), None)
    if restored is None:
        raise ValueError("Expected restored development session is absent")
    item = read_osf(source/restored["path"])
    atomic_json(output/"restored_session_fixture.json", {
        "record_id":restored["record_id"], "shape":list(item["data"].shape),
        "fs":item["fs"], "passed":True})
    recipient_folds, donor_folds = {}, {}
    paired_count = 0
    for row in rows:
        if row["target_kind"] != "controlled_recipient_reference":
            continue
        fold = row["partition"]["fold"]
        for mapping, identity in ((recipient_folds,row["recipient"]),(donor_folds,row["donor"])):
            if identity in mapping and mapping[identity] != fold:
                raise ValueError("Source identity crosses frozen outer folds")
            mapping[identity] = fold
        with np.load(parent/row["array_path"],allow_pickle=False) as arrays:
            eeg, target = arrays["eeg"], arrays["paired_reference"]
            if eeg.shape != target.shape or eeg.ndim != 2 or eeg.shape[-1] != config["window"]:
                raise ValueError("Paired waveform shape differs from contract")
            if not np.isfinite(eeg).all() or not np.isfinite(target).all():
                raise ValueError("Nonfinite paired waveform")
            if arrays["references"].shape != (2, config["window"]):
                raise ValueError("Joint HEOG/VEOG shape differs")
        paired_count += 1
    if set(recipient_folds.values()) != set(range(config["outer_folds"])):
        raise ValueError("Not all frozen outer folds are represented")
    atomic_json(output/"research_ready.json", {
        "campaign_id":config["campaign_id"], "passed":True,
        "corpus_manifest_sha256":sha256_file(parent/"corpus_manifest.jsonl"),
        "split_manifest_sha256":sha256_file(parent/"split_manifest.json"),
        "target_definition":config["target_definition"],
        "configuration_hash":canonical_hash(config), "paired_examples":paired_count,
        "recipients":len(recipient_folds), "donors":len(donor_folds),
        "verified_osf_records":len(records), "reserved_confirmation_opened":False,
        "scientific_accuracy_qualified":False, "teacher_qualified":False})
