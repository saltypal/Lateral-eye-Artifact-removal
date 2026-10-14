"""Recover immutable unscored sources on Kaggle; no mixture or method fitting."""
import json
from pathlib import Path
import re
import shutil
import tempfile
import time

import numpy as np

from .artifacts import verify_parent
from .corpus import prepare_lemon, unique_parent
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl


def choose_pairs(rows, recipients, split, count=2):
    """Source-only deterministic choice; prediction/metric values are not inputs."""
    if count != 2:
        raise ValueError("Source audit is bounded to two pairs")
    records = {record["participant"]: record for record in recipients}
    if len(records) != len(recipients):
        raise ValueError("Duplicate recipient manifest identity")
    pairs, donors, folds = [], set(), set()
    for recipient in sorted({row["recipient"] for row in rows}):
        record = records.get(recipient)
        if record is None:
            raise ValueError("Missing original recipient source")
        candidates = sorted((row for row in rows if row["recipient"] == recipient),
                            key=lambda row: (row["recipe"]["recipient_start"], row["example_id"]))
        row = candidates[0]
        partition = split["partitions"]["lemon"][recipient]
        donor_partition = split["partitions"]["osf"][row["donor"]]
        if (partition.get("role") != "development" or partition != row["partition"]
                or partition != record["partition"] or partition != donor_partition):
            raise ValueError("Recipient/donor source partition differs or is reserved")
        if row["donor"] in donors or partition["fold"] in folds:
            continue
        participant = recipient.removeprefix("lemon:")
        if not re.fullmatch(r"sub-\d+", participant):
            raise ValueError("Invalid original recipient identity")
        calibration = Path(row["calibration"])
        prefix = participant + "-"
        if (len(calibration.parts) != 3 or calibration.parts[0] != "calibration"
                or calibration.name != "calibration.npz"
                or not calibration.parent.name.startswith(prefix)):
            raise ValueError("Mixed calibration identity cannot identify original donor")
        donor_record = calibration.parent.name[len(prefix):]
        if not re.fullmatch(r"study\d+_p\d+_prep", donor_record):
            raise ValueError("Invalid original donor record")
        if "osf:" + donor_record.split("_")[1] != row["donor"]:
            raise ValueError("Original donor record disagrees with participant")
        for key in ("archive_sha256", "source_header_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", record.get(key, "")):
                raise ValueError("Original source checksum missing")
        pairs.append({"recipient": recipient, "donor": row["donor"],
                      "donor_record": donor_record, "partition": partition,
                      "recipient_start": row["recipe"]["recipient_start"],
                      "legacy_example_id": row["example_id"], "source": record})
        donors.add(row["donor"])
        folds.add(partition["fold"])
        if len(pairs) == count:
            return pairs
    raise ValueError("Need two recoverable development pairs with distinct donors")


def readiness(input_root, config):
    if (config.get("scope") != "bounded_original_reference_source_audit"
            or config.get("autovmd_enabled") is not False
            or config.get("reserved_access") != "closed until verified freeze"):
        raise ValueError("Original-source audit policy differs")
    corpus = unique_parent(input_root, "corpus_summary.json").parent
    full = unique_parent(input_root, "nonfrontal_summary.json").parent
    for name, parent, required in (
            ("corpus", corpus, ("run_spec.json", "corpus_summary.json", "split_manifest.json", "recipient_manifest.jsonl")),
            ("full", full, ("run_spec.json", "nonfrontal_summary.json", "nonfrontal_selection_manifest.jsonl", "stage_tests_summary.json"))):
        expected = config["audit_parents"][name]
        if sha256_file(parent / "artifact_manifest.json") != expected["manifest_sha256"]:
            raise ValueError("Original-source audit parent manifest differs: " + name)
        verify_parent(parent, required)
        spec = json.loads((parent / "run_spec.json").read_text())
        if any(spec.get(key) != expected[key] for key in ("run_id", "campaign_id", "git_sha")):
            raise ValueError("Original-source audit parent identity differs: " + name)
    if sha256_file(corpus / "split_manifest.json") != config["source_split_sha256"]:
        raise ValueError("Original-source audit split differs")
    summary = json.loads((full / "nonfrontal_summary.json").read_text())
    if (summary.get("examples") != 320 or summary.get("profile") != "full"
            or summary.get("source_split_sha256") != config["source_split_sha256"]
            or summary.get("reserved_confirmation_opened") is not False
            or json.loads((full / "stage_tests_summary.json").read_text()).get("passed") is not True):
        raise ValueError("Original-source audit full parent contracts differ")
    rows = read_jsonl(full / "nonfrontal_selection_manifest.jsonl")
    if any(row["partition"]["role"] != "development" for row in rows):
        raise ValueError("Reserved source in audit cohort")
    pairs = choose_pairs(rows, read_jsonl(corpus / "recipient_manifest.jsonl"),
                         json.loads((corpus / "split_manifest.json").read_text()))
    return corpus, pairs


def run(input_root, output, config, profile):
    if profile != "pilot":
        raise ValueError("Original-source recovery is a bounded pilot only")
    started = time.perf_counter()
    corpus, pairs = readiness(input_root, config)
    write_jsonl(output / "source_probe_selection.jsonl", pairs)
    atomic_json(output / "split_manifest.json", json.loads((corpus / "split_manifest.json").read_text()))
    completed = []
    for index, pair in enumerate(pairs):
        print("SOURCE_PROBE_START", index + 1, len(pairs), pair["recipient"], pair["donor"], flush=True)
        directory = output / "original_sources" / pair["recipient"].removeprefix("lemon:")
        directory.mkdir(parents=True)
        donor_base = Path("calibration") / pair["donor_record"]
        required = [(donor_base / name).as_posix() for name in ("calibration.npz", "record.json")]
        verify_parent(corpus, required)
        donor_record = json.loads((corpus / donor_base / "record.json").read_text())
        axis_names = [name.upper().replace("_", "").replace("-", "") for name in donor_record["eog_names"]]
        if (donor_record["participant"] != pair["donor"] or donor_record["partition"] != pair["partition"]
                or len(axis_names) != 2 or axis_names[0] not in ("HEOG", "EOGH")
                or axis_names[1] not in ("VEOG", "EOGV")
                or not set(donor_record["calibration_trial_ids"]).isdisjoint(donor_record["scoring_trial_ids"])):
            raise ValueError("Original donor identity/axes/trial separation differs")
        donor_path = corpus / donor_base / "calibration.npz"
        donor_arrays = dict(np.load(donor_path, allow_pickle=False))
        duration = donor_arrays["eeg"].shape[-1]
        if (donor_arrays["eeg"].ndim != 2
                or donor_arrays["eeg"].shape[0] != len(donor_arrays["channel_names"])
                or donor_arrays["references"].shape != (2, duration)
                or donor_arrays["trial_types"].shape != (duration,)
                or np.any(donor_arrays["boundaries"] <= 0)
                or np.any(donor_arrays["boundaries"] >= duration)
                or np.any(np.diff(donor_arrays["boundaries"]) <= 0)
                or not all(np.isfinite(donor_arrays[key]).all() for key in ("eeg", "references"))):
            raise ValueError("Original donor calibration shape/boundary contract differs")
        shutil.copy2(donor_path, directory / "donor_calibration.npz")
        atomic_json(directory / "donor_record.json", donor_record)
        source = pair["source"]
        request = {"participant": pair["recipient"].removeprefix("lemon:"), "url": source["url"],
                   "expected_archive_sha256": source["archive_sha256"],
                   "expected_header_sha256": source["source_header_sha256"]}
        # The original prefix is reacquired, never reconstructed from mixed EEG.
        with tempfile.TemporaryDirectory(prefix="reference-source-probe-") as temporary:
            record, calibration, windows = prepare_lemon(request, temporary, config, "full")
        if (record["channel_names"] != source["channel_names"]
                or record["native_fs"] != source["native_fs"]
                or record["calibration_end_sample_native"] != source["calibration_end_sample_native"]
                or record["eog_channels"] != source["eog_channels"]
                or record["reference"] != source["reference"]
                or record["units"] != source["units"]):
            raise ValueError("Recovered original source/preprocessing metadata differs")
        if calibration.ndim != 2 or calibration.shape[0] != len(record["channel_names"]):
            raise ValueError("Recovered original recipient calibration axis differs")
        if not np.isfinite(calibration).all():
            raise ValueError("Recovered original recipient calibration is nonfinite")
        if pair["recipient_start"] not in {window["start"] for window in windows}:
            raise ValueError("Original chronological scoring window cannot be recovered")
        np.savez_compressed(directory / "recipient_calibration.npz", eeg=calibration,
                            channel_names=np.asarray(record["channel_names"], dtype="U64"))
        window_rows = []
        for number, window in enumerate(windows):
            name = f"recipient-window-{number:02d}.npz"
            np.savez_compressed(directory / name, eeg=window["eeg"])
            window_rows.append({"array": name, "start": window["start"], "ocular_rms": window["ocular_rms"]})
        record.update(partition=pair["partition"], scoring_windows=window_rows,
                      recovered_calibration="recipient_calibration.npz")
        atomic_json(directory / "recipient_record.json", record)
        completed.append({"recipient": pair["recipient"], "donor": pair["donor"],
                          "donor_record": pair["donor_record"], "archive_sha256": record["archive_sha256"],
                          "header_sha256": record["source_header_sha256"],
                          "recipient_prefix_sha256": sha256_file(directory / "recipient_calibration.npz"),
                          "original_donor_calibration_sha256": sha256_file(donor_path),
                          "recipient_units": record["units"],
                          "donor_physical_units": donor_record.get("units", "unknown; original cached numeric units"),
                          "calibration_shape": list(calibration.shape), "scoring_windows": len(windows),
                          "algebraic_unmixing": False, "reserved_source_access": False})
        write_jsonl(output / "source_probe_completed.jsonl", completed)
        atomic_json(output / "source_probe_progress.json", {"completed_pairs": len(completed), "total_pairs": len(pairs)})
        print("SOURCE_PROBE_SAVED", len(completed), len(pairs), flush=True)
    atomic_json(output / "source_probe_summary.json", {"pairs": len(completed), "source_recovery_passed": True,
        "source_split_sha256": config["source_split_sha256"], "coupling_repaired": False,
        "new_mixtures": 0, "new_vmd_decompositions": 0, "weights_trained": False,
        "reserved_confirmation_opened": False, "qualification": False,
        "target_definition": "retained low-ocular recipient EEG; native clean status unknown",
        "donor_units": "original numerical coordinates; unknown physical units stay unknown",
        "runtime_s": time.perf_counter() - started})
