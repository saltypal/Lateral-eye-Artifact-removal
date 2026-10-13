"""Documentary source eligibility; no EEG loading or numerical denoising."""
import csv
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from .io import atomic_json


REQUIREMENTS = (
    "source_access", "frontal_mapping", "paired_target", "independent_generator",
    "signed_individual_fields", "units_reference_compatibility", "eog_polarity",
    "amplitude_coupling", "calibration_scoring_separation", "source_grouping", "selection_access",
)
STATES = frozenset(("verified", "unresolved", "absent", "closed"))
NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def assess_candidate(candidate):
    """A completed documentary assessment is distinct from benchmark readiness."""
    if any(not isinstance(candidate.get(key), str) or not candidate[key].strip()
           for key in ("id", "source", "role", "limitations")):
        raise ValueError("Candidate identity, evidence, role and limitations are required")
    requirements = candidate.get("requirements", {})
    if not isinstance(requirements, dict) or set(requirements) != set(REQUIREMENTS):
        raise ValueError("Candidate must report exactly every benchmark requirement")
    if any(not isinstance(value, str) or value not in STATES for value in requirements.values()):
        raise ValueError("Requirement states must be verified/unresolved/absent/closed")
    blockers = [{"requirement": key, "status": requirements[key]} for key in REQUIREMENTS
                if requirements[key] != "verified"]
    return {**candidate, "eligible": not blockers, "blocking_requirements": blockers,
            "assessment_kind": "documentary contract; not an empirical cleaning metric"}


def extract_sensitivity_table(document, expected_sha256):
    """Keep original text/grouped labels; never derive channel signs or fields."""
    if hashlib.sha256(document).hexdigest() != expected_sha256:
        raise ValueError("Published supplement checksum differs")
    with zipfile.ZipFile(io.BytesIO(document)) as archive:
        member = archive.getinfo("word/document.xml")
        if member.file_size > 2_000_000:
            raise ValueError("Unexpected supplement XML size")
        tree = ElementTree.fromstring(archive.read(member))
    matches = []
    for table in tree.findall(".//w:tbl", NAMESPACE):
        rows = []
        for row in table.findall("w:tr", NAMESPACE):
            cells = []
            for cell in row.findall("w:tc", NAMESPACE):
                cells.append(" ".join("".join(cell.itertext()).split()))
            rows.append(cells)
        if rows and "Blink (% of VEOG)" in " ".join(rows[0]):
            matches.append(rows)
    if len(matches) != 1:
        raise ValueError("Expected exactly one source sensitivity table")
    rows = matches[0]
    if len(rows) < 3 or any(len(row) != 7 for row in rows[2:]):
        raise ValueError("Published sensitivity table structure differs")
    if len({row[0] for row in rows[2:]}) != len(rows[2:]):
        raise ValueError("Duplicate source sensitivity row")
    return {"header_rows": rows[:2], "data_rows": rows[2:],
            "interpretation": "source sensitivity text; no signed individual channel map inferred"}


def load_supplement(spec, root):
    """Read exact reviewed source bytes from the pinned checkout, without HTTP."""
    root = Path(root).resolve()
    path = (root / spec["vendored_path"]).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != ".docx":
        raise ValueError("Documentary source must be a DOCX inside the pinned checkout")
    if path.stat().st_size > 2_000_000:
        raise ValueError("Unexpected documentary source size")
    document = path.read_bytes()
    digest = hashlib.sha256(document).hexdigest()
    if digest != spec["sha256"]:
        raise ValueError("Vendored published supplement checksum differs")
    return document, {"original_archive_url": spec["archive_url"],
        "article_url": spec["article_url"], "vendored_path": spec["vendored_path"],
        "document_sha256": digest, "document_bytes": len(document),
        "source_resolution": "exact reviewed source packaged in published Git commit",
        "live_http_attempted": False, "waveforms_loaded": False}


def run(output):
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    registry_path = root / "configs/frontal_benchmark_sources.json"
    registry = json.loads(registry_path.read_text())
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark documentary register")
    ids = [candidate["id"] for candidate in registry["candidates"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Unique nonempty source candidate identities required")
    assessments = [assess_candidate(candidate) for candidate in registry["candidates"]]
    document, download = load_supplement(registry["supplement"], root)
    table = extract_sensitivity_table(document, registry["supplement"]["sha256"])
    if [row[0] for row in table["data_rows"]] != registry["supplement"]["table_row_labels"]:
        raise ValueError("Published source row labels differ")
    (output / "published_supplement.docx").write_bytes(document)
    atomic_json(output / "published_sensitivity_table.json", table)
    atomic_json(output / "supplement_download_identity.json", download)
    atomic_json(output / "benchmark_source_registry.json", registry)
    atomic_json(output / "benchmark_source_assessments.json", assessments)
    fields = ["id", "role", *REQUIREMENTS, "eligible", "source", "limitations"]
    with (output / "benchmark_eligibility.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for item in assessments:
            writer.writerow({key: item["requirements"].get(key) if key in REQUIREMENTS else item[key] for key in fields})
    eligible = [item["id"] for item in assessments if item["eligible"]]
    atomic_json(output / "benchmark_eligibility_summary.json", {
        "audit_completed": True, "assessment_kind": "documentary eligibility, not new waveform evidence",
        "candidates": len(assessments), "eligible_candidates": eligible, "benchmark_ready": bool(eligible),
        "supplement_sha256": download["document_sha256"], "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "source_table_groups": [row[0] for row in table["data_rows"]],
        "missing_signs_inferred": False, "recipient_reference_reconciled": False,
        "paired_benchmark_launched": False, "waveforms_loaded": False, "decompositions_computed": 0,
        "autovmd_executed": False, "weights_trained": False, "parameters_fitted": False,
        "reserved_confirmation_opened": False, "qualification": False, "runtime_s": time.perf_counter() - started,
        "next_checkpoint": "IB02 remains closed until source/generator contracts pass"})
