"""Refresh a lossless metadata annex; never load EEG or recompute metrics."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Work So Far.md"
MARKER = "<!-- BEGIN GENERATED EVIDENCE -->"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def main():
    narrative = OUTPUT.read_text(encoding="utf-8").split(MARKER)[0]
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    parts = [narrative, MARKER, "\n\n## Evidence annex: snapshot provenance\n\n",
             f"Metadata captured `{now}` at repository HEAD `{commit}`. Local artifact presence does not establish live remote status. Exact saved metrics below are not recalculated.\n\n"]
    # Retrieval snapshots nested below a launch directory are not new launches.
    specs = sorted((ROOT / ".kaggle-work").glob("*/run_spec.json"))
    parts.append(f"## Appendix A. All {len(specs)} recorded launch specifications\n\n")
    for path in specs:
        parts.extend([f"### {path.parent.name}\n\n[{rel(path)}]({rel(path)}); SHA256 `{digest(path)}`.\n\n```json\n", path.read_text(encoding="utf-8-sig"), "\n```\n\n"])
    parts.append("## Appendix B. Saved summaries, tests and recovery receipts\n\nByte-identical duplicate copies appear once. Historical gate labels retain their run-specific meaning. Missing outputs are unavailable evidence, not successes.\n\n")
    seen = set()
    summary_count = 0
    for path in sorted((ROOT / "results").rglob("*.json")):
        if not ("summary" in path.name or path.name in {"execution_state.json", "prediction_retrieval_verified.json", "cache_retrieval_verified.json", "artifact_manifest.json", "retrieval_receipt.json", "failure_recovery_audit.json"}):
            continue
        if path.stat().st_size > 500000:
            continue
        checksum = digest(path)
        if checksum in seen:
            continue
        seen.add(checksum)
        summary_count += 1
        parts.extend([f"### {rel(path)}\n\n[Saved payload]({rel(path)}); SHA256 `{checksum}`.\n\n```json\n", path.read_text(encoding="utf-8-sig"), "\n```\n\n"])
    files = []
    for path in sorted((ROOT / "results").rglob("*")):
        if path.is_file() and (path.suffix.lower() in {".csv", ".jsonl", ".txt"}
                               or path.name.endswith(".csv.gz")):
            files.append({"path": rel(path), "bytes": path.stat().st_size, "sha256": digest(path)})
    index = ROOT / "docs/work_so_far_evidence_index.json"
    index.write_text(json.dumps({"snapshot_utc": now, "git_sha": commit, "tables_and_logs": files}, indent=2), encoding="utf-8")
    parts.append("## Appendix C. Exhaustive local metric/source/record/log index\n\nLarge per-channel and candidate tables remain lossless in the linked files. This indexes available evidence, including partial failures; it does not assert complete remote retrieval. [Checksummed machine-readable index](docs/work_so_far_evidence_index.json).\n\n| File | Bytes | SHA256 |\n|---|---:|---|\n")
    for item in files:
        parts.append(f'| [{item["path"]}]({item["path"]}) | {item["bytes"]} | `{item["sha256"]}` |\n')
    parts.append("\n## Appendix D. Verbatim record/participant/condition tables\n\nSaved rows retain their original units, folds and aggregation. Duplicate tables appear once. Larger tables remain indexed above; missing records are not filled in.\n\n")
    tables_seen = set()
    for item in files:
        path = ROOT / item["path"]
        if path.suffix != ".csv" or item["bytes"] > 120000:
            continue
        if not any(word in path.name.lower() for word in ("record", "participant", "condition", "mean", "comparison", "native", "uncertainty", "fold")):
            continue
        if item["sha256"] in tables_seen:
            continue
        tables_seen.add(item["sha256"])
        parts.extend([f'### {item["path"]}\n\n```csv\n', path.read_text(encoding="utf-8-sig"), "\n```\n\n"])
    parts.append("## Appendix E. Detailed historical protocols, findings and repairs\n\nSnapshots preserve all original detail. Earlier future-tense plans and statuses are historical, not current conclusions. Stopped AutoVMD plans are not active authorization.\n\n")
    documents = ["FRONTAL_METHOD_GUIDE.md", "FRONTAL_PAPER_AUDIT.md", "GUARDED_REPLAY_RESULTS_20261011.md", "GUARDED_K3_ALPHA2000_REPLAY.md", "AUTOVMD_IMPLEMENTATION.md", "EVALUATION_PROTOCOL.md", "FRONTAL_RESULTS.md", "POSTERIOR_CONTEXT_PLAN.md", "REPAIRS.md", "RESEARCH_ALTERNATIVES_20261010.md", "NEURAL_PRESERVATION_DIAGNOSIS.md", "NEURAL_DIAGNOSIS_REPAIR_002.md", "SCIENTIFIC_EXPLANATION.md", "VMD_CORRECTION_DIAGNOSIS.md", "ALGORITHM_REFERENCES.md", "NATIVE_PROTOCOL_PLAN.md", "VMD_SOBI_PLAN.md", "IMPLEMENTATION_PLAN.md", "FRONTAL_REPLICATION_PLAN.md", "SIX_HOUR_DELIVERY.md"]
    documents.extend(["NONFRONTAL_CHALLENGERS_PLAN.md", "NONFRONTAL_CHALLENGERS_RESULTS.md",
                      "FRONTAL_SNR_IMPROVEMENT_PLAN.md", "CENTRAL_COVERAGE_RESULTS.md",
                      "MATCHED_CONTEXT_RESULTS.md", "FRONTAL_SHAPE_DIAGNOSTIC_RESULTS.md"])
    for name in documents:
        path = ROOT / "docs" / name
        if path.exists():
            parts.extend([f"### Snapshot: {name}\n\n[Original](docs/{name}); captured SHA256 `{digest(path)}`.\n\n", path.read_text(encoding="utf-8-sig"), "\n\n---\n\n"])
    posterior = ROOT / "posterior results.md"
    if posterior.exists():
        parts.extend(["### Read-only posterior report snapshot\n\n", posterior.read_text(encoding="utf-8-sig"), "\n"])
    parts.append("\n## Appendix F. Implementation, notebook and configuration inventory\n\nCurrent file identities below make the architecture inspectable; each launched experiment still uses its own archived SHA, not these later files.\n\n| File | Bytes | SHA256 |\n|---|---:|---|\n")
    for folder in ("src", "configs", "notebooks", "tests", "tools"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".ipynb", ".md"} and "__pycache__" not in path.parts:
                parts.append(f"| [{rel(path)}](<{rel(path)}>) | {path.stat().st_size} | `{digest(path)}` |\n")
    OUTPUT.write_text("".join(parts), encoding="utf-8")
    print(json.dumps({"bytes": OUTPUT.stat().st_size, "launch_specs": len(specs), "distinct_summary_payloads": summary_count, "indexed_tables_logs": len(files), "embedded_csv_tables": len(tables_seen)}, indent=2))


if __name__ == "__main__":
    main()
