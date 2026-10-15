"""Repaired coupled development corpus from completely recovered frozen shards."""
import json
from pathlib import Path
import time

from .artifacts import verify_parent
from .contracts import canonical_hash
from .coupled_pilot import build_pair
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl


def declared_parent(input_root, summary_name, expected):
    candidates = []
    for summary in Path(input_root).rglob(summary_name):
        spec_path = summary.parent / "run_spec.json"
        if spec_path.is_file() and json.loads(spec_path.read_text())["run_id"] == expected["run_id"]:
            candidates.append(summary.parent)
    if len(candidates) != 1:
        raise ValueError("Missing or ambiguous declared parent: " + expected["run_id"])
    parent = candidates[0]
    if sha256_file(parent / "artifact_manifest.json") != expected["manifest_sha256"]:
        raise ValueError("Declared parent manifest differs")
    verify_parent(parent, ("run_spec.json", summary_name))
    spec = json.loads((parent / "run_spec.json").read_text())
    if any(spec.get(key) != expected[key] for key in ("run_id", "campaign_id", "git_sha")):
        raise ValueError("Declared parent source identity differs")
    return parent


def validate_shard_union(selections, summaries, expected_pairs, cohort_hash):
    if (len(selections) != 2 or len(summaries) != 2
            or {summary["shard_index"] for summary in summaries} != {0, 1}):
        raise ValueError("Need both original recovery shards")
    combined = []
    for rows, summary in zip(selections, summaries):
        if (summary.get("all_sources_recovered") is not True
                or summary.get("completed_pairs") != 16 or len(rows) != 16
                or summary.get("shard_count") != 2
                or summary.get("cohort_sha256") != cohort_hash
                or summary.get("reserved_confirmation_opened") is not False):
            raise ValueError("Recovery shard is incomplete or has a different cohort")
        combined.extend(rows)
    if len({row["recipient"] for row in combined}) != 32:
        raise ValueError("Duplicate or missing recovered recipient")
    combined.sort(key=lambda row: row["recipient"])
    if combined != expected_pairs or canonical_hash(combined) != cohort_hash:
        raise ValueError("Recovered shard union differs from the frozen cohort")
    return combined


def readiness(input_root, config):
    if (config.get("scope") != "broader_homogeneous_coupled_corpus"
            or config.get("autovmd_enabled") is not False
            or config.get("reserved_access") != "closed until verified freeze"):
        raise ValueError("Broader coupled corpus policy differs")
    corpus = declared_parent(input_root, "corpus_summary.json", config["producer_parents"]["corpus"])
    verify_parent(corpus, ("split_manifest.json",))
    if sha256_file(corpus / "split_manifest.json") != config["source_split_sha256"]:
        raise ValueError("Broader coupled source split differs")
    selections, summaries, locations = [], [], {}
    for index in (0, 1):
        source = declared_parent(input_root, "broader_source_summary.json", config["producer_parents"][f"shard{index}"])
        verify_parent(source, ("source_selection.jsonl", "source_recovery_completed.jsonl", "split_manifest.json"))
        if sha256_file(source / "split_manifest.json") != config["source_split_sha256"]:
            raise ValueError("Recovered shard split differs")
        rows = read_jsonl(source / "source_selection.jsonl")
        completed = read_jsonl(source / "source_recovery_completed.jsonl")
        if (len(completed) != len(rows)
                or len({row["recipient"] for row in completed}) != len(completed)
                or {row["recipient"] for row in completed} != {row["recipient"] for row in rows}):
            raise ValueError("Recovered completion identities differ from source selection")
        selections.append(rows)
        summaries.append(json.loads((source / "broader_source_summary.json").read_text()))
        locations.update({row["recipient"]: source for row in rows})
    root = Path(__file__).resolve().parents[2]
    cohort_file = (root / config["cohort_path"]).resolve()
    if not cohort_file.is_relative_to((root / "data_provenance").resolve()):
        raise ValueError("Cohort register path escapes provenance")
    expected = json.loads(cohort_file.read_text())["pairs"]
    pairs = validate_shard_union(selections, summaries, expected, config["cohort_sha256"])
    split = json.loads((corpus / "split_manifest.json").read_text())
    for pair in pairs:
        if (pair["partition"]["role"] != "development"
                or split["partitions"]["lemon"][pair["recipient"]] != pair["partition"]
                or split["partitions"]["osf"][pair["donor"]] != pair["partition"]):
            raise ValueError("Broader corpus recipient/donor partition differs")
    return corpus, pairs, locations, split


def run(input_root, output, config, profile):
    if profile != "full":
        raise ValueError("Broader coupled corpus requires the declared full cohort")
    corpus, pairs, locations, split = readiness(input_root, config)
    started = time.perf_counter()
    write_jsonl(output / "source_selection.jsonl", pairs)
    atomic_json(output / "split_manifest.json", split)
    (output / "arrays").mkdir()
    rows, diagnostics = [], []
    for pair in pairs:
        source = locations[pair["recipient"]]
        source_run = json.loads((source / "run_spec.json").read_text())["run_id"]
        pair_rows, diagnostic = build_pair(source, corpus, pair, output, config)
        if len(pair_rows) != 10:
            raise ValueError("Repaired source pair does not contain ten declared conditions")
        for row in pair_rows:
            row["pilot_recipient_exposed"] = row["recipient"] in config["pilot_recipient_ids"]
            row["pilot_donor_exposed"] = row["donor"] in config["pilot_donor_ids"]
            row["source_recovery_run"] = source_run
        rows.extend(pair_rows)
        diagnostics.append(diagnostic)
        write_jsonl(output / "corpus_manifest.jsonl", rows)
        atomic_json(output / "producer_diagnostics.json", diagnostics)
        atomic_json(output / "broader_corpus_progress.json", {"completed_pairs": len(diagnostics), "examples": len(rows)})
        print("BROADER_COUPLED_PAIR_SAVED", len(diagnostics), len(pairs), pair["recipient"], flush=True)
    if len(rows) != 320 or len({row["example_id"] for row in rows}) != 320:
        raise ValueError("Broader repaired corpus requires exactly 320 unique examples")
    atomic_json(output / "broader_corpus_summary.json", {"examples": len(rows), "pairs": len(pairs),
                "donors": len({pair["donor"] for pair in pairs}), "cohort_sha256": config["cohort_sha256"],
                "source_split_sha256": config["source_split_sha256"], "construction_contracts_passed": True,
                "new_vmd_decompositions": 0, "teacher_qualified": False, "reserved_confirmation_opened": False,
                "regression_generated_bias_remains": True, "native_clean_status": "unknown low-ocular reference",
                "runtime_s": time.perf_counter() - started})
