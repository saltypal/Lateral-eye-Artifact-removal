"""Source-only original calibration recovery for the fixed broader cohort."""
import json
import shutil
import time

from .artifacts import verify_parent
from .contracts import canonical_hash
from .corpus import unique_parent
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .reference_coupling_audit import development_pairs, recover_original_pair


def select_shard(pairs, shard_index, shard_count=2):
    if (shard_count != 2 or not isinstance(shard_index, int)
            or shard_index not in (0, 1)):
        raise ValueError("Broader recovery uses exactly two declared shards")
    ordered = sorted(pairs, key=lambda pair: pair["recipient"])
    if len({pair["recipient"] for pair in ordered}) != len(ordered):
        raise ValueError("Duplicate recipient in broader source cohort")
    return ordered[shard_index::shard_count]


def readiness(input_root, config, experiment=None):
    if (config.get("scope") != "broader_original_source_recovery"
            or config.get("autovmd_enabled") is not False
            or config.get("reserved_access") != "closed until verified freeze"
            or config.get("anchored_vmd_weight") != .25
            or config.get("fixed_recipes") != [{"K": 3, "alpha": 2000}]):
        raise ValueError("Broader source recovery policy differs")
    parents = {}
    for name, summary_name in (("corpus", "corpus_summary.json"),
                               ("selection", "nonfrontal_summary.json"),
                               ("probe", "source_probe_summary.json")):
        parent = unique_parent(input_root, summary_name).parent
        expected = config["source_parents"][name]
        if sha256_file(parent / "artifact_manifest.json") != expected["manifest_sha256"]:
            raise ValueError("Broader recovery parent manifest differs: " + name)
        verify_parent(parent, ("run_spec.json", summary_name))
        spec = json.loads((parent / "run_spec.json").read_text())
        if any(spec.get(key) != expected[key] for key in ("run_id", "campaign_id", "git_sha")):
            raise ValueError("Broader recovery parent identity differs: " + name)
        if json.loads((parent / summary_name).read_text()).get("reserved_confirmation_opened") is not False:
            raise ValueError("Broader recovery parent opened reserved sources")
        parents[name] = parent
    corpus, selection = parents["corpus"], parents["selection"]
    verify_parent(corpus, ("recipient_manifest.jsonl", "split_manifest.json"))
    verify_parent(selection, ("nonfrontal_selection_manifest.jsonl", "stage_tests_summary.json"))
    if (sha256_file(corpus / "split_manifest.json") != config["source_split_sha256"]
            or json.loads((selection / "stage_tests_summary.json").read_text()).get("passed") is not True):
        raise ValueError("Broader source split or parent tests differ")
    rows = read_jsonl(selection / "nonfrontal_selection_manifest.jsonl")
    if len(rows) != 320 or any(row["partition"]["role"] != "development" for row in rows):
        raise ValueError("Unexpected or reserved broader selection rows")
    split = json.loads((corpus / "split_manifest.json").read_text())
    pairs = list(development_pairs(rows, read_jsonl(corpus / "recipient_manifest.jsonl"), split))
    if (len(pairs) != 32 or len({pair["donor"] for pair in pairs}) != 5
            or len({pair["partition"]["fold"] for pair in pairs}) != 5
            or canonical_hash(pairs) != config["cohort_sha256"]):
        raise ValueError("Frozen broader source cohort differs")
    expected = json.loads(config_path(config).read_text())
    if expected["pairs"] != pairs:
        raise ValueError("Source cohort register differs from published selection")
    if experiment is None:
        return parents, pairs, [], split
    shard = select_shard(pairs, experiment.get("shard_index"))
    if len(shard) != 16:
        raise ValueError("Recovery shard must contain sixteen recipients")
    return parents, pairs, shard, split


def config_path(config):
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    path = (root / config["cohort_path"]).resolve()
    if not path.is_relative_to((root / "data_provenance").resolve()):
        raise ValueError("Cohort register escapes data provenance directory")
    return path


def reuse_verified_pair(probe, pair, output):
    verify_parent(probe, ("source_probe_selection.jsonl", "source_probe_completed.jsonl"))
    selected = {row["recipient"]: row for row in read_jsonl(probe / "source_probe_selection.jsonl")}
    if pair["recipient"] not in selected:
        return None
    if selected[pair["recipient"]] != pair:
        raise ValueError("Recovered pilot source identity differs from broader cohort")
    prefix = "original_sources/" + pair["recipient"].removeprefix("lemon:") + "/"
    manifest = json.loads((probe / "artifact_manifest.json").read_text())
    required = tuple(name for name in manifest if name.startswith(prefix))
    if not required:
        raise ValueError("Recovered pilot source has no verified payloads")
    verify_parent(probe, required)
    shutil.copytree(probe / prefix, output / prefix)
    completed = [row for row in read_jsonl(probe / "source_probe_completed.jsonl")
                 if row["recipient"] == pair["recipient"]]
    if len(completed) != 1:
        raise ValueError("Recovered pilot completion identity is ambiguous")
    return {**completed[0], "recovery_origin": "verified_two_pair_source_parent"}


def run(input_root, output, config, profile, experiment):
    if profile != "full":
        raise ValueError("Broader original-source recovery requires the declared full cohort")
    parents, pairs, shard, split = readiness(input_root, config, experiment)
    started = time.perf_counter()
    write_jsonl(output / "global_source_selection.jsonl", pairs)
    write_jsonl(output / "source_selection.jsonl", shard)
    atomic_json(output / "split_manifest.json", split)
    completed, failures = [], []
    for index, pair in enumerate(shard):
        print("BROADER_SOURCE_START", index + 1, len(shard), pair["recipient"], flush=True)
        try:
            record = reuse_verified_pair(parents["probe"], pair, output)
            if record is None:
                record = recover_original_pair(parents["corpus"], pair, output, config)
                record["recovery_origin"] = "original_archive_verified_against_published_hashes"
            completed.append(record)
            print("BROADER_SOURCE_SAVED", len(completed), len(shard), pair["recipient"], flush=True)
        except Exception as error:
            failure = {"recipient": pair["recipient"], "donor": pair["donor"],
                       "error": type(error).__name__ + ": " + str(error)}
            failures.append(failure)
            print("BROADER_SOURCE_UNAVAILABLE", failure, flush=True)
        write_jsonl(output / "source_recovery_completed.jsonl", completed)
        atomic_json(output / "source_recovery_failures.json", failures)
        atomic_json(output / "source_recovery_progress.json", {"completed": len(completed),
                    "failed": len(failures), "requested": len(shard)})
    atomic_json(output / "broader_source_summary.json", {"requested_pairs": len(shard),
                "completed_pairs": len(completed), "all_sources_recovered": not failures and len(completed) == len(shard),
                "failures": failures, "shard_index": experiment["shard_index"], "shard_count": 2,
                "global_pairs": len(pairs), "cohort_sha256": config["cohort_sha256"],
                "source_split_sha256": config["source_split_sha256"], "new_mixtures": 0,
                "new_vmd_decompositions": 0, "teacher_qualified": False,
                "reserved_confirmation_opened": False, "runtime_s": time.perf_counter() - started,
                "availability_audit_only": True,
                "historical_exposure": "development; not independent confirmation"})
