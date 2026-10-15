"""Author immutable-parent configurations from fully recovered Kaggle outputs.

Metadata and byte inspection only: this tool never loads EEG or runs metrics.
Commit/push the generated configuration before using kaggle_campaign submit.
"""
import argparse
import hashlib
import json
from pathlib import Path

from build_notebooks import build


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def recovered(run_id, summary_name):
    submitted = read(ROOT / ".kaggle-work" / run_id / "run_spec.json")
    destination = ROOT / "results" / submitted["campaign_id"] / run_id
    receipt = read(destination / "retrieval_receipt.json")
    if (receipt.get("all_manifest_payloads_verified") is not True
            or receipt.get("completion_verified") is not True):
        raise ValueError("Need complete, fully recovered parent: " + run_id)
    paths = list(destination.rglob(summary_name))
    if len(paths) != 1:
        raise ValueError("Missing or ambiguous completed summary: " + run_id)
    parent = paths[0].parent
    spec = read(parent / "run_spec.json")
    if any(spec.get(key) != submitted.get(key) for key in ("run_id", "campaign_id", "git_sha", "stage")):
        raise ValueError("Recovered source identity differs: " + run_id)
    manifest = read(parent / "artifact_manifest.json")
    if set(receipt["verified_payloads"]) != set(manifest):
        raise ValueError("Receipt coverage differs from manifest")
    for relative, expected in manifest.items():
        path = (parent / relative).resolve()
        if not path.is_relative_to(parent.resolve()) or digest(path) != expected:
            raise ValueError("Recovered payload changed: " + relative)
    if read(parent / "execution_state.json").get("status") != "complete":
        raise ValueError("Recovered execution is incomplete")
    pin = {key: spec[key] for key in ("run_id", "campaign_id", "git_sha")}
    pin["manifest_sha256"] = digest(parent / "artifact_manifest.json")
    kernel = submitted["kernel"] + "/" + str(submitted["kernel_version"])
    return parent, read(paths[0]), pin, kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("corpus", "review"))
    parser.add_argument("--contracts-run", default="broader-module-contracts-001")
    parser.add_argument("--shard0", default="broader-sources-shard0-001")
    parser.add_argument("--shard1", default="broader-sources-shard1-001")
    parser.add_argument("--corpus-run", default="broader-coupled-corpus-001")
    args = parser.parse_args()
    _, contracts, _, _ = recovered(args.contracts_run, "contracts_summary.json")
    if contracts.get("passed") is not True:
        raise ValueError("Broader implementation contracts did not pass")
    source = read(ROOT / "configs/broader_source_recovery.json")
    if args.stage == "corpus":
        config = read(ROOT / "configs/coupled_pilot.json")
        pins, kernels = {}, ["satyapaladugu/vmd-eog-corpus/2"]
        for index, run_id in enumerate((args.shard0, args.shard1)):
            _, summary, pin, kernel = recovered(run_id, "broader_source_summary.json")
            if (summary.get("all_sources_recovered") is not True or summary.get("completed_pairs") != 16
                    or summary.get("shard_index") != index or summary.get("shard_count") != 2
                    or summary.get("cohort_sha256") != source["cohort_sha256"]
                    or summary.get("source_split_sha256") != source["source_split_sha256"]
                    or summary.get("reserved_confirmation_opened") is not False):
                raise ValueError("Source shard is not eligible: " + run_id)
            pins[f"shard{index}"] = pin
            kernels.append(kernel)
        config.update(campaign_id="vmd-broader-corpus-20261015", scope="broader_homogeneous_coupled_corpus")
        config["producer_parents"] = {"corpus": source["source_parents"]["corpus"], **pins}
        config["imported_parents"] = {pin["run_id"]: pin["manifest_sha256"] for pin in config["producer_parents"].values()}
        config["pilot_recipient_ids"] = ["lemon:sub-032305", "lemon:sub-032306"]
        config["pilot_donor_ids"] = ["osf:p10", "osf:p02"]
        stage, number = "broader-coupled-corpus", "73_Broader_Coupled_Corpus"
    else:
        _, summary, pin, kernel = recovered(args.corpus_run, "broader_corpus_summary.json")
        if (summary.get("examples") != 320 or summary.get("pairs") != 32
                or summary.get("construction_contracts_passed") is not True
                or summary.get("reserved_confirmation_opened") is not False
                or summary.get("cohort_sha256") != source["cohort_sha256"]):
            raise ValueError("Broader corpus is not eligible")
        config = read(ROOT / "configs/coupled_frontal_review.json")
        config.update(campaign_id="vmd-broader-review-20261015", scope="broader_fixed_anchored_frontal_validation",
                      review_parent=pin, imported_parents={pin["run_id"]: pin["manifest_sha256"]})
        kernels = [kernel]
        stage, number = "broader-frontal-review", "74_Broader_Fixed_Frontal_Review"
    config["cohort_path"], config["cohort_sha256"] = source["cohort_path"], source["cohort_sha256"]
    config["anchored_vmd_weight"] = .25
    config["model_authorization"] = "Frozen classical development validation only; no models or confirmation"
    filename = "configs/" + stage.replace("-", "_") + ".json"
    (ROOT / filename).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    spec = {"stage": stage, "run_id": stage + "-manual", "git_sha": "", "campaign_id": config["campaign_id"],
            "profile": "full", "config_path": filename, "kernel_sources": kernels, "dataset_sources": []}
    build(ROOT / "notebooks" / (number + ".ipynb"), stage, spec)
    print(json.dumps({"stage": stage, "config": filename, "versioned_kernel_sources": kernels}, indent=2))


if __name__ == "__main__":
    main()
