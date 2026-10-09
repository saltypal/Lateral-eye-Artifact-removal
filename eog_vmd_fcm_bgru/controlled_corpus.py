"""Metadata-first controlled corpus recipes; waveform materialisation stays explicit."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping
import json
import numpy as np

from .campaign_contracts import atomic_json, canonical_hash


def fit_corpus(recordings: list[Mapping[str, Any]], split: Mapping[str, Any], output: Path,
               config: Mapping[str, Any], parents: Mapping[str, Path] | None = None) -> dict[str, Any]:
    """Select eligible OSF donors/recipients without claiming native clean EEG."""
    assignment = split["assignment"]
    donors, recipients, exclusions = [], [], []
    for row in recordings:
        if row.get("dataset") != "osf":
            continue
        if row.get("eog", {}).get("heog") is not None and row.get("eog", {}).get("veog") is not None:
            donors.append(row["record_id"])
        else:
            exclusions.append({"record_id": row["record_id"], "reason": "missing_verified_heog_or_veog"})
        # A low-ocular candidate is still a native-clean-unknown reference.
        if row.get("eligibility", {}).get("low_ocular_candidate", False):
            recipients.append(row["record_id"])
    if not donors or not recipients:
        raise ValueError("Controlled corpus requires verified OSF HEOG/VEOG donors and declared low-ocular recipients")
    fit = {"schema_version": 2, "kind": "corpus_fit_manifest", "recording_manifest_hash": canonical_hash(recordings),
           "split_hash": split["split_hash"], "generator": "empirical HEOG/VEOG-associated fields",
           "native_clean_status": "unknown; controlled targets recover the selected recipient reference, not biological ground truth",
           "limitations": ["Empirical fields may contain eye-locked neural activity.",
                           "No unverified Klados anatomy or participant identity enters regional claims.",
                           "Recipe generation does not establish real-data denoising performance."],
           "donor_ids": sorted(donors), "recipient_ids": sorted(recipients), "assignment": assignment,
           "requested_rank2_shortcut_comparison": bool(config.get("rank2_shortcut_comparison", True)),
           "parents": {name: str(path) for name, path in (parents or {}).items()}}
    fit["fit_hash"] = canonical_hash(fit)
    atomic_json(Path(output) / "corpus_fit_manifest.json", fit)
    return fit


def build_corpus(recordings: list[Mapping[str, Any]], split: Mapping[str, Any], fit: Mapping[str, Any],
                 output: Path, config: Mapping[str, Any], parents: Mapping[str, Path] | None = None) -> dict[str, Any]:
    """Produce reproducible, split-safe mixture recipes. No waveform is synthesized here."""
    by_id = {row["record_id"]: row for row in recordings}
    examples = []
    recipe_count = int(config.get("recipes_per_recipient", 2))
    if recipe_count <= 0:
        raise ValueError("recipes_per_recipient must be positive")
    for recipient_id in fit["recipient_ids"]:
        recipient = by_id[recipient_id]
        target_partition = split["assignment"][recipient_id]
        compatible_donors = [donor_id for donor_id in fit["donor_ids"]
                             if split["assignment"][donor_id] == target_partition and donor_id != recipient_id]
        if not compatible_donors:
            continue
        for index in range(recipe_count):
            donor_id = compatible_donors[index % len(compatible_donors)]
            donor = by_id[donor_id]
            recipe = {"field_source": "empirical_heog_veog_associated", "mixing": "estimated_on_fit_partition_only",
                      "gain_seed": int(config.get("seed", 42)) + index,
                      "lag_samples": 0, "rank2_shortcut_control": bool(fit["requested_rank2_shortcut_comparison"]),
                      "recipient_fs": recipient.get("fs_native"), "donor_fs": donor.get("fs_native")}
            examples.append({"example_id": f"mix:{recipient_id}:{donor_id}:{index}",
                             "record_id": recipient_id, "participant_id": recipient.get("participant_id"),
                             "recipient_id": recipient_id, "donor_id": donor_id,
                             "split_role": target_partition, "array_path": None,
                             "target_kind": "controlled_recipient_reference",
                             "native_clean_status": "unknown", "montage_contract": "recipient montage only; no invented channels",
                             "channels": recipient.get("channels", []), "recipe": recipe,
                             "recipe_hash": canonical_hash(recipe),
                             "limitations": fit["limitations"]})
    if not examples:
        raise ValueError("No split-compatible donor/recipient pairs; refusing cross-partition corpus")
    corpus_hash = canonical_hash(examples)
    path = Path(output) / "corpus_manifest.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in examples), encoding="utf-8")
    summary = {"schema_version": 2, "kind": "corpus_build_summary", "examples": len(examples),
               "corpus_hash": corpus_hash, "split_hash": split["split_hash"], "fit_hash": fit["fit_hash"],
               "materialized_waveforms": False,
               "warning": "Only recipes were built. A later Kaggle waveform materialiser must record fitted maps, alignment, units and field hashes.",
               "parents": {name: str(path) for name, path in (parents or {}).items()}}
    atomic_json(Path(output) / "corpus_build_summary.json", summary)
    return summary


_NPZ_KEYS = {"eeg", "mask", "regions", "hemispheres", "coordinates", "coordinate_mask"}


def iter_materialized_examples(manifest_path: Path):
    """Yield validated training entries; recipe-only entries are intentionally unusable."""
    manifest_path = Path(manifest_path)
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if not entry.get("array_path"):
            raise ValueError("Corpus is recipe-only; waveform materialization is required before training")
        array_path = (manifest_path.parent / entry["array_path"]).resolve()
        if not array_path.is_relative_to(manifest_path.parent.resolve()) or not array_path.is_file():
            raise ValueError("Invalid corpus array_path")
        with np.load(array_path, allow_pickle=False) as arrays:
            missing = _NPZ_KEYS.difference(arrays.files)
            if missing:
                raise ValueError("Corpus NPZ missing fields: " + ", ".join(sorted(missing)))
            eeg = arrays["eeg"]
            if eeg.ndim != 2 or arrays["mask"].shape != eeg.shape[:1]:
                raise ValueError("Corpus EEG/mask shape mismatch")
            if "paired_reference" in arrays and arrays["paired_reference"].shape != eeg.shape:
                raise ValueError("paired_reference shape mismatch")
        yield entry
