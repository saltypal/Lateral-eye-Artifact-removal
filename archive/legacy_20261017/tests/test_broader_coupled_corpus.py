"""Reject incomplete or overlapping original-source shards before generation."""
import copy

import pytest

from vmd_eog.broader_coupled_corpus import validate_shard_union
from vmd_eog.contracts import canonical_hash


def fixture():
    pairs = [{"recipient": f"lemon:sub-{i:06d}"} for i in range(32)]
    digest = canonical_hash(pairs)
    summaries = [{"shard_index": i, "shard_count": 2, "all_sources_recovered": True, "completed_pairs": 16,
                  "cohort_sha256": digest, "reserved_confirmation_opened": False} for i in range(2)]
    return pairs, [pairs[::2], pairs[1::2]], summaries, digest


def test_union_requires_complete_disjoint_original_sources():
    pairs, shards, summaries, digest = fixture()
    assert validate_shard_union(shards, summaries, pairs, digest) == pairs
    broken = copy.deepcopy(summaries)
    broken[1]["all_sources_recovered"] = False
    with pytest.raises(ValueError, match="incomplete"):
        validate_shard_union(shards, broken, pairs, digest)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_shard_union([shards[0], shards[0]], summaries, pairs, digest)
    wrong = copy.deepcopy(pairs)
    wrong[0]["recipient"] = "lemon:sub-999999"
    with pytest.raises(ValueError, match="frozen cohort"):
        validate_shard_union(shards, summaries, wrong, digest)
