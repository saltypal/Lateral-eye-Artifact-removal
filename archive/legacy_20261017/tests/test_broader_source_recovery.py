"""Source-only cohort contracts; numerical suite runs on Kaggle."""
import copy

import pytest

from vmd_eog.broader_source_recovery import select_shard
from vmd_eog.reference_coupling_audit import development_pairs
from test_reference_coupling_audit import metadata_fixture


def test_development_selection_retains_shared_donors_without_changing_folds():
    rows, recipients, split = metadata_fixture()
    duplicate = copy.deepcopy(rows[:2])
    for row in duplicate:
        row["recipient"] = "lemon:sub-000009"
        row["calibration"] = "calibration/sub-000009-study01_p0_prep/calibration.npz"
    record = {**recipients[0], "participant": "lemon:sub-000009"}
    split["partitions"]["lemon"][record["participant"]] = record["partition"]
    selected = list(development_pairs(rows + duplicate, recipients + [record], split))
    assert len(selected) == 3
    assert len({pair["donor"] for pair in selected}) == 2
    assert selected == list(development_pairs((rows + duplicate)[::-1], (recipients + [record])[::-1], split))


def test_two_shards_are_disjoint_complete_and_order_invariant():
    pairs = [{"recipient": f"lemon:sub-{i:06d}"} for i in range(32)]
    first, second = select_shard(pairs, 0), select_shard(pairs, 1)
    assert len(first) == len(second) == 16
    assert {p["recipient"] for p in first}.isdisjoint({p["recipient"] for p in second})
    assert sorted(first + second, key=lambda pair: pair["recipient"]) == pairs
    assert first == select_shard(pairs[::-1], 0)
    with pytest.raises(ValueError, match="Duplicate"):
        select_shard(pairs + pairs[:1], 0)
    with pytest.raises(ValueError, match="two declared"):
        select_shard(pairs, 2)
