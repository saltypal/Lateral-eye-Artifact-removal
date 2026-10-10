import pytest
from vmd_eog.autovmd import AutoVMD,balanced_rows


def row(identity,recipient,donor,level=0,condition="blink"):
    return {"example_id":identity,"recipient":recipient,"donor":donor,"condition":condition,
            "input_snr_db":level,"target_kind":"controlled_recipient_reference",
            "partition":{"role":"development","fold":0}}


def test_selection_rejects_cross_role_identity_leakage():
    examples = [row("train","person-a","person-b"),row("validation","person-b","person-c")]
    with pytest.raises(ValueError,match="overlap"):
        AutoVMD({}).select(["train"],["validation"],{"rows":examples})


def test_balanced_cache_covers_all_levels_without_target_ranking():
    examples = [row(str(index),"a","b",level) for index,level in enumerate((-5,0,5))]
    examples += [row("duplicate","a","b",0)]
    assert len(balanced_rows(examples)) == 3
    assert {record["input_snr_db"] for record in balanced_rows(examples)} == {-5,0,5}


def test_confirmation_cannot_enter_parameter_selection():
    examples = [row("train","a","b"),row("validation","c","d")]
    examples[1]["partition"]["role"] = "confirmation"
    with pytest.raises(ValueError,match="confirmation"):
        AutoVMD({}).select(["train"],["validation"],{"rows":examples})
