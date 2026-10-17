"""Search promotion contracts, executed in the Kaggle stage test suite."""
import json
import pytest
from vmd_eog import neural_experiments


def parent(root, inner_fold, paused=False):
    folder = root / str(inner_fold)
    folder.mkdir()
    summary = {"campaign_id":"test", "reserved_confirmation_opened":False,
        "training_complete":not paused,"paused_runtime":paused,
        "snr_db":8.0,"preservation_passed":True,"runtime_s":10.0,
        "epochs_completed":5,"experiment":{"arm":"regional","outer_fold":0,
            "inner_fold":inner_fold,"K":3,"alpha":250,"seed":42,
            "learning_rate":.001,"epochs":5,"loss_profile":"mse",
            "evaluate_outer":False}}
    (folder / "training_summary.json").write_text(json.dumps(summary))


def test_paused_candidate_cannot_be_promoted(tmp_path, monkeypatch):
    monkeypatch.setattr(neural_experiments,"verify_parent",lambda *args: None)
    parent(tmp_path,0,paused=True)
    with pytest.raises(ValueError,match="Paused"):
        neural_experiments.select_search(tmp_path,tmp_path,{"campaign_id":"test"},
            {"outer_fold":0,"required_candidates":1})


def test_missing_inner_fold_cannot_be_promoted(tmp_path, monkeypatch):
    monkeypatch.setattr(neural_experiments,"verify_parent",lambda *args: None)
    parent(tmp_path,0)
    with pytest.raises(ValueError,match="incomplete"):
        neural_experiments.select_search(tmp_path,tmp_path,{"campaign_id":"test"},
            {"outer_fold":0,"required_candidates":1,"inner_folds":[0,1,2]})
