"""Aggregation must preserve invalid channels and independent source units."""
import json
import numpy as np
import pandas as pd
import pytest
from vmd_eog.paper_report import example_means, comparison_family
from vmd_eog.contracts import require_approval


def test_channel_first_mean_keeps_undefined_channel_in_eligibility():
    frame = pd.DataFrame({"example": ["a", "a", "b", "b"],
                          "snr": [0., 20., 0., np.nan]})
    result = example_means(frame, ["example"], ["snr"]).set_index("example")
    assert result.loc["a", "snr"] == 10.
    assert np.isnan(result.loc["b", "snr"])
    assert result.loc["b", "snr_nonfinite_channels"] == 1


def test_incomplete_participant_comparison_is_not_silently_restricted():
    frame = pd.DataFrame({"condition": ["blink"]*4,
        "source": ["p01", "p02", "p01", "p02"],
        "method": ["identity", "identity", "direct", "direct"],
        "cc": [.5, .6, .9, np.nan]})
    result = comparison_family(frame, "source", ["condition"], ["cc"])
    assert len(result) == 2
    assert (result.participant_count == 2).all()
    assert (result.finite_matched_participants == 1).all()
    assert result.status.str.startswith("unavailable").all()


def test_old_project_only_gate_cannot_authorize_models(tmp_path):
    gate = {"campaign_id": "fixture", "passed": True}
    (tmp_path/"classical_gate.json").write_text(json.dumps(gate))
    with pytest.raises(RuntimeError, match="Prescribed paper evaluation is incomplete"):
        require_approval("student-paired", tmp_path, {"campaign_id": "fixture"})
