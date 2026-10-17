"""Ensure implemented research stages reach the neural dispatcher on Kaggle."""
from pathlib import Path

import pytest

from vmd_eog import campaign, neural_campaign, runner


@pytest.mark.parametrize("stage", neural_campaign.IMPLEMENTED_STAGES)
def test_schema_two_stage_reaches_neural_dispatcher(monkeypatch, stage):
    calls = []
    monkeypatch.setattr(neural_campaign, "execute", lambda *args: calls.append(args))

    def wrong_dispatch(*args):
        pytest.fail("Implemented research stage was sent to the legacy campaign")

    monkeypatch.setattr(campaign, "execute", wrong_dispatch)
    config = {"schema_version": 2}
    experiment = {"parents": ["fixed-pilot-001", "student-pilot-001"]}
    runner.dispatch_stage(stage, Path("input"), Path("output"), config, "full", experiment)
    assert calls == [(stage, Path("input"), Path("output"), config, "full", experiment)]
