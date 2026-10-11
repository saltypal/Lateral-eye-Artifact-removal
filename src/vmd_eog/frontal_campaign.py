"""Policy for the fixed-recipe, frontal-only campaign; no numerical imports."""
import json
from pathlib import Path

from .artifacts import verify_parent
from .contracts import canonical_hash


STAGES = ("frontal-fixture", "frontal-comparison", "frontal-native", "frontal-evidence", "frontal-evaluation", "frontal-cache-audit", "frontal-frozen-transfer", "frontal-gain-hybrid", "frontal-hybrid-evidence", "frontal-gain-unit-audit", "frontal-calibration-hybrid", "frontal-native-calibration", "frontal-axis-calibration", "frontal-event-calibration", "frontal-native-events", "frontal-event-gain-calibration", "frontal-native-gain-events")
FIXED_SETTINGS = ((3, 2000), (4, 2000))


def validate_policy(config, stage):
    if config.get("schema_version") != 3:
        raise ValueError("Frontal campaign requires schema 3")
    if config.get("scope") != "verified_frontal_only":
        raise ValueError("VMD scope must be verified frontal only")
    if config.get("autovmd_enabled") is not False:
        raise ValueError("AutoVMD is stopped in this campaign")
    if stage not in STAGES:
        raise RuntimeError("Stage is outside the implemented frontal-first checkpoint: " + stage)
    settings = tuple((row["K"], row["alpha"]) for row in config["fixed_recipes"])
    if settings != FIXED_SETTINGS:
        raise ValueError("Only the two approved fixed recipes are permitted")
    if config.get("reserved_access") != "closed until verified freeze":
        raise ValueError("Reserved sources must remain closed")
    if config.get("model_authorization") != "After validated frontal and regional classical evidence":
        raise ValueError("Classical-first sequencing differs from authorization")


def require_checkpoint(stage, input_root, config):
    validate_policy(config, stage)
    if stage == "frontal-fixture":
        return
    readiness = list(Path(input_root).rglob("research_ready.json"))
    fixtures = list(Path(input_root).rglob("frontal_fixture_summary.json"))
    if len(readiness) != 1 or len(fixtures) != 1:
        raise RuntimeError("Need exactly one readiness and one frontal fixture parent")
    verify_parent(readiness[0].parent, ("research_ready.json",))
    ready = json.loads(readiness[0].read_text())
    if ready.get("passed") is not True or ready.get("reserved_confirmation_opened") is not False:
        raise RuntimeError("Imported readiness has not passed with confirmation closed")
    if ready.get("target_definition") != config["target_definition"]:
        raise RuntimeError("Readiness target definition differs")
    verify_parent(fixtures[0].parent, ("frontal_fixture_summary.json",))
    fixture = json.loads(fixtures[0].read_text())
    if fixture.get("passed") is not True or fixture.get("configuration_hash") != canonical_hash(config):
        raise RuntimeError("Frontal numerical checkpoint differs or failed")
    if stage == "frontal-native":
        from .frontal_native import controlled_parent
        controlled_parent(input_root)
    if stage in ("frontal-native-calibration", "frontal-native-events", "frontal-native-gain-events"):
        from .frontal_native import calibration_parent
        calibration_parent(input_root,event_hybrid=stage in ("frontal-native-events","frontal-native-gain-events"),
            event_gain_scaled=stage == "frontal-native-gain-events")
    if stage == "frontal-evaluation":
        from .frontal_evaluation import selected_parent
        selected_parent(input_root)


def execute(stage, input_root, output, config, profile, experiment):
    validate_policy(config, stage)
    from .frontal_replication import run_fixture, run_comparison
    if stage == "frontal-fixture":
        return run_fixture(output, config)
    if stage == "frontal-cache-audit":
        from .frontal_cache_audit import run_audit
        return run_audit(input_root, output, config)
    if stage == "frontal-hybrid-evidence":
        from .frontal_hybrid_evidence import run_evidence
        return run_evidence(input_root, output, config)
    if stage in ("frontal-frozen-transfer", "frontal-gain-hybrid", "frontal-gain-unit-audit", "frontal-calibration-hybrid", "frontal-axis-calibration", "frontal-event-calibration", "frontal-event-gain-calibration"):
        from .frontal_frozen_transfer import run_transfer
        return run_transfer(input_root, output, config,
            hybrid=stage in ("frontal-gain-hybrid", "frontal-gain-unit-audit"),
            unit_audit=stage == "frontal-gain-unit-audit",
            calibration_hybrid=stage in ("frontal-calibration-hybrid", "frontal-axis-calibration", "frontal-event-calibration", "frontal-event-gain-calibration"),
            axis_hybrid=stage == "frontal-axis-calibration",
            event_hybrid=stage in ("frontal-event-calibration", "frontal-event-gain-calibration"),
            event_gain_scaled=stage == "frontal-event-gain-calibration")
    if stage in ("frontal-native", "frontal-native-calibration", "frontal-native-events", "frontal-native-gain-events"):
        from .frontal_native import run_native
        return run_native(input_root, output, config, profile,
            calibration_hybrid=stage in ("frontal-native-calibration", "frontal-native-events", "frontal-native-gain-events"),
            event_hybrid=stage in ("frontal-native-events","frontal-native-gain-events"),
            event_gain_scaled=stage == "frontal-native-gain-events")
    if stage == "frontal-evidence":
        from .frontal_evidence import run_evidence
        return run_evidence(input_root, output, config)
    if stage == "frontal-evaluation":
        from .frontal_evaluation import evaluate
        return evaluate(input_root, output, config)
    return run_comparison(input_root, output, config, profile, experiment or {})
