"""Small immutable contracts; numerical computation is guarded separately."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

PHASE_A = ("contracts", "audit", "source-fixture", "corpus", "vmd", "posterior", "regional", "review")
PHASE_B = ("teacher-oof", "student-paired", "student-distill", "freeze", "final-eval", "export")


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def require_kaggle():
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Numerical campaign execution is Kaggle-only")


def require_approval(stage, input_root, config):
    if stage not in PHASE_B:
        return
    files = list(Path(input_root).rglob("classical_gate.json"))
    if len(files) != 1:
        raise RuntimeError("Phase B needs one completed classical gate artifact")
    gate = json.loads(files[0].read_text())
    if gate.get("campaign_id") != config["campaign_id"] or gate.get("passed") is not True:
        raise RuntimeError("Classical gate is not passing for this campaign")
    if config.get("model_authorization") != "Proceed to models if all classical gates pass":
        raise RuntimeError("Explicit conditional model authorization is absent")
    if not gate.get("selected_recipe_hash") or not gate.get("evidence_hashes"):
        raise RuntimeError("Classical gate has no traceable evidence")


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    stage: str
    git_sha: str
    campaign_id: str = "vmd-eog-20261010"
    profile: str = "pilot"
    parent_run: str | None = None
    kernel_sources: tuple = ()
    dataset_sources: tuple = ()

    def validate(self):
        if self.stage not in PHASE_A + PHASE_B:
            raise ValueError("Unknown campaign stage")
        if len(self.git_sha) != 40 or any(c not in "0123456789abcdef" for c in self.git_sha):
            raise ValueError("RunSpec needs the exact published Git SHA")
        if not self.run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in self.run_id):
            raise ValueError("Unsafe run identifier")
        if self.profile not in ("pilot", "full"):
            raise ValueError("Profile must be pilot or full")
        return asdict(self)
