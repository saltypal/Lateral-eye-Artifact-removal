"""Small immutable contracts; numerical computation is guarded separately."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

PHASE_A = ("contracts", "audit", "source-fixture", "classical-fixture", "paper-fixture", "native-protocol", "vmd-diagnosis", "vmd-sobi", "corpus", "vmd", "posterior", "regional", "review")
PHASE_B = ("teacher-oof", "student-paired", "student-distill", "freeze", "final-eval", "export")
RESEARCH_STAGES = ("research-ready", "neural-fixture", "neural-diagnosis", "autovmd-cache", "autovmd-search", "router-train", "neural-review", "benchmark", "posterior-context")


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def require_kaggle():
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Numerical campaign execution is Kaggle-only")


def require_approval(stage, input_root, config):
    if config.get("schema_version") == 2:
        if config.get("model_authorization") != "Parallel paired learning after verified research readiness":
            raise RuntimeError("New campaign authorization is absent")
        if stage in ("contracts", "research-ready", "neural-fixture", "native-protocol", "paper-fixture"):
            return
        if stage in RESEARCH_STAGES + PHASE_B:
            from .artifacts import verify_parent
            files = list(Path(input_root).rglob("research_ready.json"))
            if len(files) != 1:
                raise RuntimeError("Need exactly one research readiness parent")
            verify_parent(files[0].parent, ("research_ready.json",))
            gate = json.loads(files[0].read_text())
            if gate.get("campaign_id") != config["campaign_id"] or gate.get("passed") is not True:
                raise RuntimeError("Research readiness did not pass")
            if gate.get("reserved_confirmation_opened") is not False:
                raise RuntimeError("Research readiness has opened reserved sources")
            if stage in ("teacher-oof", "student-distill"):
                require_gate(input_root, "teacher_eligible.json", config)
            if stage in ("final-eval", "export"):
                require_gate(input_root, "model_freeze.json", config)
            return
    if stage not in PHASE_B:
        return
    files = list(Path(input_root).rglob("classical_gate.json"))
    if len(files) != 1:
        raise RuntimeError("Phase B needs one completed classical gate artifact")
    gate = json.loads(files[0].read_text())
    if gate.get("campaign_id") != config["campaign_id"] or gate.get("passed") is not True:
        raise RuntimeError("Classical gate is not passing for this campaign")
    if gate.get("paper_evaluation_complete") is not True:
        raise RuntimeError("Prescribed paper evaluation is incomplete; old project metrics cannot authorize models")
    if config.get("model_authorization") != "Proceed to models if all classical gates pass":
        raise RuntimeError("Explicit conditional model authorization is absent")
    if not gate.get("selected_recipe_hash") or not gate.get("evidence_hashes"):
        raise RuntimeError("Classical gate has no traceable evidence")
    from .artifacts import verify_parent
    directory=files[0].parent
    verify_parent(directory,("classical_gate.json","approved_recipe.json","review_summary.json"))
    recipe=json.loads((directory/"approved_recipe.json").read_text())
    if canonical_hash(recipe)!=gate["selected_recipe_hash"]:
        raise RuntimeError("Approved recipe differs from passing evidence snapshot")
    review=json.loads((directory/"review_summary.json").read_text())
    selected=[r for r in review.get("methods",[]) if r["method"]==gate.get("selected_method")]
    if len(selected)!=1 or not selected[0].get("passed") or not all(selected[0].get("gates",{}).values()):
        raise RuntimeError("Selected method does not satisfy archived classical gates")
    if gate.get("reserved_confirmation_opened") is not False:
        raise RuntimeError("Classical selection has opened reserved confirmation data")
    from .io import sha256_file
    for name,digest in gate["evidence_hashes"].items():
        relative=gate.get("evidence_files",{}).get(name)
        if relative is None:
            raise RuntimeError("Classical evidence payload is absent")
        verify_parent(directory,(relative,))
        if sha256_file(directory/relative)!=digest:
            raise RuntimeError("Classical evidence snapshot differs from recorded hash")


def require_gate(input_root, name, config):
    from .artifacts import verify_parent
    files = list(Path(input_root).rglob(name))
    if len(files) != 1:
        raise RuntimeError("Need exactly one " + name)
    verify_parent(files[0].parent, (name,))
    gate = json.loads(files[0].read_text())
    if gate.get("campaign_id") != config["campaign_id"] or gate.get("passed") is not True:
        raise RuntimeError(name + " has not passed")


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
        if self.stage not in PHASE_A + PHASE_B + RESEARCH_STAGES:
            raise ValueError("Unknown campaign stage")
        if len(self.git_sha) != 40 or any(c not in "0123456789abcdef" for c in self.git_sha):
            raise ValueError("RunSpec needs the exact published Git SHA")
        if not self.run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in self.run_id):
            raise ValueError("Unsafe run identifier")
        if self.profile not in ("pilot", "full"):
            raise ValueError("Profile must be pilot or full")
        return asdict(self)
