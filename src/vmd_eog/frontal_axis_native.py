"""Identity checks for an exploratory native preservation repair.

Failed qualification evidence may inform a repair experiment. It never becomes
permission for teacher training, confirmation evaluation or deployment.
"""
import json
from pathlib import Path

from .artifacts import verify_parent
from .io import sha256_file


CONTROLLED = {
    "run_id": "frontal-monotonic-events-001",
    "git_sha": "939fadf435440b3cac7436f03c3ce6d80dfa52aa",
    "manifest": "1ab061565aca07fb826d48477087c949720dfa300facad90be3ebe2008876d20",
}
NATIVE = {
    "run_id": "frontal-native-monotonic-001",
    "git_sha": "437198ca4d4ea4aff3003bb15775c8b29fb49ac9",
    "manifest": "220fa71434f8d9bfba5b42eff88926939548e65e2457853cd94aa89fe21e5f6d",
}


def checked_parent(input_root, summary_name, expected):
    paths = list(Path(input_root).rglob(summary_name))
    if len(paths) != 1:
        raise ValueError("Attach exactly one fixed repair parent: " + summary_name)
    parent = paths[0].parent
    if sha256_file(parent / "artifact_manifest.json") != expected["manifest"]:
        raise ValueError("Repair parent manifest differs: " + summary_name)
    verify_parent(parent, (summary_name, "run_spec.json"))
    spec = json.loads((parent / "run_spec.json").read_text())
    state = json.loads((parent / "execution_state.json").read_text())
    if any(spec.get(key) != expected[key] for key in ("run_id", "git_sha")):
        raise ValueError("Repair parent source identity differs")
    if state.get("status") != "complete":
        raise ValueError("Repair parent did not complete")
    summary = json.loads(paths[0].read_text())
    if (summary.get("scope") != "verified_frontal_only" or
            summary.get("reserved_confirmation_opened") is not False or
            summary.get("autovmd_executed") is not False or
            summary.get("calibration_hybrid") is not True or
            summary.get("event_hybrid") is not True):
        raise ValueError("Repair parent scope differs")
    return parent, summary


def exploratory_parent(input_root):
    """Require the exact failed-development precedent, without qualifying it."""
    controlled, summary = checked_parent(input_root, "frontal_transfer_summary.json", CONTROLLED)
    if (summary.get("K") != 3 or summary.get("alpha") != 2000 or
            summary.get("monotonic_events") is not True or
            summary.get("weights_trained") is not False or summary.get("qualification") is not False):
        raise ValueError("Repair requires unqualified fixed K3/2000 monotonic evidence")
    native, summary = checked_parent(input_root, "native_frontal_summary.json", NATIVE)
    if summary.get("profile") != "pilot" or summary.get("recordings") != 2:
        raise ValueError("Repair requires the fixed two-record native precedent")
    if (summary.get("teacher_qualified") is not False or
            summary.get("native_parameter_retuning") is not False):
        raise ValueError("Native precedent qualification or tuning scope differs")
    return controlled, native
