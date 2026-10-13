"""Retrospective error identities over frozen frontal shape predictions."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .frontal_artifact_shape import SHAPE, VMD
from .frontal_methods import frontal_indices
from .frontal_review import strict_mean
from .io import atomic_json, read_jsonl, sha256_file


PARENT = {
    "run_id": "frontal-shape-full-001",
    "git_sha": "1444b4e226b1e879ad46b2fe638c4d77dc26a753",
    "manifest": "7543c9799eb0911beee1c2c3c5e4c22d1b4fb245c088cbbcb7e5b33c7194586f",
}
TERMS = ("full_mse", "shape_mse", "mse_delta", "discarded_field_mse", "cross_term", "closure_error")


def error_terms(original, target, full_artifact, shaped_artifact):
    """Evaluate a frozen correction; no parameter estimation or new predictions."""
    arrays = [np.asarray(value, float) for value in (original, target, full_artifact, shaped_artifact)]
    if (arrays[0].ndim != 2 or min(arrays[0].shape) < 1 or
            any(value.shape != arrays[0].shape or not np.isfinite(value).all() for value in arrays)):
        raise ValueError("Four finite matched [channels, samples] arrays are required")
    original, target, full_artifact, shaped_artifact = arrays
    full_error = original - full_artifact - target
    shaped_error = original - shaped_artifact - target
    discarded = full_artifact - shaped_artifact
    scale = max(float(np.max(np.abs(value))) for value in arrays)
    np.testing.assert_allclose(shaped_error, full_error + discarded,
                               atol=64 * np.finfo(float).eps * scale, rtol=0)
    full_mse = np.mean(full_error ** 2, axis=-1)
    shape_mse = np.mean(shaped_error ** 2, axis=-1)
    discarded_mse = np.mean(discarded ** 2, axis=-1)
    cross = 2 * np.mean(full_error * discarded, axis=-1)
    delta = shape_mse - full_mse
    closure = delta - (discarded_mse + cross)
    np.testing.assert_allclose(delta, discarded_mse + cross,
                               atol=128 * np.finfo(float).eps * scale ** 2, rtol=0)
    return dict(full_mse=full_mse, shape_mse=shape_mse, mse_delta=delta,
                discarded_field_mse=discarded_mse, cross_term=cross, closure_error=closure)


def diagnostic_parent(input_root):
    paths = list(Path(input_root).rglob("shape_summary.json"))
    if len(paths) != 1:
        raise ValueError("Attach exactly one frozen artifact-shape result")
    parent = paths[0].parent
    if sha256_file(parent / "artifact_manifest.json") != PARENT["manifest"]:
        raise ValueError("Shape diagnostic parent manifest differs")
    verify_parent(parent, ("shape_summary.json", "run_spec.json", "stage_tests_summary.json",
                           "shape_source_rows.jsonl", "shape_channel_scores.csv.gz"))
    spec = json.loads((parent / "run_spec.json").read_text())
    summary = json.loads(paths[0].read_text())
    if any(spec.get(key) != PARENT[key] for key in ("run_id", "git_sha")):
        raise ValueError("Shape diagnostic parent source differs")
    if json.loads((parent / "execution_state.json").read_text()).get("status") != "complete":
        raise ValueError("Shape diagnostic parent is incomplete")
    if json.loads((parent / "stage_tests_summary.json").read_text()).get("passed") is not True:
        raise ValueError("Shape parent numerical contracts failed")
    required = {"examples": 320, "baseline_prediction_parity_checks": 960,
                "baseline_metric_parity_checks": 112320, "scope": "verified_frontal_only",
                "K": 3, "alpha": 2000, "qualification": False,
                "reserved_confirmation_opened": False, "autovmd_executed": False,
                "decompositions_computed": 0, "gains_refitted": False, "weights_trained": False}
    if any(summary.get(key) != value for key, value in required.items()):
        raise ValueError("Shape diagnostic parent scope differs")
    return parent


def run(input_root, output, config, profile):
    if profile != "full":
        raise ValueError("Saved-output diagnostic retains the complete cohort")
    started = time.perf_counter()
    parent = diagnostic_parent(input_root)
    sources = read_jsonl(parent / "shape_source_rows.jsonl")
    if len(sources) != 320 or len({row["example_id"] for row in sources}) != 320:
        raise ValueError("Diagnostic cohort must contain the exact 320 unique saved windows")
    archived = pd.read_csv(parent / "shape_channel_scores.csv.gz")
    records, waveform_checks, dc_checks = [], 0, 0
    for number, row in enumerate(sources):
        name = row["example_id"]
        paths = ("shape_predictions/" + name + ".npz", "shape_decisions/" + name + ".jsonl")
        verify_parent(parent, paths)
        details = read_jsonl(parent / paths[1])
        with np.load(parent / paths[0], allow_pickle=False) as saved:
            original, target = saved["original"], saved["target"]
            ids = frontal_indices(original, saved["names"], saved["regions"], saved["mask"])
            if len(details) != len(ids):
                raise ValueError("Saved frontal decision coverage differs")
            for method in (VMD, SHAPE):
                np.testing.assert_array_equal(saved[method], original - saved["artifact_" + method])
                outside = np.ones(len(original), bool); outside[ids] = False
                np.testing.assert_array_equal(saved[method][outside], original[outside])
                waveform_checks += 1
            full, shape = saved["artifact_" + VMD][ids], saved["artifact_" + SHAPE][ids]
            valid = np.zeros(original.shape[-1], bool); valid[10:-10] = True
            if not valid.any() or np.any(full[:, ~valid]) or np.any(shape[:, ~valid]):
                raise ValueError("Saved artifact invalid lag edges differ")
            scale = max(float(np.max(np.abs(full))), float(np.max(np.abs(shape))))
            np.testing.assert_allclose((full - shape)[:, valid].mean(-1), 0,
                                       atol=64 * np.finfo(float).eps * scale, rtol=0)
            dc_checks += len(ids)
            terms = error_terms(original[ids], target[ids], full, shape)
            for local, index in enumerate(ids):
                detail = details[local]
                channel = str(saved["names"][index])
                if detail["channel"] != channel:
                    raise ValueError("Saved diagnostic channel order differs")
                failure = detail.get("failure")
                if not failure:
                    if not 0 <= detail["rank"] <= len(detail["selected_modes"]) <= 3:
                        raise ValueError("Saved selected-span rank bounds differ")
                records.append(dict(example_id=name, recipient=row["recipient"], donor=row["donor"],
                    condition=row["condition"], input_snr_db=row["input_snr_db"], channel=channel,
                    failure=failure, selected_modes=detail["selected_modes"], rank=detail.get("rank"),
                    retained_centered_energy=detail.get("retained_centered_energy"),
                    discarded_centered_energy=detail.get("discarded_centered_energy"),
                    **{key: float(value[local]) for key, value in terms.items()}))
        atomic_json(output / "shape_error_progress.json", dict(completed=number + 1, total=320))
    frame = pd.DataFrame(records)
    keys = ["example_id", "channel"]
    mse_checks = 0
    for method, term in ((VMD, "full_mse"), (SHAPE, "shape_mse")):
        previous = archived[archived.evaluated_family == method][keys + ["mse"]]
        joined = frame.merge(previous, on=keys, validate="one_to_one")
        if len(joined) != len(frame) or len(previous) != len(frame):
            raise ValueError("Archived diagnostic MSE coverage differs")
        np.testing.assert_allclose(joined[term], joined.mse, rtol=1e-12, atol=1e-24)
        mse_checks += len(joined)
    frame.to_csv(output / "shape_error_channel_terms.csv.gz", index=False, compression="gzip")
    fields = list(TERMS)
    strata = frame.groupby(["recipient", "condition", "input_snr_db"], dropna=False)[fields].agg(strict_mean)
    strata.to_csv(output / "shape_error_recipient_condition_terms.csv")
    strata.groupby(["condition", "input_snr_db"], dropna=False).agg(strict_mean).to_csv(output / "shape_error_condition_terms.csv")
    recipient = strata[strata.index.get_level_values("condition") != "clean"].groupby("recipient").agg(strict_mean)
    recipient.to_csv(output / "shape_error_recipient_terms.csv")
    recipient.agg(strict_mean).to_csv(output / "shape_error_balanced_terms.csv", header=["value"])
    atomic_json(output / "shape_error_summary.json", dict(examples=320, channel_rows=len(frame),
        recipients=int(frame.recipient.nunique()), donors=int(frame[frame.condition != "clean"].donor.nunique()),
        waveform_checks=waveform_checks, dc_channel_checks=dc_checks, mse_parity_checks=mse_checks,
        inherited_failure_rows_scored=int(frame.failure.notna().sum()),
        maximum_absolute_mse_closure_error=float(np.abs(frame.closure_error).max()),
        parent_manifest_sha256=PARENT["manifest"], runtime_s=time.perf_counter() - started,
        scope="verified_frontal_only", analysis="retrospective error identity; not a correction recipe",
        target_access="frozen paired evaluation only; no fitting", predictions_changed=False,
        parameters_fitted=False, decompositions_computed=0, autovmd_executed=False,
        reserved_confirmation_opened=False, weights_trained=False, qualification=False))
