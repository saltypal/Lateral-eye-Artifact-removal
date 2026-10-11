"""Read-only Kaggle diagnosis of archived fixed frontal cache reconstruction."""
import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .experiments import corpus_parent
from .frontal_methods import FixedRecipe, frontal_indices
from .frontal_replication import ArchivedModes
from .io import atomic_json, sha256_file
from .posterior_context import select_examples


def run_audit(input_root, output, config):
    corpus, manifest = corpus_parent(input_root)
    cache = ArchivedModes(input_root, corpus, config)
    rows = select_examples(manifest, "full")
    records = []
    for number, row in enumerate(rows):
        data = dict(np.load(corpus / row["array_path"], allow_pickle=False))
        ids = frontal_indices(data["eeg"], data["channel_names"], data["regions"], data["mask"])
        for K in (3, 4):
            parent, item = cache.index[(row["example_id"], K)]
            if item["source_sha256"] != sha256_file(corpus / row["array_path"]):
                raise ValueError("Audit source identity differs")
            verify_parent(parent, (item["path"],))
            if sha256_file(parent / item["path"]) != item["sha256"]:
                raise ValueError("Audit payload identity differs")
            with np.load(parent / item["path"], allow_pickle=False) as saved:
                pieces = saved["components"]
                infos = {entry["channel"]: entry for entry in item["diagnostics"]}
                for index in ids:
                    reference = data["eeg"][index].astype(np.float64)
                    modes, residual = pieces[index, :K], pieces[index, K]
                    sum32 = modes.sum(0) + residual
                    sum64 = modes.sum(0, dtype=np.float64) + residual.astype(np.float64)
                    difference = sum64 - reference
                    bad = np.abs(difference) > 1e-11 + 1e-5 * np.abs(reference)
                    records.append({"example_id": row["example_id"], "K": K,
                        "channel": str(data["channel_names"][index]),
                        "dtype": str(pieces.dtype), "solver_failure": infos[int(index)].get("failure"),
                        "passes_float32_sum": bool(np.allclose(sum32, reference, rtol=1e-5, atol=1e-11)),
                        "passes_float64_sum": bool(not bad.any()),
                        "bad_samples_float64": int(bad.sum()),
                        "max_abs_error_float32": float(np.max(np.abs(sum32-reference))),
                        "max_abs_error_float64": float(np.max(np.abs(difference))),
                        "rms_relative_error_float64": float(np.linalg.norm(difference)/max(np.linalg.norm(reference), 1e-30)),
                        "worst_sample": int(np.argmax(np.abs(difference))),
                        "signal_peak": float(np.max(np.abs(reference)))})
        if (number+1) % 40 == 0:
            pd.DataFrame(records).to_csv(output / "frontal_cache_audit.csv", index=False)
            print("FRONTAL_CACHE_AUDIT", number+1, len(rows), flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "frontal_cache_audit.csv", index=False)
    atomic_json(output / "frontal_cache_audit_summary.json", {
        "scope": "verified_frontal_only", "autovmd_executed": False,
        "reserved_confirmation_opened": False, "examples": len(rows),
        "channel_recipe_checks": len(records),
        "float32_reconstruction_failures": int((~frame.passes_float32_sum).sum()),
        "float64_reconstruction_failures": int((~frame.passes_float64_sum).sum()),
        "solver_failures": int(frame.solver_failure.notna().sum()),
        "maximum_error_float32": float(frame.max_abs_error_float32.max()),
        "maximum_error_float64": float(frame.max_abs_error_float64.max()),
        "maximum_rms_relative_error": float(frame.rms_relative_error_float64.max()),
        "cache_mutated": False, "decompositions_computed": 0,
        "qualification": False, "purpose": "diagnosis; unchanged reconstruction tolerance"})
