"""Kaggle-only numerical parity gate for the lower-memory VMD implementation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from scipy.io import loadmat
from .dataset_io import preprocess
from .provenance import save_json, sha256_file
from .vmd_expert import decompose
from .vmd_rolling import rolling_decompose


def check_engine(output, repository):
    paths = list(Path("/kaggle/input").rglob("publisher_matlab/Contaminated_Data.mat"))
    if len(paths) != 1:
        raise RuntimeError("Attach one verified Klados publisher-source output")
    catalog = json.loads((repository / "data_provenance/klados_mendeley_v4_catalog.json").read_text())
    entry = next(item for item in catalog["files"] if item["name"] == paths[0].name)
    if sha256_file(paths[0]) != entry["sha256"]:
        raise RuntimeError("VMD parity source hash mismatch")
    payload = loadmat(paths[0], simplify_cells=True)
    rows = []
    fixture = None
    for recording in [2, 3]:
        for channel in [0, 9]:
            values = preprocess(payload[f"sim{recording}_con"])[channel, 2000:3024]
            fixture = values if fixture is None else fixture
            for modes, alpha in [(3, 250), (5, 1000), (8, 2000)]:
                reference, _, expected = decompose(values, modes, alpha, relative_tolerance=True, tolerance=1e-6)
                actual, residual, measured = rolling_decompose(values, modes, alpha)
                vector_error = float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))
                center_error = float(np.max(np.abs(np.asarray(expected["centers_hz"]) - measured["centers_hz"])))
                passed = bool(vector_error < 1e-5 and center_error < 1e-4
                              and expected["iterations"] == measured["iterations"])
                np.testing.assert_allclose(actual.sum(axis=0) + residual, values, atol=5e-5, rtol=1e-6)
                rows.append({"publisher_recording": recording, "channel": channel, "K": modes, "alpha": alpha,
                    "vector_relative_difference": vector_error, "center_max_difference_hz": center_error,
                    "reference_iterations": expected["iterations"], "rolling_iterations": measured["iterations"],
                    "hit_iteration_limit": measured["hit_iteration_limit"], "parity_pass": passed,
                    "reference_runtime_s": expected["runtime_s"], "rolling_runtime_s": measured["runtime_s"],
                    "reference_iteration_array_lower_bound_bytes": expected["vmdpy_iterative_array_lower_bound_bytes"],
                    "rolling_spectral_iterate_storage_bytes": measured["spectral_iterate_storage_bytes"]})
                pd.DataFrame(rows).to_csv(output / "vmd_engine_parity.csv", index=False)
                print("VMD engine parity", recording, channel, modes, alpha, passed, flush=True)
    np.save(output / "training_only_benchmark_fixture.npy", fixture)
    resources = []
    for backend in ["reference", "rolling"]:
        response = subprocess.check_output([sys.executable, "-m", "eog_vmd_fcm_bgru.vmd_engine_check",
            "--fixture", str(output / "training_only_benchmark_fixture.npy"), "--backend", backend], text=True)
        resources.append(json.loads(response))
    save_json(output / "vmd_engine_resources.json", resources)
    passed = all(row["parity_pass"] for row in rows)
    save_json(output / "vmd_engine_gate.json", {"parity_pass": passed, "source_sha256": entry["sha256"],
        "training_only_publisher_recordings": [2, 3], "test_records_used": [],
        "quality_improvement_established": False, "full_validation": False,
        "parity_scope": "12 actual training-channel K/alpha cases plus numeric fixtures",
        "memory_scope": "fresh-process high-water RSS includes imports; array sizes are calculated storage only"})
    if not passed:
        raise RuntimeError("VMD engine parity failed; do not use it for features")


def benchmark_subprocess(fixture, backend):
    import resource
    values = np.load(fixture, allow_pickle=False)
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    times = []
    for _ in range(5):
        started = time.perf_counter()
        if backend == "reference":
            decompose(values, 5, 1000, relative_tolerance=True, tolerance=1e-6)
        else:
            rolling_decompose(values, 5, 1000)
        times.append(time.perf_counter() - started)
    print(json.dumps({"backend": backend, "K": 5, "alpha": 1000, "samples": len(values),
        "repeats": 5, "median_runtime_s": float(np.median(times)),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "process_peak_before_calls_kib": baseline}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--backend", choices=["reference", "rolling"], required=True)
    args = parser.parse_args()
    benchmark_subprocess(args.fixture, args.backend)
