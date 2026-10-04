"""Test shared publisher contamination coefficients as possible leakage groups."""
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from .provenance import save_json, sha256_file


def inspect_coefficient_groups(root, output, repository):
    sources = list(Path("/kaggle/input").rglob("publisher_matlab/Pure_Data.mat"))
    if len(sources) != 1:
        raise RuntimeError("Attach one verified original Klados source-check output")
    directory = sources[0].parent
    catalog = json.loads((repository / "data_provenance/klados_mendeley_v4_catalog.json").read_text())
    payload = {}
    for item in catalog["files"]:
        path = directory / item["name"]
        if sha256_file(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise RuntimeError("Publisher coefficient audit source hash mismatch")
        payload[path.name] = loadmat(path, simplify_cells=True)
    recording_ids = sorted(int(key[3:].split("_")[0]) for key in payload["Pure_Data.mat"] if key.startswith("sim"))
    coefficients, diagnostics = [], []
    for recording in recording_ids:
        pure = np.asarray(payload["Pure_Data.mat"][f"sim{recording}_resampled"], dtype=np.float64)
        contaminated = np.asarray(payload["Contaminated_Data.mat"][f"sim{recording}_con"], dtype=np.float64)
        eyes = np.stack([np.asarray(payload[name][f"{prefix}_{recording}"], dtype=np.float64)
                         for name, prefix in [("HEOG.mat", "heog"), ("VEOG.mat", "veog")]])
        if contaminated.shape != pure.shape or eyes.shape[-1] != pure.shape[-1]:
            raise RuntimeError("Publisher coefficient audit alignment mismatch")
        artifact = contaminated - pure
        fitted, _, rank, singular = np.linalg.lstsq(eyes.T, artifact.T, rcond=None)
        estimated = fitted.T @ eyes
        relative_error = np.linalg.norm(artifact - estimated) / max(np.linalg.norm(artifact), np.finfo(float).tiny)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
        diagnostics.append({"publisher_recording": recording, "relative_linear_model_error": float(relative_error),
                            "rank": int(rank), "condition_number": condition, "coefficients_heog_veog": fitted.T.tolist()})
        coefficients.append(fitted.T.flatten())
    coefficients = np.stack(coefficients)
    norms = np.linalg.norm(coefficients, axis=1)
    distances = np.linalg.norm(coefficients[:, None] - coefficients[None], axis=-1) / np.maximum(
        np.maximum(norms[:, None], norms[None]), np.finfo(float).tiny)
    np.fill_diagonal(distances, 0)
    tree = linkage(squareform(distances, checks=True), method="complete")
    groupings = []
    for tolerance in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
        labels = fcluster(tree, tolerance, criterion="distance")
        groups = [[recording_ids[index] for index in np.flatnonzero(labels == group)] for group in np.unique(labels)]
        groupings.append({"relative_coefficient_tolerance": tolerance, "group_count": len(groups), "source_recording_groups": groups})
    stable_pairs = all(item["group_count"] == 27 and all(len(group) == 2 for group in item["source_recording_groups"])
                       for item in groupings[1:4])
    stable_pairs = stable_pairs and all(item["source_recording_groups"] == groupings[1]["source_recording_groups"] for item in groupings[1:4])
    model_exact = all(row["rank"] == 2 and row["relative_linear_model_error"] <= 1e-5 for row in diagnostics)
    selected = groupings[2]
    save_json(output / "publisher_coefficient_fits.json", diagnostics)
    save_json(output / "coefficient_group_sensitivity.json", groupings)
    np.save(output / "coefficient_relative_distances.npy", distances)
    mapping = json.loads((repository / "data_provenance/klados_verified_export_mapping.json").read_text())
    export_groups = []
    for record in mapping["records"]:
        source_id = record["publisher_ids"]["klados_pure_eeg.npy"][0]
        group = next(index for index, members in enumerate(selected["source_recording_groups"]) if source_id in members)
        export_groups.append({"npy_record": record["npy_record"], "publisher_recording": source_id, "coefficient_group": group})
    save_json(output / "klados_coefficient_export_groups.json", export_groups)
    summary = {"publisher_recordings": len(recording_ids), "export_records": len(export_groups),
        "linear_two_reference_model_verified": model_exact, "stable_27_pairs_across_tolerances": stable_pairs,
        "provisional_shared_coefficient_groups_supported": bool(model_exact and stable_pairs),
        "participant_identity_verified": False, "current_train_test_split_changed": False,
        "interpretation": "shared contamination coefficients are a grouping clue, not explicit subject labels",
        "paper": "https://doi.org/10.1016/j.dib.2016.06.032", "victory": False}
    save_json(output / "klados_coefficient_group_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
