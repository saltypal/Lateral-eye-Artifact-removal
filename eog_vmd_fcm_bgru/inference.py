"""Fresh-load offline inference on explicitly preprocessed EEG tensors."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .model_factory import build_deployment_model, LEGACY_ARCHITECTURE, REGIONAL_ARCHITECTURE
from .bundle_io import digest_file


def load_model(bundle, device="cpu"):
    metadata = json.loads((bundle / "model_contract.json").read_text())
    if digest_file(bundle / "model.pt") != metadata["model_sha256"]:
        raise ValueError("Saved model SHA256 differs from its inference contract")
    architecture_id = metadata.get("architecture_id", LEGACY_ARCHITECTURE)
    schema_version = int(metadata.get("bundle_schema_version", 1))
    if schema_version not in (1, 2):
        raise ValueError(f"Unsupported bundle schema version: {schema_version}")
    if schema_version == 1 and architecture_id != LEGACY_ARCHITECTURE:
        raise ValueError("Schema v1 bundles may contain only the legacy EEG-only student")
    if schema_version == 2 and architecture_id != REGIONAL_ARCHITECTURE:
        raise ValueError("Schema v2 bundle architecture is not recognized")
    model = build_deployment_model(architecture_id, metadata["architecture"]).to(device)
    state = torch.load(bundle / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval(), metadata


def predict_preprocessed(model, eeg, mask, regions, device="cpu", hemispheres=None, coordinates=None, coordinate_mask=None):
    if eeg.ndim != 3 or mask.shape != eeg.shape[:2] or regions.shape != eeg.shape[:2]:
        raise ValueError("Expected EEG [B,C,T] and aligned mask/regions [B,C]")
    if not np.isin(mask, [0, 1]).all() or not (mask.sum(axis=1) > 0).all():
        raise ValueError("Binary channel masks must retain at least one electrode per cap")
    if not np.isin(regions, [0, 1, 2, 3]).all():
        raise ValueError("Region IDs must be frontal=0, posterior=1, central=2 or unknown=3")
    if not np.isfinite(eeg[mask.astype(bool)]).all():
        raise ValueError("Valid EEG channels must be finite")
    arguments = [torch.as_tensor(eeg, dtype=torch.float32, device=device), torch.as_tensor(mask, device=device),
                 torch.as_tensor(regions, dtype=torch.long, device=device)]
    if getattr(model, "architecture_id", None) == REGIONAL_ARCHITECTURE:
        batch, channels = eeg.shape[:2]
        hemispheres = np.full((batch, channels), 3, dtype=np.int64) if hemispheres is None else hemispheres
        if hemispheres.shape != (batch, channels) or not np.isin(hemispheres, [0, 1, 2, 3]).all():
            raise ValueError("V2 hemispheres must align with EEG and use IDs 0..3")
        if coordinates is not None and coordinates.shape[:2] != (batch, channels):
            raise ValueError("V2 coordinates must align with EEG [B,C,D]")
        if coordinate_mask is not None and coordinate_mask.shape != (batch, channels):
            raise ValueError("V2 coordinate mask must align with EEG [B,C]")
        arguments.extend([torch.as_tensor(hemispheres, dtype=torch.long, device=device),
                          None if coordinates is None else torch.as_tensor(coordinates, dtype=torch.float32, device=device),
                          None if coordinate_mask is None else torch.as_tensor(coordinate_mask, device=device)])
    with torch.no_grad():
        output = model(*arguments)
    return {name: value.cpu().numpy() for name, value in output.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preprocessed", action="store_true", required=True,
                        help="Confirm independent-trial 200-Hz, 0.5-40-Hz offline preprocessing")
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    data = np.load(arguments.input, allow_pickle=False)
    model, contract = load_model(arguments.bundle, arguments.device)
    if int(data["fs"]) != contract["preprocessing"]["fs"]:
        raise ValueError("Sampling frequency differs from the model contract")
    eeg = data["eeg"]
    mask = data["mask"] if "mask" in data else np.ones(eeg.shape[:2], dtype=np.int64)
    regions = data["regions"] if "regions" in data else np.full(eeg.shape[:2], 3, dtype=np.int64)
    hemispheres = data["hemispheres"] if "hemispheres" in data else None
    coordinates = data["coordinates"] if "coordinates" in data else None
    coordinate_mask = data["coordinate_mask"] if "coordinate_mask" in data else None
    result = predict_preprocessed(model, eeg, mask, regions, arguments.device, hemispheres, coordinates, coordinate_mask)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {**result, "mask": mask, "regions": regions, "fs": 200}
    if hemispheres is not None:
        payload["hemispheres"] = hemispheres
    if coordinates is not None:
        payload["coordinates"] = coordinates
    if coordinate_mask is not None:
        payload["coordinate_mask"] = coordinate_mask
    np.savez_compressed(arguments.output, **payload)
    print("Saved offline predictions", arguments.output, flush=True)
