"""Fresh-load offline inference on explicitly preprocessed EEG tensors."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .student import SharedChannelStudent
from .bundle_io import digest_file


def load_model(bundle, device="cpu"):
    metadata = json.loads((bundle / "model_contract.json").read_text())
    if digest_file(bundle / "model.pt") != metadata["model_sha256"]:
        raise ValueError("Saved model SHA256 differs from its inference contract")
    model = SharedChannelStudent(**metadata["architecture"]).to(device)
    state = torch.load(bundle / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval(), metadata


def predict_preprocessed(model, eeg, mask, regions, device="cpu"):
    if eeg.ndim != 3 or mask.shape != eeg.shape[:2] or regions.shape != eeg.shape[:2]:
        raise ValueError("Expected EEG [B,C,T] and aligned mask/regions [B,C]")
    if not np.isin(mask, [0, 1]).all() or not (mask.sum(axis=1) > 0).all():
        raise ValueError("Binary channel masks must retain at least one electrode per cap")
    if not np.isin(regions, [0, 1, 2, 3]).all():
        raise ValueError("Region IDs must be frontal=0, posterior=1, central=2 or unknown=3")
    if not np.isfinite(eeg[mask.astype(bool)]).all():
        raise ValueError("Valid EEG channels must be finite")
    with torch.no_grad():
        output = model(torch.as_tensor(eeg, dtype=torch.float32, device=device),
                       torch.as_tensor(mask, device=device),
                       torch.as_tensor(regions, dtype=torch.long, device=device))
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
    result = predict_preprocessed(model, eeg, mask, regions, arguments.device)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, **result, mask=mask, regions=regions, fs=200)
    print("Saved offline predictions", arguments.output, flush=True)
