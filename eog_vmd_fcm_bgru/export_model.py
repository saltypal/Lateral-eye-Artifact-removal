"""Export a trained model and verify actual predictions in a fresh process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from .provenance import save_json, sha256_file


def export_and_verify(network, output, eeg, expected):
    bundle = output / "offline_model"
    bundle.mkdir(exist_ok=True)
    run = json.loads((output / "run_config.json").read_text())
    training = json.loads((output / "training_summary.json").read_text())
    torch.save({name: value.detach().cpu() for name, value in network.state_dict().items()}, bundle / "model.pt")
    metadata = {"git_sha": run["git_sha"], "model_sha256": sha256_file(bundle / "model.pt"),
                "architecture": {"features": network.features, "hidden": network.gru.hidden_size}, "training": training,
                "preprocessing": {"fs": 200, "band_hz": [0.5, 40], "butterworth_order": 4,
                                  "filter": "scipy.signal.sosfiltfilt per independent trial",
                                  "resampling": "scipy.signal.resample_poly before filtering",
                                  "reference": "retain source reference; training units unverified"},
                "normalization": "per channel/window mean and standard deviation, floor 1e-6",
                "inputs": {"eeg": "[B,C,T] float32 preprocessed; EEG only",
                           "mask": "[B,C] binary; invalid electrodes pass through",
                           "regions": "[B,C] 0 frontal, 1 posterior, 2 central, 3 unknown"},
                "channel_handling": "shared weights and masked pooling; permutation metadata must align",
                "operating_mode": "offline; no streaming state; whole-window future information",
                "quality_limit": "feasibility checkpoint; no complete-removal or unseen-cap accuracy certification"}
    save_json(bundle / "model_contract.json", metadata)
    fixture = output / "fresh_load_input.npz"
    np.savez_compressed(fixture, eeg=eeg, fs=200, mask=np.ones(eeg.shape[:2], dtype=np.int64),
                        regions=np.full(eeg.shape[:2], 3, dtype=np.int64))
    prediction_path = output / "fresh_load_output.npz"
    subprocess.run([sys.executable, "-m", "eog_vmd_fcm_bgru.inference", "--bundle", str(bundle),
                    "--input", str(fixture), "--output", str(prediction_path), "--preprocessed"],
                   check=True, env=os.environ.copy())
    restored = np.load(prediction_path, allow_pickle=False)["cleaned"]
    np.testing.assert_allclose(restored, expected, atol=1e-4, rtol=1e-4)
    save_json(output / "fresh_load_verification.json", {"passed": True, "fresh_process": True,
              "source_device": str(next(network.parameters()).device), "reload_device": "cpu",
              "max_absolute_difference": float(np.max(np.abs(restored - expected))),
              "absolute_tolerance": 1e-4, "relative_tolerance": 1e-4,
              "fixture": "actual held-out smoke EEG; same preprocessed window"})
