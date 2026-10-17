"""Historical 2499-parameter EOG-assisted model, preserved without architecture edits.
Copied from the original snr_student.py; source checkpoint Git 2ec0b3cd3da636cc3aed3b476ff890a6f56e2627.
This is a runtime EOG + VMD model, not an EEG-only deployment network.
"""
import torch
from torch import nn

class EOGGainStudent(nn.Module):
    """VMD-informed nonlinear gains on an explicit two-reference projection.

    The same MLP processes any number of EEG channels. Signed regression
    coefficients retain opposite horizontal-eye polarities across electrodes.
    """
    def __init__(self):
        super().__init__()
        self.gains = nn.Sequential(nn.Linear(15, 48), nn.GELU(), nn.Linear(48, 32),
                                   nn.GELU(), nn.Linear(32, 2))
        nn.init.zeros_(self.gains[-1].weight)
        nn.init.zeros_(self.gains[-1].bias)

    def correction_gains(self, features, correlations, mask, regions):
        return 2 * torch.tanh(self.gains(features))

    def forward(self, eeg, modes, references, mask, regions):
        if references.shape != (eeg.shape[0], 2, eeg.shape[-1]):
            raise ValueError("Explicit aligned HEOG and VEOG are required")
        samples = eeg.shape[-1]
        values = eeg - eeg.mean(dim=-1, keepdim=True)
        eyes = references - references.mean(dim=-1, keepdim=True)
        eyes = eyes / eyes.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        gram = eyes @ eyes.transpose(1, 2) / samples
        regularized = gram + 1e-4 * torch.eye(2, device=eeg.device, dtype=eeg.dtype)[None]
        coefficients = torch.linalg.solve(regularized, eyes @ values.transpose(1, 2) / samples).transpose(1, 2)
        components = coefficients[..., None] * eyes[:, None]
        waves = torch.cat([values[:, :, None], modes], dim=2)
        rms = waves.square().mean(dim=-1).sqrt().clamp_min(1e-6)
        correlations = torch.einsum("bckt,bet->bcke", waves, eyes) / (samples * rms[..., None])
        fraction = modes.square().mean(dim=-1) / values.square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
        standardized = values / rms[:, :, :1]
        kurtosis = standardized.pow(4).mean(dim=-1, keepdim=True).clamp_max(100) / 10
        # 8 correlations + 3 relative mode energies + 1 kurtosis + 2 signed
        # regression amplitudes + 1 known-frontal flag = 15 shared features.
        features = torch.cat([correlations.flatten(2), fraction, kurtosis,
                              coefficients / rms[:, :, :1], (regions == 0).to(eeg.dtype)[..., None]], dim=-1)
        gain = self.correction_gains(features, correlations, mask, regions)
        artifact = (components * gain[..., None]).sum(dim=2)
        artifact = torch.where(mask.bool()[..., None], artifact, torch.zeros_like(artifact))
        return {"artifact": artifact, "cleaned": eeg - artifact}

class EOGContextGainStudent(EOGGainStudent):
    """Shared support evidence gates signed corrections throughout the cap.

    Prefer verified frontal support when available; unknown montages use all
    observed channels. This fallback never invents a frontal electrode order.
    The signed HEOG/VEOG projection itself is unchanged by this evidence pool.
    """
    def __init__(self):
        super().__init__()
        self.gains = nn.Sequential(nn.Linear(17, 48), nn.GELU(), nn.Linear(48, 32),
                                   nn.GELU(), nn.Linear(32, 2))
        nn.init.zeros_(self.gains[-1].weight)
        nn.init.zeros_(self.gains[-1].bias)
        self.threshold_logit = nn.Parameter(torch.tensor(0.5108256))  # 0.6 in bounded [0.1,0.9].
        self.inference_gate_mode = "soft"
        self.inference_gate_offset = 0.0

    def correction_gains(self, features, correlations, mask, regions):
        frontal = mask.bool() & (regions == 0)
        support = torch.where(frontal.any(dim=1, keepdim=True), frontal, mask.bool())
        evidence = correlations[:, :, 0].abs().masked_fill(~support[..., None], 0).amax(dim=1)
        repeated = evidence[:, None].expand(-1, features.shape[1], -1)
        threshold = 0.1 + 0.8 * torch.sigmoid(self.threshold_logit)
        maximum = evidence.amax(dim=-1, keepdim=True)
        if self.inference_gate_mode == "soft":
            gate = torch.sigmoid(40 * (maximum - threshold))
        elif self.inference_gate_mode == "hard":
            gate = (maximum >= threshold + self.inference_gate_offset).to(features.dtype)
        else:
            raise ValueError("Unknown inference gate mode")
        return 2 * torch.tanh(self.gains(torch.cat([features, repeated], dim=-1))) * gate[:, None]
