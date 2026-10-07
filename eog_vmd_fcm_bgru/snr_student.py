"""Scale-sensitive SNR losses and cap-independent VMD neural cleaners.

These are offline window models. The reference model explicitly needs HEOG
and VEOG at inference; it must never be reported as the EEG-only student.
"""
import torch
from torch import nn
from torch.nn import functional as F


def masked_snr_db(prediction, target, mask, floor=1e-8):
    """Per-window 10 log10(clean energy / reconstruction-error energy).

    This retains amplitude errors. It is not scale-invariant SI-SNR. The floor
    is relative to target energy, making the numerical guard unit invariant.
    """
    if prediction.shape != target.shape or mask.shape != target.shape[:2]:
        raise ValueError("Prediction, paired target and channel mask must align")
    if not torch.all(mask.sum(dim=1) > 0):
        raise ValueError("Each window needs a valid EEG channel")
    valid = mask.bool()[..., None]
    clean = torch.where(valid, target, torch.zeros_like(target))
    error = torch.where(valid, prediction - target, torch.zeros_like(target))
    energy = clean.square().sum(dim=(1, 2))
    if not torch.all(energy > 0):
        raise ValueError("SNR needs positive paired-clean energy")
    ratio = error.square().sum(dim=(1, 2)) / energy
    return -10 * torch.log10(ratio.clamp_min(floor))


def snr_target_loss(prediction, target, mask, target_db=20.0):
    """Optimize both direct negative SNR and a smooth 20-dB shortfall."""
    snr = masked_snr_db(prediction, target, mask)
    return (-snr / 10 + F.softplus((target_db - snr) / 5)).mean()


def identity_penalty(artifact, clean, mask, margin=0.005):
    """Per-window preservation barrier, stricter than the 1% selection gate."""
    valid = mask.bool()[..., None]
    error = torch.where(valid, artifact, torch.zeros_like(artifact))
    target = torch.where(valid, clean, torch.zeros_like(clean))
    ratio = error.square().sum(dim=(1, 2)) / target.square().sum(dim=(1, 2)).clamp_min(1e-12)
    return (torch.log1p(ratio / margin**2)).mean()


class TemporalBlock(nn.Module):
    def __init__(self, features, dilation):
        super().__init__()
        self.depthwise = nn.Conv1d(features, features, 9, padding=4 * dilation,
                                   dilation=dilation, groups=features)
        self.pointwise = nn.Conv1d(features, features, 1)
        self.normalization = nn.GroupNorm(4, features)

    def forward(self, values):
        return values + self.pointwise(F.gelu(self.normalization(self.depthwise(values))))


class VMDSpatialStudent(nn.Module):
    """Shared channel TCN: retain full-rate VMD waves and learn their correction.

    No fixed electrode index is interpreted as frontal. With verified regions,
    a separate frontal pool supports other channels without averaging raw
    opposing ocular polarities. Masks are applied before spatial pooling.
    """
    def __init__(self, features=32):
        super().__init__()
        self.features = features
        self.encoder = nn.Sequential(nn.Conv1d(5, features, 9, padding=4), nn.GELU(),
                                     *[TemporalBlock(features, d) for d in [1, 2, 4, 8]])
        self.context = nn.Conv1d(features * 3, features, 1)
        self.head = nn.Conv1d(features * 2, 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, eeg, modes, references, mask, regions):
        batch, channels, samples = eeg.shape
        if modes.shape != (batch, channels, 3, samples) or mask.shape != (batch, channels):
            raise ValueError("VMD vectors and EEG/mask must align")
        if not torch.all(mask.sum(dim=1) > 0):
            raise ValueError("Every cap needs an observed channel")
        valid = mask.bool()[..., None]
        values = torch.where(valid, eeg, torch.zeros_like(eeg))
        scale = values.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        modes = torch.where(valid[:, :, None], modes, torch.zeros_like(modes))
        remainder = values - modes.sum(dim=2)
        stack = torch.cat([values[:, :, None], modes, remainder[:, :, None]], dim=2) / scale[:, :, None]
        pooled = F.avg_pool1d(stack.reshape(batch * channels, 5, samples), 4, ceil_mode=True)
        encoded = self.encoder(pooled).reshape(batch, channels, self.features, -1)
        weights = mask[:, :, None, None].to(encoded.dtype)
        mean = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        maximum = encoded.masked_fill(~mask.bool()[:, :, None, None], -torch.inf).amax(dim=1)
        frontal = (mask.bool() & (regions == 0))[:, :, None, None].to(encoded.dtype)
        frontal_mean = (encoded * frontal).sum(dim=1) / frontal.sum(dim=1).clamp_min(1)
        context = F.gelu(self.context(torch.cat([mean, maximum, frontal_mean], dim=1)))
        repeated = context[:, None].expand(-1, channels, -1, -1)
        head = self.head(torch.cat([encoded, repeated], dim=2).reshape(batch * channels, self.features * 2, -1))
        head = F.interpolate(head, size=samples, mode="linear", align_corners=False).reshape(batch, channels, 4, samples)
        # Signed mode weights permit residual correction while full-rate waves
        # preserve phase and detail lost by the previous scalar interpolation.
        artifact = (torch.tanh(head[:, :, :3]) * modes).sum(dim=2) + head[:, :, 3] * scale
        artifact = torch.where(valid, artifact, torch.zeros_like(artifact))
        return {"artifact": artifact, "cleaned": eeg - artifact}


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

    def correction_gains(self, features, correlations, mask, regions):
        frontal = mask.bool() & (regions == 0)
        support = torch.where(frontal.any(dim=1, keepdim=True), frontal, mask.bool())
        evidence = correlations[:, :, 0].abs().masked_fill(~support[..., None], 0).amax(dim=1)
        repeated = evidence[:, None].expand(-1, features.shape[1], -1)
        threshold = 0.1 + 0.8 * torch.sigmoid(self.threshold_logit)
        gate = torch.sigmoid(40 * (evidence.amax(dim=-1, keepdim=True) - threshold))
        return 2 * torch.tanh(self.gains(torch.cat([features, repeated], dim=-1))) * gate[:, None]
