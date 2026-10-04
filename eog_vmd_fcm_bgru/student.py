"""Offline EEG-only residual student with parameters independent of cap size.

VMD/ICA supply training teachers, never hidden runtime EOG inputs. The two
neural heads are temporal/context experts, not literal implementations of ICA.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SharedChannelStudent(nn.Module):
    def __init__(self, features=24, hidden=32):
        super().__init__()
        self.features = features
        self.temporal = nn.Sequential(nn.Conv1d(1, 16, 9, stride=2, padding=4), nn.GELU(),
                                      nn.Conv1d(16, features, 9, stride=2, padding=4), nn.GELU())
        # LayerNorm acts on feature coordinates, not batch/padded electrodes.
        self.normalization = nn.LayerNorm(features)
        self.context = nn.Sequential(nn.Linear(features * 2, features), nn.GELU())
        self.gru = nn.GRU(features * 2, hidden, batch_first=True, bidirectional=True)
        self.temporal_residual = nn.Linear(hidden * 2, 1)
        self.context_residual = nn.Linear(hidden * 2 + features, 1)
        self.gate = nn.Linear(hidden * 2, 1)
        self.router_delta = nn.Linear(hidden * 2, 1)
        nn.init.zeros_(self.router_delta.weight)
        nn.init.zeros_(self.router_delta.bias)
        self.register_buffer("region_prior", torch.tensor([0.8, 0.2, 0.5, 0.5]))

    def forward(self, eeg, channel_mask, regions):
        batch, channels, samples = eeg.shape
        if channel_mask.shape != (batch, channels) or regions.shape != (batch, channels):
            raise ValueError("EEG, mask and region metadata must align")
        if not torch.all(channel_mask.sum(dim=1) > 0):
            raise ValueError("Every cap must contain at least one valid channel")
        mask = channel_mask.bool()
        valid_input = torch.where(mask[..., None], eeg, torch.zeros_like(eeg))
        center = valid_input.mean(dim=-1, keepdim=True)
        scale = valid_input.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized = (valid_input - center) / scale
        encoded = self.temporal(normalized.reshape(batch * channels, 1, samples))
        tokens = encoded.shape[-1]
        encoded = encoded.reshape(batch, channels, self.features, tokens).permute(0, 1, 3, 2)
        encoded = self.normalization(encoded)
        weights = mask[..., None, None].to(encoded.dtype)
        count = weights.sum(dim=1).clamp_min(1)
        mean = (encoded * weights).sum(dim=1) / count
        variance = ((encoded - mean[:, None]) ** 2 * weights).sum(dim=1) / count
        context = self.context(torch.cat([mean, torch.sqrt(variance + 1e-6)], dim=-1))
        repeated = context[:, None].expand(batch, channels, tokens, self.features)
        sequence = torch.cat([encoded, repeated], dim=-1).reshape(batch * channels, tokens, self.features * 2)
        recurrent, _ = self.gru(sequence)
        temporal = self.temporal_residual(recurrent)
        spatial = self.context_residual(torch.cat([recurrent, repeated.reshape(batch * channels, tokens, self.features)], dim=-1))
        prior = self.region_prior[regions].reshape(batch * channels, 1, 1)
        router = torch.sigmoid(torch.logit(prior) + self.router_delta(recurrent))
        gate = torch.sigmoid(self.gate(recurrent))
        residual = gate * (router * temporal + (1 - router) * spatial)
        residual = F.interpolate(residual.transpose(1, 2), size=samples, mode="linear", align_corners=False)
        residual = residual.reshape(batch, channels, samples) * scale
        residual = torch.where(mask[..., None], residual, torch.zeros_like(residual))
        return {"cleaned": eeg - residual, "artifact": residual,
                "gate": gate.reshape(batch, channels, tokens),
                "router": router.reshape(batch, channels, tokens)}


def reconstruction_loss(prediction, target, mask):
    weights = mask[..., None].to(prediction.dtype)
    denominator = (weights.sum() * prediction.shape[-1]).clamp_min(1)
    mse = ((prediction - target) ** 2 * weights).sum() / denominator
    derivative_error = torch.diff(prediction, dim=-1) - torch.diff(target, dim=-1)
    derivative = (derivative_error ** 2 * weights).sum() / denominator
    predicted_spectrum = torch.fft.rfft(prediction, dim=-1).abs()
    target_spectrum = torch.fft.rfft(target, dim=-1).abs()
    spectral = ((torch.log1p(predicted_spectrum) - torch.log1p(target_spectrum)) ** 2 * weights).mean()
    return mse + 0.1 * derivative + 0.05 * spectral
