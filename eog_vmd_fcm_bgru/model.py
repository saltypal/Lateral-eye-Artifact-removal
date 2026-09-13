"""Efficient order-agnostic spatial BiGRU artifact-estimation network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from .config import ModelConfig


class SeparableBlock(nn.Module):
    """Depthwise-separable temporal block with stride-two downsampling."""

    def __init__(self, input_channels: int, output_channels: int, stride: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                input_channels,
                input_channels,
                kernel_size=9,
                stride=stride,
                padding=4,
                groups=input_channels,
                bias=False,
            ),
            nn.BatchNorm1d(input_channels),
            nn.GELU(),
            nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class EfficientSpatialBiGRU(nn.Module):
    """Estimate ocular artifact and use a calibrated gate for subtraction.

    Inputs support a variable number of EEG channels through ``channel_mask``.
    The spatial summary uses channelwise mean, max, and deviation statistics;
    channel order therefore is not treated as electrode identity.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        features: int = 32,
        hidden: int = 48,
        bidirectional: bool = True,
        use_spatial: bool = True,
        use_gate: bool = True,
        use_vmd: bool = True,
        use_fcm: bool = True,
    ):
        super().__init__()
        self.config = config or ModelConfig(
            temporal_features=features,
            gru_hidden=hidden,
            bidirectional=bidirectional,
            use_spatial=use_spatial,
            use_gate=use_gate,
            use_vmd=use_vmd,
            use_fcm=use_fcm,
        )
        features = self.config.temporal_features
        hidden = self.config.gru_hidden

        self.encoder = nn.Sequential(
            SeparableBlock(6, 24),
            SeparableBlock(24, features),
        )
        self.context = nn.Sequential(
            nn.Linear(features * 3, features),
            nn.GELU(),
            nn.Linear(features, features),
        )
        self.channel_attention = nn.Sequential(
            nn.Linear(features * 2, features),
            nn.GELU(),
            nn.Linear(features, features),
            nn.Sigmoid(),
        )
        self.gru = nn.GRU(
            features,
            hidden,
            batch_first=True,
            bidirectional=self.config.bidirectional,
        )
        gru_features = hidden * (2 if self.config.bidirectional else 1)
        self.project = nn.Sequential(nn.Linear(gru_features, 32), nn.GELU())
        self.artifact_head = nn.Sequential(
            nn.Conv1d(32, 24, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(24, 1, kernel_size=1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=1),
        )
        self.burden_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.type_head = nn.Sequential(
            nn.Linear(features, 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )

    def forward(
        self,
        eeg: torch.Tensor,
        vmd_modes: torch.Tensor,
        fcm_membership: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return cleaned signal, artifact estimate, gate, burden, and type logits."""
        batch_size, channel_count, sample_count = eeg.shape
        if not self.config.use_vmd:
            vmd_modes = torch.zeros_like(vmd_modes)
        if not self.config.use_fcm:
            fcm_membership = torch.zeros_like(fcm_membership)

        membership = fcm_membership.unsqueeze(-1).expand(-1, -1, -1, sample_count)
        signal_stack = torch.cat([eeg.unsqueeze(2), vmd_modes], dim=2)
        weighting = torch.cat(
            [torch.ones_like(fcm_membership[:, :, :1]), membership[:, :, :, 0]],
            dim=2,
        )
        signal_stack = signal_stack * (0.5 + 0.5 * weighting.unsqueeze(-1))

        encoded = self.encoder(signal_stack.reshape(batch_size * channel_count, 6, sample_count))
        downsampled_samples = encoded.shape[-1]
        encoded = encoded.reshape(batch_size, channel_count, -1, downsampled_samples).permute(0, 1, 3, 2)

        mask = channel_mask[:, :, None, None]
        denominator = mask.sum(dim=1).clamp_min(1.0)
        channel_mean = (encoded * mask).sum(dim=1) / denominator
        channel_variance = ((encoded - channel_mean[:, None]) ** 2 * mask).sum(dim=1) / denominator
        channel_maximum = encoded.masked_fill(mask == 0, -1e9).amax(dim=1)
        context = self.context(
            torch.cat([channel_mean, channel_maximum, channel_variance.sqrt()], dim=-1)
        )

        if self.config.use_spatial:
            repeated_context = context[:, None].expand_as(encoded)
            attention = self.channel_attention(torch.cat([encoded, repeated_context], dim=-1))
            encoded = encoded + attention * repeated_context

        sequence = encoded.reshape(batch_size * channel_count, downsampled_samples, -1)
        sequence, _ = self.gru(sequence)
        projected = self.project(sequence).transpose(1, 2)

        artifact = functional.interpolate(
            self.artifact_head(projected),
            size=sample_count,
            mode="linear",
            align_corners=False,
        ).reshape(batch_size, channel_count, sample_count)
        gate = torch.sigmoid(
            functional.interpolate(
                self.gate_head(projected),
                size=sample_count,
                mode="linear",
                align_corners=False,
            )
        ).reshape(batch_size, channel_count, sample_count)
        if not self.config.use_gate:
            gate = torch.ones_like(gate)

        pooled = projected.mean(dim=-1).reshape(batch_size, channel_count, -1)
        burden = self.burden_head(pooled).squeeze(-1)
        global_context = (context * channel_mask[:, :, None].mean(dim=1, keepdim=True)).mean(dim=1)
        type_logits = self.type_head(global_context)
        cleaned = eeg - gate * artifact
        return {
            "cleaned": cleaned,
            "artifact": artifact,
            "gate": gate,
            "burden_db": burden,
            "type_logits": type_logits,
        }
