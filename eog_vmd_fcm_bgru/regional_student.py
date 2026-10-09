"""EEG-only regional residual student for offline EOG-artifact removal.

The model deliberately accepts no EOG, VMD, ICA, FCM, clean-target, or teacher
tensor at inference.  Those signals belong to the offline teacher/training
pipeline.  Channel identities are supplied through metadata tensors so an EEG
row index is never treated as an electrode name.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


REGION_COUNT = 4
HEMISPHERE_COUNT = 4


@dataclass(frozen=True)
class RegionalStudentConfig:
    """Architecture contract for :class:`RegionalEEGStudentV2`."""

    features: int = 32
    hidden: int = 32
    kernel_size: int = 7
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    coordinate_dim: int = 3
    event_classes: int = 4
    bidirectional: bool = True


class DepthwiseResidualBlock(nn.Module):
    """Full-rate depthwise temporal residual block."""

    def __init__(self, features: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            features, features, kernel_size, padding=padding,
            dilation=dilation, groups=features, bias=False,
        )
        self.norm = nn.GroupNorm(4, features)
        self.pointwise = nn.Conv1d(features, features, 1)

    def forward(self, values: Tensor) -> Tensor:
        return values + self.pointwise(F.gelu(self.norm(self.depthwise(values))))


def _validate_metadata(
    eeg: Tensor,
    channel_mask: Tensor,
    regions: Tensor,
    hemispheres: Tensor | None,
    coordinates: Tensor | None,
    coordinate_mask: Tensor | None,
    coordinate_dim: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if eeg.ndim != 3:
        raise ValueError("Expected EEG [B,C,T]")
    batch, channels, _ = eeg.shape
    if channel_mask.shape != (batch, channels) or regions.shape != (batch, channels):
        raise ValueError("EEG, channel mask and region metadata must align")
    if not torch.all((channel_mask == 0) | (channel_mask == 1)):
        raise ValueError("Channel mask must be binary")
    if not torch.all(channel_mask.sum(dim=1) > 0):
        raise ValueError("Every cap must contain at least one valid channel")
    if not torch.all((regions >= 0) & (regions < REGION_COUNT)):
        raise ValueError("Regions must use frontal=0, posterior=1, central=2, unknown=3")
    if hemispheres is None:
        hemispheres = torch.full_like(regions, 3)
    if hemispheres.shape != (batch, channels) or not torch.all((hemispheres >= 0) & (hemispheres < HEMISPHERE_COUNT)):
        raise ValueError("Hemispheres must use left=0, right=1, midline=2, unknown=3")
    if coordinates is None:
        coordinates = eeg.new_zeros((batch, channels, coordinate_dim))
        coordinate_mask = torch.zeros((batch, channels), dtype=torch.bool, device=eeg.device)
    else:
        if coordinates.shape != (batch, channels, coordinate_dim):
            raise ValueError("Coordinates must have shape [B,C,coordinate_dim]")
        if coordinate_mask is None:
            coordinate_mask = torch.isfinite(coordinates).all(dim=-1)
        if coordinate_mask.shape != (batch, channels):
            raise ValueError("Coordinate mask must align with EEG channels")
        coordinates = torch.nan_to_num(coordinates)
    return channel_mask.bool(), regions.long(), hemispheres.long(), coordinates, coordinate_mask.bool()


def _masked_pool(values: Tensor, selector: Tensor, fallback: Tensor) -> tuple[Tensor, Tensor]:
    """Pool [B,C,D,T] with a [B,C] selector and explicit fallback."""
    weights = selector[:, :, None, None].to(values.dtype)
    count = weights.sum(dim=1)
    available = count[:, 0, 0] > 0
    pooled = (values * weights).sum(dim=1) / count.clamp_min(1.0)
    pooled = torch.where(available[:, None, None], pooled, fallback)
    return pooled, available


class RegionalEEGStudentV2(nn.Module):
    """Shared full-rate TCN + offline BiGRU with signed frontal context.

    Region and hemisphere values are metadata, not predictions.  If frontal
    side metadata is absent, the signed lateral context is exactly zero and its
    availability flag is false; the model then uses the shared fallback path.
    """

    architecture_id = "regional_eeg_student_v2"

    def __init__(self, config: RegionalStudentConfig | None = None, **overrides):
        super().__init__()
        if config is not None and overrides:
            raise ValueError("Pass either RegionalStudentConfig or keyword architecture values")
        self.config = config or RegionalStudentConfig(**overrides)
        cfg = self.config
        if cfg.features % 4:
            raise ValueError("features must be divisible by four for GroupNorm")
        self.input_projection = nn.Conv1d(1, cfg.features, 1)
        self.blocks = nn.Sequential(*[
            DepthwiseResidualBlock(cfg.features, cfg.kernel_size, dilation)
            for dilation in cfg.dilations
        ])
        self.region_embedding = nn.Embedding(REGION_COUNT, 4)
        self.hemisphere_embedding = nn.Embedding(HEMISPHERE_COUNT, 4)
        self.coordinate_projection = nn.Linear(cfg.coordinate_dim + 1, cfg.features)
        self.metadata_projection = nn.Linear(cfg.features + 8, cfg.features)
        # own temporal feature + global + routed region + common frontal + signed lateral + metadata
        self.fusion = nn.Linear(cfg.features * 6, cfg.features)
        self.gru = nn.GRU(cfg.features, cfg.hidden, batch_first=True, bidirectional=cfg.bidirectional)
        recurrent_features = cfg.hidden * (2 if cfg.bidirectional else 1)
        self.frontal_head = nn.Linear(recurrent_features, 1)
        self.posterior_head = nn.Linear(recurrent_features, 1)
        self.shared_head = nn.Linear(recurrent_features, 1)
        self.gate_head = nn.Linear(recurrent_features, 1)
        self.event_head = nn.Linear(recurrent_features, cfg.event_classes)
        self.quality_head = nn.Linear(recurrent_features, 1)
        for head in (self.frontal_head, self.posterior_head, self.shared_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def architecture_config(self) -> dict:
        return asdict(self.config)

    def _regional_context(self, encoded: Tensor, mask: Tensor, regions: Tensor, hemispheres: Tensor):
        global_context, _ = _masked_pool(encoded, mask, torch.zeros_like(encoded[:, 0]))
        frontal_context, frontal_available = _masked_pool(encoded, mask & (regions == 0), global_context)
        posterior_context, posterior_available = _masked_pool(encoded, mask & (regions == 1), global_context)
        central_context, central_available = _masked_pool(encoded, mask & (regions == 2), global_context)
        # Unknown metadata never creates a guessed anatomical subregion: it
        # always receives the shared all-channel context.
        unknown_context = global_context
        unknown_available = (mask & (regions == 3)).any(dim=1)
        region_stack = torch.stack([frontal_context, posterior_context, central_context, unknown_context], dim=1)
        routing = regions[:, :, None, None, None].expand(-1, -1, 1, encoded.shape[2], encoded.shape[3])
        routed = region_stack[:, None].expand(-1, encoded.shape[1], -1, -1, -1).gather(2, routing).squeeze(2)

        left_context, left_available = _masked_pool(encoded, mask & (regions == 0) & (hemispheres == 0), frontal_context)
        right_context, right_available = _masked_pool(encoded, mask & (regions == 0) & (hemispheres == 1), frontal_context)
        bilateral_available = left_available & right_available
        common = 0.5 * (left_context + right_context)
        lateral = right_context - left_context
        common = torch.where((left_available | right_available)[:, None, None], common, torch.zeros_like(common))
        lateral = torch.where(bilateral_available[:, None, None], lateral, torch.zeros_like(lateral))
        context_available = torch.stack([
            frontal_available, posterior_available, central_available, unknown_available,
            left_available, right_available, bilateral_available,
        ], dim=-1)
        return global_context, routed, common, lateral, context_available

    def forward(
        self,
        eeg: Tensor,
        channel_mask: Tensor,
        regions: Tensor,
        hemispheres: Tensor | None = None,
        coordinates: Tensor | None = None,
        coordinate_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        mask, regions, hemispheres, coordinates, coordinate_mask = _validate_metadata(
            eeg, channel_mask, regions, hemispheres, coordinates, coordinate_mask, self.config.coordinate_dim,
        )
        batch, channels, samples = eeg.shape
        safe_eeg = torch.nan_to_num(eeg)
        values = torch.where(mask[..., None], safe_eeg, torch.zeros_like(safe_eeg))
        center = values.mean(dim=-1, keepdim=True)
        scale = values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized = (values - center) / scale
        encoded = self.blocks(self.input_projection(normalized.reshape(batch * channels, 1, samples)))
        encoded = encoded.reshape(batch, channels, self.config.features, samples)
        global_context, routed_context, frontal_common, frontal_lateral, availability = self._regional_context(
            encoded, mask, regions, hemispheres,
        )
        coordinate_feature = torch.cat([coordinates, coordinate_mask[..., None].to(coordinates.dtype)], dim=-1)
        metadata = self.metadata_projection(torch.cat([
            self.coordinate_projection(coordinate_feature), self.region_embedding(regions), self.hemisphere_embedding(hemispheres),
        ], dim=-1))
        metadata = metadata[:, :, :, None].expand(-1, -1, -1, samples)
        fused = torch.cat([
            encoded,
            global_context[:, None].expand(-1, channels, -1, -1),
            routed_context,
            frontal_common[:, None].expand(-1, channels, -1, -1),
            frontal_lateral[:, None].expand(-1, channels, -1, -1),
            metadata,
        ], dim=2)
        fused = F.gelu(self.fusion(fused.permute(0, 1, 3, 2))).reshape(batch * channels, samples, self.config.features)
        recurrent, _ = self.gru(fused)
        frontal_prediction = self.frontal_head(recurrent)
        posterior_prediction = self.posterior_head(recurrent)
        shared_prediction = self.shared_head(recurrent)
        is_frontal = (regions.reshape(-1) == 0)[:, None, None]
        is_posterior = (regions.reshape(-1) == 1)[:, None, None]
        raw_artifact = torch.where(is_frontal, frontal_prediction, torch.where(is_posterior, posterior_prediction, shared_prediction))
        gate = torch.sigmoid(self.gate_head(recurrent))
        artifact = (gate * raw_artifact).reshape(batch, channels, samples) * scale
        artifact = torch.where(mask[..., None], artifact, torch.zeros_like(artifact))
        cleaned = eeg - artifact
        event_logits = self.event_head(recurrent).reshape(batch, channels, samples, self.config.event_classes)
        quality_logit = self.quality_head(recurrent).mean(dim=1).reshape(batch, channels)
        gate = gate.reshape(batch, channels, samples)
        return {
            "cleaned": cleaned,
            "artifact": artifact,
            "intervention_gate": torch.where(mask[..., None], gate, torch.zeros_like(gate)),
            "event_logits": torch.where(mask[..., None, None], event_logits, torch.zeros_like(event_logits)),
            "quality_logit": torch.where(mask, quality_logit, torch.zeros_like(quality_logit)),
            "context_available": availability,
            "coordinate_available": coordinate_mask & mask,
        }
