"""Preservation-constrained losses for the EEG-only regional student."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class LossV2Config:
    paired_weight: float = 1.0
    identity_weight: float = 0.25
    spectral_weight: float = 0.05
    covariance_weight: float = 0.05
    teacher_weight: float = 0.10
    event_weight: float = 0.05
    snr_shortfall_weight: float = 0.05
    snr_target_db: float = 20.0
    stft_n_fft: int = 128
    stft_hop_length: int = 64


def loss_terms_to_jsonable(terms: dict[str, Tensor]) -> dict[str, float]:
    """Detach scalar training terms for run manifests and JSON summaries."""
    return {name: float(value.detach().cpu()) for name, value in terms.items()}


def _valid_values(values: Tensor, mask: Tensor) -> Tensor:
    return torch.where(mask.bool()[..., None], torch.nan_to_num(values), torch.zeros_like(values))


def _normalized_error(prediction: Tensor, target: Tensor, mask: Tensor, weights: Tensor | None = None) -> Tensor:
    valid = mask.bool()[..., None].to(prediction.dtype)
    if weights is not None:
        if weights.shape != prediction.shape:
            raise ValueError("Burden weights must align with [B,C,T] predictions")
        valid = valid * torch.clamp(torch.nan_to_num(weights), min=0)
    error = (_valid_values(prediction - target, mask).square() * valid).sum(dim=(1, 2))
    energy = (_valid_values(target, mask).square() * valid).sum(dim=(1, 2)).clamp_min(1e-12)
    return (error / energy).mean()


def amplitude_sensitive_snr_db(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    error = _valid_values(prediction - target, mask).square().sum(dim=(1, 2))
    energy = _valid_values(target, mask).square().sum(dim=(1, 2)).clamp_min(1e-12)
    return -10.0 * torch.log10((error / energy).clamp_min(1e-12))


def _spectral_stft_loss(prediction: Tensor, target: Tensor, mask: Tensor, n_fft: int, hop: int) -> Tensor:
    samples = prediction.shape[-1]
    if samples < 8:
        return prediction.new_zeros(())
    n_fft = min(n_fft, samples)
    hop = min(hop, max(1, n_fft // 2))
    valid = mask.bool().reshape(-1)
    source = _valid_values(prediction, mask).reshape(-1, samples)[valid]
    reference = _valid_values(target, mask).reshape(-1, samples)[valid]
    if source.numel() == 0:
        return prediction.new_zeros(())
    window = torch.hann_window(n_fft, dtype=prediction.dtype, device=prediction.device)
    source_stft = torch.stft(source, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    target_stft = torch.stft(reference, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    return F.smooth_l1_loss(torch.log1p(source_stft.abs()), torch.log1p(target_stft.abs()))


def _covariance_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    values = _valid_values(prediction, mask)
    reference = _valid_values(target, mask)
    valid = mask.bool()
    total = prediction.new_zeros(())
    used = 0
    for batch in range(prediction.shape[0]):
        keep = valid[batch]
        if int(keep.sum()) < 2:
            continue
        estimate = values[batch, keep]
        clean = reference[batch, keep]
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
        clean = clean - clean.mean(dim=-1, keepdim=True)
        estimated_covariance = estimate @ estimate.transpose(0, 1) / max(1, estimate.shape[-1] - 1)
        clean_covariance = clean @ clean.transpose(0, 1) / max(1, clean.shape[-1] - 1)
        total = total + F.smooth_l1_loss(estimated_covariance, clean_covariance)
        used += 1
    return total / max(used, 1)


def regional_student_loss(
    output: dict[str, Tensor],
    dirty: Tensor,
    mask: Tensor,
    paired_clean: Tensor | None = None,
    clean_input: Tensor | None = None,
    teacher_artifact: Tensor | None = None,
    teacher_weight: Tensor | None = None,
    teacher_oof: Tensor | None = None,
    event_targets: Tensor | None = None,
    event_mask: Tensor | None = None,
    burden_weight: Tensor | None = None,
    config: LossV2Config = LossV2Config(),
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return total loss and every logged term.

    Teacher supervision is accepted only with an explicit out-of-fold boolean
    marker.  Runtime inference never receives any of these training tensors.
    """
    if output["cleaned"].shape != dirty.shape or output["artifact"].shape != dirty.shape:
        raise ValueError("Student output and dirty EEG must align")
    zero = dirty.new_zeros(())
    terms = {name: zero for name in ("paired", "identity", "spectral", "covariance", "teacher", "event", "snr_shortfall")}
    if paired_clean is not None:
        if paired_clean.shape != dirty.shape:
            raise ValueError("Paired clean EEG must align with dirty EEG")
        terms["paired"] = _normalized_error(output["cleaned"], paired_clean, mask, burden_weight)
        terms["spectral"] = _spectral_stft_loss(output["cleaned"], paired_clean, mask, config.stft_n_fft, config.stft_hop_length)
        terms["covariance"] = _covariance_loss(output["cleaned"], paired_clean, mask)
        snr = amplitude_sensitive_snr_db(output["cleaned"], paired_clean, mask)
        terms["snr_shortfall"] = F.softplus((config.snr_target_db - snr) / 5.0).mean()
    if clean_input is not None:
        if clean_input.shape != dirty.shape:
            raise ValueError("Clean-input preservation EEG must align with dirty EEG")
        # The denominator is the clean input's own energy, not the dirty scale.
        clean_energy = _valid_values(clean_input, mask).square().sum(dim=(1, 2)).clamp_min(1e-12)
        artifact_energy = _valid_values(output["artifact"], mask).square().sum(dim=(1, 2))
        terms["identity"] = (artifact_energy / clean_energy).mean()
    if teacher_artifact is not None:
        if teacher_artifact.shape != dirty.shape or teacher_weight is None or teacher_oof is None:
            raise ValueError("Teacher artifact, confidence weight, and OOF marker are jointly required")
        if teacher_weight.shape not in (dirty.shape, dirty.shape[:2]):
            raise ValueError("Teacher weight must have shape [B,C,T] or [B,C]")
        if teacher_weight.ndim == 2:
            teacher_weight = teacher_weight[..., None].expand_as(dirty)
        if teacher_oof.shape not in (dirty.shape[:1], dirty.shape[:2]):
            raise ValueError("Teacher OOF marker must have shape [B] or [B,C]")
        oof = teacher_oof[:, None] if teacher_oof.ndim == 1 else teacher_oof
        if not bool(oof.bool().all()):
            raise ValueError("Teacher distillation requires out-of-fold targets for every supplied channel")
        terms["teacher"] = _normalized_error(output["artifact"], teacher_artifact, mask, teacher_weight)
    if event_targets is not None:
        logits = output["event_logits"]
        if event_targets.shape != logits.shape:
            raise ValueError("Event targets must align with [B,C,T,event_classes]")
        if event_mask is None or event_mask.shape != logits.shape[:-1]:
            raise ValueError("Event mask [B,C,T] is required with event targets")
        valid = mask.bool()[..., None] & event_mask.bool()
        error = F.binary_cross_entropy_with_logits(logits, event_targets.to(logits.dtype), reduction="none").mean(dim=-1)
        terms["event"] = (error * valid).sum() / valid.sum().clamp_min(1)
    total = (
        config.paired_weight * terms["paired"] + config.identity_weight * terms["identity"] +
        config.spectral_weight * terms["spectral"] + config.covariance_weight * terms["covariance"] +
        config.teacher_weight * terms["teacher"] + config.event_weight * terms["event"] +
        config.snr_shortfall_weight * terms["snr_shortfall"]
    )
    return total, {**terms, "total": total}
