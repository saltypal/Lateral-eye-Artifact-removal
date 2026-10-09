"""Kaggle-only contracts for the EEG-only regional deployment student."""
import inspect

import pytest
import torch

from eog_vmd_fcm_bgru.losses_v2 import LossV2Config, regional_student_loss
from eog_vmd_fcm_bgru.model_factory import (
    LEGACY_ARCHITECTURE,
    REGIONAL_ARCHITECTURE,
    architecture_metadata,
    build_deployment_model,
)
from eog_vmd_fcm_bgru.regional_student import RegionalEEGStudentV2, RegionalStudentConfig


def small_model():
    return RegionalEEGStudentV2(RegionalStudentConfig(features=8, hidden=4, dilations=(1, 2))).eval()


def metadata(batch=1, channels=4):
    regions = torch.tensor([[0, 0, 1, 3]]).expand(batch, -1).clone()
    hemispheres = torch.tensor([[0, 1, 3, 3]]).expand(batch, -1).clone()
    coordinates = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]]).expand(batch, -1, -1).clone()
    coordinate_mask = torch.ones(batch, channels, dtype=torch.bool)
    return regions, hemispheres, coordinates, coordinate_mask


def test_v2_zero_initialized_correction_is_identity_and_invalid_channels_pass_through():
    torch.manual_seed(7)
    model = small_model()
    eeg = torch.randn(1, 4, 64)
    eeg[:, 3] = float("nan")
    mask = torch.tensor([[1, 1, 1, 0]])
    regions, hemispheres, coordinates, coordinate_mask = metadata()
    output = model(eeg, mask, regions, hemispheres, coordinates, coordinate_mask)
    torch.testing.assert_close(output["artifact"][:, :3], torch.zeros_like(eeg[:, :3]))
    torch.testing.assert_close(output["cleaned"][:, :3], eeg[:, :3])
    assert torch.isnan(output["cleaned"][:, 3]).all()
    assert not output["artifact"][:, 3].any()
    assert not output["intervention_gate"][:, 3].any()
    assert not output["event_logits"][:, 3].any()


def test_v2_channel_and_metadata_permutation_is_equivariant_after_nonzero_head():
    torch.manual_seed(8)
    model = small_model()
    with torch.no_grad():
        model.frontal_head.bias.fill_(0.1)
        model.posterior_head.bias.fill_(0.2)
        model.shared_head.bias.fill_(0.3)
    eeg = torch.randn(1, 4, 64)
    mask = torch.ones(1, 4)
    regions, hemispheres, coordinates, coordinate_mask = metadata()
    expected = model(eeg, mask, regions, hemispheres, coordinates, coordinate_mask)["artifact"]
    order = torch.tensor([2, 0, 3, 1])
    actual = model(eeg[:, order], mask[:, order], regions[:, order], hemispheres[:, order], coordinates[:, order], coordinate_mask[:, order])["artifact"]
    torch.testing.assert_close(actual, expected[:, order], atol=1e-5, rtol=1e-4)


def test_signed_frontal_context_changes_lateral_sign_but_not_common_context():
    model = small_model()
    encoded = torch.zeros(1, 3, 8, 5)
    encoded[:, 0] = 2.0
    encoded[:, 1] = -1.0
    mask = torch.ones(1, 3, dtype=torch.bool)
    regions = torch.tensor([[0, 0, 1]])
    hemispheres = torch.tensor([[0, 1, 3]])
    _, _, common, lateral, availability = model._regional_context(encoded, mask, regions, hemispheres)
    # Swap side assignments while keeping the physical waveforms fixed.
    _, _, swapped_common, swapped_lateral, _ = model._regional_context(encoded, mask, regions, hemispheres[:, [1, 0, 2]])
    torch.testing.assert_close(common, swapped_common)
    torch.testing.assert_close(lateral, -swapped_lateral)
    assert availability[0, -1]


def test_loss_requires_oof_teacher_and_clean_identity_uses_own_scale():
    eeg = torch.ones(1, 2, 64)
    mask = torch.ones(1, 2)
    output = {"cleaned": eeg.clone(), "artifact": torch.zeros_like(eeg), "event_logits": torch.zeros(1, 2, 64, 4)}
    _, terms = regional_student_loss(output, eeg * 10, mask, paired_clean=eeg, clean_input=eeg, config=LossV2Config())
    assert terms["identity"] == 0
    with pytest.raises(ValueError, match="out-of-fold"):
        regional_student_loss(output, eeg, mask, teacher_artifact=torch.zeros_like(eeg), teacher_weight=torch.ones(1, 2), teacher_oof=torch.tensor([False]))


def test_deployment_registry_excludes_teacher_models_and_v2_forward_has_no_eog_argument():
    model = small_model()
    architecture_id, architecture, schema = architecture_metadata(model)
    assert architecture_id == REGIONAL_ARCHITECTURE and schema == 2
    assert isinstance(build_deployment_model(architecture_id, architecture), RegionalEEGStudentV2)
    assert isinstance(build_deployment_model(LEGACY_ARCHITECTURE, {"features": 24, "hidden": 32}), object)
    assert "eog" not in inspect.signature(RegionalEEGStudentV2.forward).parameters
    with pytest.raises(ValueError, match="Unsupported deployment architecture"):
        build_deployment_model("eog_context_gain_student", {})
