"""Kaggle-only numerical tests for actual-SNR optimization and model routing."""
import pytest
import torch
from eog_vmd_fcm_bgru.snr_student import (masked_snr_db, snr_target_loss,
    identity_penalty, VMDSpatialStudent, EOGGainStudent)


def test_snr_has_known_db_gradient_and_is_scale_sensitive():
    target = torch.ones(2, 3, 64)
    mask = torch.ones(2, 3)
    estimate = (target * 1.1).requires_grad_()
    torch.testing.assert_close(masked_snr_db(estimate, target, mask), torch.full((2,), 20.0))
    torch.testing.assert_close(masked_snr_db(estimate * 1e-6, target * 1e-6, mask), torch.full((2,), 20.0))
    loss = snr_target_loss(estimate, target, mask)
    loss.backward()
    assert torch.isfinite(estimate.grad).all() and (estimate.grad > 0).all()
    assert (masked_snr_db(target * 2, target, mask) < 1).all()
    with pytest.raises(ValueError, match="positive"):
        masked_snr_db(target, target * 0, mask)


def test_snr_and_identity_ignore_padded_garbage():
    target = torch.ones(1, 2, 32)
    estimate = target + 0.2
    expected = masked_snr_db(estimate, target, torch.ones(1, 2))
    padded_target = torch.cat([target, torch.zeros(1, 1, 32)], dim=1)
    padded_estimate = torch.cat([estimate, torch.full((1, 1, 32), 1e6)], dim=1)
    mask = torch.tensor([[1, 1, 0]])
    torch.testing.assert_close(masked_snr_db(padded_estimate, padded_target, mask), expected)
    assert identity_penalty(target * 0, target, torch.ones(1, 2)) == 0


@pytest.mark.parametrize("constructor", [VMDSpatialStudent, EOGGainStudent])
def test_student_permutation_padding_identity_and_checkpoint(constructor, tmp_path):
    torch.manual_seed(17)
    model = constructor().eval()
    eeg, modes, eyes = torch.randn(2, 4, 128), torch.randn(2, 4, 3, 128), torch.randn(2, 2, 128)
    mask, regions = torch.ones(2, 4), torch.tensor([[0, 1, 3, 3]]).expand(2, -1)
    torch.testing.assert_close(model(eeg, modes, eyes, mask, regions)["cleaned"], eeg)
    # Exercise nonzero correction, not only the zero-initialized identity.
    with torch.no_grad():
        head = model.head if hasattr(model, "head") else model.gains[-1]
        head.weight.normal_(std=0.01)
    expected = model(eeg, modes, eyes, mask, regions)["artifact"]
    order = torch.tensor([2, 0, 3, 1])
    changed = model(eeg[:, order], modes[:, order], eyes, mask[:, order], regions[:, order])["artifact"]
    torch.testing.assert_close(changed, expected[:, order], atol=1e-5, rtol=1e-4)
    padded_eeg = torch.cat([eeg, torch.full((2, 1, 128), 1e6)], dim=1)
    padded_modes = torch.cat([modes, torch.full((2, 1, 3, 128), 1e6)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(2, 1)], dim=1)
    padded_regions = torch.cat([regions, torch.full((2, 1), 3)], dim=1)
    actual = model(padded_eeg, padded_modes, eyes, padded_mask, padded_regions)["artifact"]
    torch.testing.assert_close(actual[:, :4], expected, atol=1e-5, rtol=1e-4)
    assert not actual[:, 4].any()
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = constructor().eval()
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(reloaded(eeg, modes, eyes, mask, regions)["artifact"], expected)
