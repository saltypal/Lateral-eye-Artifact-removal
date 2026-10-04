"""Run only on Kaggle for this campaign. Numeric fixtures are not dataset evidence."""
import numpy as np
import torch
import pytest
from scipy.io import savemat
from eog_vmd_fcm_bgru.channel_regions import region_ids, fuse_residuals
from eog_vmd_fcm_bgru.vmd_expert import decompose
from eog_vmd_fcm_bgru.student import SharedChannelStudent
from eog_vmd_fcm_bgru.provenance import read_osf
from eog_vmd_fcm_bgru.spatial_expert import ICAExpert, armbr_correction
from eog_vmd_fcm_bgru.dataset_io import annotated_score_slice, trial_condition

torch.set_num_threads(2)


def test_vmd_odd_length_residual_keeps_alignment():
    t = np.arange(1001) / 200
    x = np.sin(2 * np.pi * 3 * t) + 0.3 * np.sin(2 * np.pi * 12 * t)
    vectors, residual, diagnostics = decompose(x)
    assert vectors.shape == (5, 1001)
    np.testing.assert_allclose(vectors.sum(axis=0) + residual, x, atol=1e-6)
    assert diagnostics["centers_hz"] == sorted(diagnostics["centers_hz"])


def test_fusion_identity_and_permutation():
    rng = np.random.default_rng(42)
    eeg = rng.normal(size=(4, 500))
    zero = np.zeros_like(eeg)
    names = ["Fp1", "FC1", "O1", "unknown"]
    assert region_ids(names).tolist() == [0, 2, 1, 3]
    np.testing.assert_array_equal(fuse_residuals(eeg, zero, zero, names), eeg)
    a, b = rng.normal(size=eeg.shape), rng.normal(size=eeg.shape)
    order = [2, 0, 3, 1]
    expected = fuse_residuals(eeg, a, b, names)[order]
    actual = fuse_residuals(eeg[order], a[order], b[order], [names[i] for i in order])
    np.testing.assert_allclose(actual, expected)


def test_student_padding_and_metadata_permutation_do_not_change_valid_outputs():
    torch.manual_seed(42)
    model = SharedChannelStudent().eval()
    eeg = torch.randn(1, 4, 129)
    mask = torch.ones(1, 4)
    regions = torch.tensor([[0, 1, 2, 3]])
    with torch.no_grad():
        expected = model(eeg, mask, regions)["cleaned"]
        padded = torch.cat([eeg, torch.full((1, 2, 129), 1e6)], dim=1)
        padded_mask = torch.cat([mask, torch.zeros(1, 2)], dim=1)
        padded_regions = torch.cat([regions, torch.full((1, 2), 3)], dim=1)
        actual = model(padded, padded_mask, padded_regions)["cleaned"]
        torch.testing.assert_close(actual[:, :4], expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual[:, 4:], padded[:, 4:], atol=0, rtol=0)
        order = [2, 0, 3, 1]
        reordered = model(eeg[:, order], mask[:, order], regions[:, order])["cleaned"]
        torch.testing.assert_close(reordered, expected[:, order], atol=1e-5, rtol=1e-5)


def test_student_channel_count_does_not_change_parameter_count_and_reload(tmp_path):
    torch.manual_seed(42)
    model = SharedChannelStudent().eval()
    parameter_count = sum(item.numel() for item in model.parameters())
    with torch.no_grad():
        for count in [1, 8, 19, 32, 64, 128]:
            x = torch.randn(1, count, 65)
            output = model(x, torch.ones(1, count), torch.full((1, count), 3))
            assert output["cleaned"].shape == x.shape
            assert torch.isfinite(output["cleaned"]).all()
            assert sum(item.numel() for item in model.parameters()) == parameter_count
        torch.save(model.state_dict(), tmp_path / "state.pt")
        restored = SharedChannelStudent().eval()
        restored.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True))
        torch.testing.assert_close(restored(x, torch.ones(1, count), torch.full((1, count), 3))["cleaned"], output["cleaned"])


@pytest.mark.parametrize("format", ["scipy", "v7.3"])
@pytest.mark.parametrize("storage", ["embedded", "renamed_external"])
@pytest.mark.parametrize("eog_name", ["HEOG", "EOGL2"])
def test_eeglab_annotations_keep_integer_codes_and_do_not_enter_eeg(tmp_path, format, storage, eog_name):
    data = np.zeros((5, 101, 2), dtype=np.float32)
    data[:3] = np.random.default_rng(42).normal(size=(3, 101, 2))
    data[3, :, 0], data[3, :, 1] = 1, 5
    data[4, :, 0], data[4, :, 1] = 2, 4
    locations = [{"labels": name, "type": "EEG"} for name in ["Fp1", "O1", eog_name, "artifactclasses", "label"]]
    path = tmp_path / "study02_p01_prep.set"
    payload = {"EEG": {"nbchan": 5, "pnts": 101, "trials": 2, "srate": 200,
                       "data": data, "chanlocs": locations, "ref": "original"}}
    if storage == "renamed_external":
        data.ravel(order="F").tofile(path.with_suffix(".fdt"))
        payload["EEG"]["data"] = "old_export_filename.fdt"
    if format == "scipy":
        savemat(path, payload)
    else:
        import hdf5storage
        hdf5storage.savemat(str(path), payload, appendmat=False, format="7.3", store_python_metadata=False)
    item = read_osf(path)
    assert item["data"].shape == (2, 5, 101)
    assert item["eeg_indices"] == [0, 1]
    assert item["eog_indices"] == [2]
    assert set(np.unique(item["sample_labels"])) == {1, 5}
    assert item["trial_labels"] == [2, 4]
    assert item["participant"] == "p01"
    np.testing.assert_array_equal(item["data"], np.moveaxis(data, -1, 0))
    if storage == "renamed_external":
        assert item["external_source"]["same_session_companion_fallback"]


def test_short_or_rank_deficient_ica_calibration_is_rejected():
    with pytest.raises(ValueError, match="10 seconds"):
        ICAExpert.fit(np.ones((3, 1000)), np.ones((2, 1000)))
    with pytest.raises(ValueError, match="deficient rank"):
        ICAExpert.fit(np.ones((3, 2000)), np.ones((2, 2000)))


def test_armbr_does_not_guess_frontal_channel_order():
    with pytest.raises(ValueError, match="verified frontopolar"):
        armbr_correction(np.ones((19, 2000)), ["unknown"] * 19, 2000)


def test_osf_smoke_scoring_contains_real_events_after_calibration():
    labels = np.zeros(5000, dtype=int)
    labels[4800:4850] = 5
    trial = {"trial_label": 4, "eeg": np.zeros((3, 5000)), "labels": labels}
    scoring = annotated_score_slice(trial)
    assert trial_condition(trial) == "blink"
    assert scoring.start >= 2000 and scoring.stop <= 5000
    assert scoring.stop - scoring.start == 1024
    assert (labels[scoring] == 5).sum() == 50
    trial["labels"] = np.zeros_like(labels)
    assert annotated_score_slice(trial) is None


def test_rest_spectrum_does_not_concatenate_annotation_gaps():
    from eog_vmd_fcm_bgru.evaluation import ocular_proxies
    raw = np.random.default_rng(42).normal(size=(3, 1024))
    labels = np.zeros(1024, dtype=int)
    labels[:400] = 6
    labels[600:1000] = 6
    metrics = ocular_proxies(raw, raw.copy(), {}, labels)
    assert metrics["rest_samples"] == 800
    assert metrics["rest_spectral_contiguous_samples"] == 0
    assert np.isnan(metrics["rest_raw_alpha_error_db"])
    labels[:] = 6
    metrics = ocular_proxies(raw, raw.copy(), {}, labels)
    assert metrics["rest_spectral_contiguous_samples"] == 1024
    assert metrics["rest_raw_alpha_error_db"] == 0
