"""Frontal scope, entropy, correction and source-isolation checks on Kaggle."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vmd_eog.frontal_campaign import validate_policy
from vmd_eog.frontal_methods import (frontal_indices, fuzzy_entropy,
    estimate_components, reference_intervals, FrontalCorrector)
from vmd_eog.frontal_replication import candidates, fit_detectors, summary, detector_features, grouped_selection, validate_cached_reconstruction
from vmd_eog.frontal_native import validate_record_selection
from vmd_eog.frontal_evaluation import validate_evaluation_sources
from vmd_eog.frontal_review import controlled_gates, strict_mean
from vmd_eog.frontal_evidence import check_metric_parity


def configuration():
    return json.loads((Path(__file__).parents[1] / "configs/frontal_replication.json").read_text())


def test_frozen_transfer_only_corrects_verified_frontal_and_preserves_permutation():
    from vmd_eog.frontal_frozen_transfer import apply_frontal
    eeg = np.arange(4*64, dtype=np.float32).reshape(4,64) + 1
    names = np.array(["F3", "P3", "F4", "Cz"])
    regions = np.array([0,1,0,2])
    refs = np.ones((2,64), dtype=np.float32)
    bundles = {index: (np.ones((3,64)), np.zeros(64), {}) for index in (0,2)}
    def model(values, modes, eyes, mask, features):
        assert values.shape[1] == 2
        assert (features == 3).all()
        return {"artifact": values*.25}
    cleaned, artifact, _ = apply_frontal(model,eeg,refs,names,regions,np.ones(4,bool),bundles)
    np.testing.assert_array_equal(cleaned[[1,3]],eeg[[1,3]])
    np.testing.assert_allclose(cleaned[[0,2]],eeg[[0,2]]*.75)
    order = np.array([2,3,0,1])
    changed = {local: bundles[int(original)] for local,original in enumerate(order) if int(original) in bundles}
    result, _, _ = apply_frontal(model,eeg[order],refs,names[order],regions[order],np.ones(4,bool),changed)
    np.testing.assert_array_equal(result,cleaned[order])
    np.testing.assert_array_equal(eeg-cleaned,artifact)


def test_frozen_transfer_failed_decomposition_is_scored_passthrough():
    from vmd_eog.frontal_frozen_transfer import apply_frontal
    values = np.ones((1,64),dtype=np.float32)
    def model(eeg,*args):
        return {"artifact":eeg*.5}
    modes = {0:(np.ones((3,64)),np.zeros(64),{"hit_iteration_limit":True})}
    cleaned, artifact, diagnostics = apply_frontal(model,values,np.ones((2,64)),np.array(["Fz"]),np.array([0]),np.ones(1,bool),modes)
    np.testing.assert_array_equal(cleaned,values)
    assert not artifact.any() and diagnostics[0]["failure"] == "scored_passthrough"
    with pytest.raises(ValueError,match="exactly"):
        apply_frontal(model,values,np.ones((2,64)),np.array(["Fz"]),np.array([0]),np.ones(1,bool),{})


def test_gain_unit_adapter_converts_all_inputs_and_restores_output_units():
    from vmd_eog.frontal_frozen_transfer import apply_frontal
    values = np.full((1,64),2e-5)
    references = np.full((2,64),3e-5)
    bundles = {0:(np.full((3,64),4e-6),np.zeros(64),{})}
    def model(eeg,modes,eyes,*args):
        np.testing.assert_allclose(eeg.numpy(),20.,rtol=1e-6)
        np.testing.assert_allclose(modes.numpy(),4.,rtol=1e-6)
        np.testing.assert_allclose(eyes.numpy(),30.,rtol=1e-6)
        return {"artifact":eeg*.25}
    cleaned, artifact, _ = apply_frontal(model,values,references,
        np.array(["Fz"]),np.array([0]),np.ones(1,bool),bundles,unit_scale=1e6)
    np.testing.assert_allclose(cleaned,values*.75,rtol=1e-6)
    np.testing.assert_allclose(artifact,values*.25,rtol=1e-6)
    with pytest.raises(ValueError,match="positive"):
        apply_frontal(model,values,references,np.array(["Fz"]),np.array([0]),np.ones(1,bool),bundles,unit_scale=0.)


def test_corrector_reconstruction_uses_same_float64_contract_as_cache_reader():
    modes = np.tile(np.array([[1.], [.1], [0.]],dtype=np.float32),(1,64))
    residual = np.full(64,-1.1,dtype=np.float32)
    original = modes.sum(0,dtype=np.float64) + residual.astype(np.float64)
    spec = dict(family="vmd_projected",K=3,alpha=2000,trigger=.99,
                component_threshold=.2,kind="projected",strength=1.,lags=[0],penalty=.01)
    corrector = FrontalCorrector(spec,configuration())
    result, artifact, _ = corrector.denoise(original[None],np.zeros((2,64)),np.zeros((2,64)),
        ["Fz"],[0],mode_bundles={0:(modes,residual,{})})
    np.testing.assert_array_equal(result,original[None])
    assert not artifact.any()


def test_calibration_hybrid_corrects_signed_fields_and_preserves_units_regions():
    from vmd_eog.frontal_calibration_transfer import calibration_transfer
    from vmd_eog.reference import lag_matrix
    generator = np.random.default_rng(78)
    refs = generator.normal(size=(2,256))
    cal_refs = generator.normal(size=(2,2048))
    matrix = np.array([[.8,-.3,.4,.1,-.2,.2],[-.6,.2,-.4,.1,.3,-.2],[0.,0.,0.,0.,0.,0.]])
    design, valid = lag_matrix(refs,(-10,0,10))
    cal_design,_ = lag_matrix(cal_refs,(-10,0,10))
    field = matrix@design
    field[:,~valid] = 0
    original = field + generator.normal(size=field.shape)*.005
    calibration = dict(eeg=matrix@cal_design,references=cal_refs,boundaries=np.array([1024]))
    names, regions, mask = np.array(["F3","F4","P3"]),np.array([0,0,1]),np.ones(3,bool)
    bundles = {index:(np.stack([field[index],np.zeros(256),np.zeros(256)]),original[index]-field[index],{}) for index in (0,1)}
    corrected, artifact, detail = calibration_transfer(original,refs,calibration,names,regions,mask,bundles)
    assert all(row["triggered"] for row in detail)
    # Native trials persist these decisions before marking a record complete.
    restored = json.loads(json.dumps(detail,allow_nan=False))
    assert all(type(row["triggered"]) is bool for row in restored)
    assert np.linalg.norm(corrected[:2]) < .05*np.linalg.norm(original[:2])
    np.testing.assert_array_equal(corrected[2],original[2])
    scaled_cal = {**calibration,"eeg":calibration["eeg"]*1e-6,"references":cal_refs*1e-6}
    scaled_modes = {index:(modes*1e-6,residual*1e-6,info) for index,(modes,residual,info) in bundles.items()}
    scaled,_,_ = calibration_transfer(original*1e-6,refs*1e-6,scaled_cal,names,regions,mask,scaled_modes)
    np.testing.assert_allclose(scaled,corrected*1e-6,rtol=1e-8,atol=1e-12)
    np.testing.assert_allclose(original-corrected,artifact)
    zero_refs = np.zeros_like(refs)
    clean,clean_artifact,_ = calibration_transfer(original,zero_refs,calibration,names,regions,mask)
    np.testing.assert_array_equal(clean,original)
    assert not clean_artifact.any()


def test_native_calibration_requires_both_matched_controlled_gates(tmp_path,monkeypatch):
    from vmd_eog import frontal_native
    monkeypatch.setattr(frontal_native,"verify_parent",lambda *args:None)
    summary = dict(calibration_hybrid=True,scope="verified_frontal_only",
        reserved_confirmation_opened=False,autovmd_executed=False,K=3,alpha=2000)
    (tmp_path/"frontal_transfer_summary.json").write_text(json.dumps(summary))
    gates = [dict(arm="calibration_raw_gain",controlled_development_gates_pass=True),
             dict(arm="K3_calibration_mode_gain",controlled_development_gates_pass=False)]
    (tmp_path/"frontal_controlled_gates.json").write_text(json.dumps(gates))
    with pytest.raises(ValueError,match="passed controlled gates"):
        frontal_native.calibration_parent(tmp_path)
    gates[1]["controlled_development_gates_pass"] = True
    (tmp_path/"frontal_controlled_gates.json").write_text(json.dumps(gates))
    assert frontal_native.calibration_parent(tmp_path) == tmp_path


def test_fixed_gain_axis_support_and_zero_calibration_offset_contracts():
    import torch
    from vmd_eog.frontal_frozen_transfer import gain_adapter
    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.threshold_logit = torch.nn.Parameter(torch.tensor(0.))
        def correction_gains(self,features,correlations,mask,regions):
            return torch.ones(features.shape[0],features.shape[1],2)
        def forward(self,eeg,modes,references,mask,regions):
            correlations = torch.zeros(eeg.shape[0],eeg.shape[1],1,2)
            correlations[:,:,:,0] = .9
            correlations[:,:,:,1] = .1
            self.correction_gains(eeg[:,:,:1],correlations,mask,regions)
            return {"artifact":torch.zeros_like(eeg),"cleaned":eeg}
    model = gain_adapter(Base,axis_gate=True,anchored=True)
    time = torch.arange(64,dtype=torch.float32)
    references = torch.stack([torch.sin(time/5)+2,torch.cos(time/7)-3])[None]
    eeg = references[:,:1]*.3 + references[:,1:]*.2 + 1
    mask, regions = torch.ones(1,1),torch.full((1,1),3)
    with pytest.raises(ValueError,match="calibration"):
        model(eeg,torch.zeros(1,1,3,64),references,mask,regions)
    model.calibration_mean = references.mean(-1,keepdim=True)
    result = model(eeg,torch.zeros(1,1,3,64),references,mask,regions)
    torch.testing.assert_close(model.latest_gains,torch.tensor([[[1.,0.]]]))
    torch.testing.assert_close(result["artifact"],torch.zeros_like(eeg))
    torch.testing.assert_close(result["cleaned"],eeg)


def test_archived_reconstruction_precision_preserves_vectors_and_detects_damage():
    modes = np.array([[1.], [.1]], dtype=np.float32)
    residual = np.array([-1.1], dtype=np.float32)
    reference = modes.astype(np.float64).sum(0) + residual.astype(np.float64)
    assert not np.allclose(modes.sum(0) + residual, reference, rtol=1e-5, atol=1e-11)
    before_modes, before_residual = modes.copy(), residual.copy()
    validate_cached_reconstruction(modes, residual, reference)
    np.testing.assert_array_equal(modes, before_modes)
    np.testing.assert_array_equal(residual, before_residual)
    assert modes.dtype == np.float32 and residual.dtype == np.float32
    with pytest.raises(ValueError, match="fail reconstruction"):
        validate_cached_reconstruction(modes, residual + .001, reference)


def test_frontal_policy_blocks_autovmd_neural_and_arbitrary_recipes():
    config = configuration()
    validate_policy(config, "frontal-fixture")
    for stage in ("autovmd-cache", "autovmd-search", "router-train", "student-paired", "regional"):
        with pytest.raises(RuntimeError, match="checkpoint"):
            validate_policy(config, stage)
    for key, value in (("autovmd_enabled", True), ("scope", "all")):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_policy(changed, "frontal-fixture")
    changed = copy.deepcopy(config)
    changed["fixed_recipes"][0]["K"] = 5
    with pytest.raises(ValueError, match="fixed recipes"):
        validate_policy(changed, "frontal-fixture")


def test_verified_frontal_names_masks_and_permutation():
    values = np.zeros((5, 64))
    names = np.array(["F3", "FP2", "P3", "FC1", "E1"])
    regions = np.array([0, 0, 1, 2, 3])
    assert frontal_indices(values, names, regions).tolist() == [0, 1]
    assert frontal_indices(values, names, regions, [True, False, True, True, True]).tolist() == [0]
    order = np.array([4, 2, 1, 0, 3])
    indices = frontal_indices(values[order], names[order], regions[order])
    assert set(names[order][indices]) == {"F3", "FP2"}
    with pytest.raises(ValueError, match="verified"):
        frontal_indices(values, names, [0, 0, 0, 2, 3])


def test_fuzzy_entropy_against_scalar_pair_oracle_and_units():
    values = np.array([.1, .4, -.1, .2, .6, -.2, .3, .9, -.5, .4])
    unit = (values - values.mean()) / values.std()
    phi = []
    for dimension in (1, 2):
        vectors = [unit[i:i+dimension] - unit[i:i+dimension].mean()
                   for i in range(len(unit)-dimension+1)]
        weights = [np.exp(-max(abs(first-second))**2/.2)
                   for i, first in enumerate(vectors) for j, second in enumerate(vectors) if i != j]
        phi.append(np.mean(weights))
    expected = np.log(phi[0]) - np.log(phi[1])
    assert fuzzy_entropy(values, block=3) == pytest.approx(expected)
    assert fuzzy_entropy(values * 1e-6) == pytest.approx(expected)
    assert fuzzy_entropy(np.zeros(32)) == 0.


def test_interval_calibration_axes_no_target_and_no_boundary_leakage():
    generator = np.random.default_rng(25)
    calibration = generator.normal(size=(2, 4000))
    references = np.zeros((2, 128))
    references[0, 40:50] = 8.
    references[1, 85:95] = -8.
    support = reference_intervals(references, calibration, fs=200)
    assert support[0, 45] > .99 and support[1, 90] > .99
    assert support[:, :15].sum() == 0 and support[:, -10:].sum() == 0
    np.testing.assert_allclose(reference_intervals(references*1e-6, calibration*1e-6), support)
    np.testing.assert_allclose(reference_intervals(references[::-1], calibration[::-1]), support[::-1])


def test_mode_projection_permutation_and_preserved_residual():
    times = np.arange(1024)/200
    references = np.stack([np.sin(2*np.pi*2*times), np.cos(2*np.pi*3*times)])
    brain = .3*np.sin(2*np.pi*12*times)
    modes = np.stack([2*references[0], references[1], brain])
    residual = .1*np.sin(2*np.pi*35*times)
    original = modes.sum(0)+residual
    spec = dict(kind="projected", component_threshold=.6, lags=[0], penalty=.01, strength=1.)
    artifact, detail = estimate_components(modes, references, spec, references)
    reordered, _ = estimate_components(modes[[2,0,1]], references, spec, references)
    np.testing.assert_allclose(artifact, reordered, atol=1e-12)
    assert set(detail["selected"]) == {0,1}
    # Input-minus-artifact necessarily retains residual, unlike summing only
    # retained VMD modes. Compare its known 35-Hz contribution explicitly.
    np.testing.assert_allclose((original-artifact)-(modes.sum(0)-artifact), residual, atol=1e-12)


def test_fixed_candidate_bank_has_no_all_channel_or_adaptive_vmd():
    specs = candidates(configuration(), "full")
    assert {spec["K"] for spec in specs.values()} == {None, 3, 4}
    assert {spec["alpha"] for spec in specs.values()} == {None, 2000}
    svm = [spec for spec in specs.values() if spec["family"] == "svm_sobi_apen_fixed_adaptation"]
    assert len(svm) == 1 and svm[0]["K"] == 3
    for spec in specs.values():
        if spec["K"] is not None:
            assert spec["comparison_arm"].endswith(f"_K{spec['K']}")
    projected = [spec for spec in specs.values() if spec["family"] == "vmd_projected"]
    assert {tuple(spec["lags"]) for spec in projected} == {(0,), (-10,0,10), (-20,-10,0,10,20)}
    assert {spec["penalty"] for spec in projected} == {.001,.01,.1}


def test_whole_montage_interface_changes_frontal_only_and_has_no_target_argument():
    times = np.arange(1024)/200
    references = np.stack([np.sin(2*np.pi*2*times), np.cos(2*np.pi*3*times)])
    brain = np.sin(2*np.pi*12*times)
    original = np.stack([brain+2*references[0], brain+references[1], brain])
    spec = dict(family="direct", trigger=.4, lags=[0], penalty=.01, strength=1.)
    corrector = FrontalCorrector(spec, configuration())
    cleaned, artifact, diagnostics = corrector.denoise(original, references, references,
        ["F3", "P3", "E1"], [0, 1, 3])
    np.testing.assert_array_equal(cleaned[1:], original[1:])
    np.testing.assert_array_equal(artifact[1:], np.zeros_like(original[1:]))
    assert np.linalg.norm(cleaned[0]-brain) < np.linalg.norm(original[0]-brain)*.1
    assert len(diagnostics) == 1 and diagnostics[0]["channel"] == "F3"


def test_undefined_accuracy_cannot_win_selection_by_silent_mean_exclusion():
    common = dict(candidate_hash="a", recipient="r1", input_snr_db=0.,
        snr_energy_db=12., rrmse_time=.1, pearson_cc=.9, mse=.01,
        snr_improvement_db=12., failure=None, clean_change=0., alpha_db=0.,
        beta_db=0., covariance_error=0.)
    rows = [dict(common, condition="blink"), dict(common, condition="blink", mse=np.nan),
            dict(common, condition="clean", input_snr_db=np.nan)]
    report = summary(pd.DataFrame(rows), {"a": dict(family="direct")}, configuration()["preservation"])
    assert report.iloc[0].partial_preservation_pass
    assert not report.iloc[0].accuracy_coverage_complete
    assert not report.iloc[0].selection_eligible


def test_detector_refuses_donor_crossfold_leakage(tmp_path):
    rows = []
    generator = np.random.default_rng(10)
    for fold in range(2):
        for condition in ("clean", "blink"):
            name = f"{fold}-{condition}.npz"
            np.savez(tmp_path/name, eeg=generator.normal(size=(1,1024)),
                channel_names=np.array(["F3"]), regions=np.array([0]), mask=np.array([True]))
            rows.append(dict(array_path=name, recipient=f"r{fold}", donor="same-donor",
                             calibration=name, partition={"fold":fold}, condition=condition))
    with pytest.raises(ValueError, match="leaks donor"):
        fit_detectors(tmp_path, rows, configuration())


def test_detector_features_do_not_confuse_volts_and_microvolts():
    generator = np.random.default_rng(20)
    signal = generator.normal(size=1024) * 1e-5
    calibration = generator.normal(size=4000) * 2e-5
    first = detector_features(signal, 200, calibration.std())
    second = detector_features(signal * 1e6, 200, calibration.std() * 1e6)
    np.testing.assert_allclose(first, second, rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError, match="calibration"):
        detector_features(signal, 200, 0.)


def test_native_detector_rejects_source_exposure_and_unverified_scaling():
    record = dict(participant="osf:p03", partition=dict(role="development", fold=2))
    evidence = dict(fold=2, training_donors=["osf:p04"],
        amplitude_scaling="unscored same-channel calibration SD; no scoring target scaling")
    validate_record_selection(record, evidence)
    with pytest.raises(ValueError, match="exposed"):
        validate_record_selection(record, {**evidence, "training_donors": ["osf:p03"]})
    with pytest.raises(ValueError, match="normalization"):
        validate_record_selection(record, {**evidence, "amplitude_scaling": None})


def test_external_mode_cache_cannot_smuggle_a_posterior_decomposition():
    samples = np.sin(2*np.pi*3*np.arange(1024)/200)
    original = np.stack([samples, samples])
    spec = dict(family="vmd_projected", K=3, alpha=2000, trigger=.4,
                component_threshold=.2, kind="projected", strength=1., lags=[0], penalty=.01)
    corrector = FrontalCorrector(spec, configuration())
    bundle = (np.stack([samples, np.zeros_like(samples), np.zeros_like(samples)]),
              np.zeros_like(samples), {})
    with pytest.raises(ValueError, match="precisely"):
        corrector.denoise(original, np.stack([samples, samples]), np.stack([samples, samples]),
            ["F3", "P3"], [0, 1], mode_bundles={0: bundle, 1: bundle})


def test_fixed_svm_evaluation_is_not_gated_by_other_outer_fold_predictions(tmp_path):
    specs = {
        "identity": dict(family="identity", comparison_arm="identity"),
        "direct": dict(family="direct", comparison_arm="direct"),
        "svm": dict(family="svm_sobi_apen_fixed_adaptation", K=3,
                    comparison_arm="svm_sobi_apen_fixed_adaptation_K3")}
    records = []
    for fold in range(2):
        for condition in ("blink", "clean"):
            for candidate, spec in specs.items():
                records.append(dict(candidate_hash=candidate, family=spec["family"],
                    example_id=f"{fold}-{condition}", channel="F3", recipient=f"r{fold}",
                    donor=f"d{fold}", fold=fold, condition=condition, input_snr_db=0.,
                    snr_energy_db=12., rrmse_time=.1, pearson_cc=.9, mse=.01,
                    snr_improvement_db=12., failure=None, alpha_db=0., beta_db=0.,
                    covariance_error=0., clean_change=.5 if candidate == "svm" else 0.))
    result = grouped_selection(pd.DataFrame(records), specs, tmp_path, configuration())
    svm = result[result.evaluated_family == "svm_sobi_apen_fixed_adaptation_K3"]
    assert set(svm.candidate_hash) == {"svm"}
    assert (svm[svm.condition == "clean"].clean_change > .01).all()


def test_full_evaluation_preserves_recipient_and_donor_exclusions():
    row = dict(recipient="r2", donor="d2", partition=dict(role="development", fold=2))
    evidence = dict(fold=2, training_recipients=["r1"], training_donors=["d1"])
    validate_evaluation_sources(row, evidence)
    for key, value in (("training_recipients", ["r2"]), ("training_donors", ["d2"])):
        with pytest.raises(ValueError, match="exposed"):
            validate_evaluation_sources(row, {**evidence, key: value})
    with pytest.raises(ValueError, match="reserved"):
        validate_evaluation_sources({**row, "partition": dict(role="confirmation", fold=2)}, evidence)


def test_controlled_gates_keep_bad_records_and_undefined_metrics_visible():
    rows = []
    for arm in ("identity", "direct", "vmd_projected_K3"):
        for number, condition in enumerate(("blink", "clean", "clean")):
            rows.append(dict(evaluated_family=arm, example_id=f"e{number}",
                channel="F3", recipient="r1", donor="d1", condition=condition,
                input_snr_db=0., snr_energy_db=16., mse=.01, rrmse_time=.1,
                pearson_cc=.99, cc_deterioration=0., clean_change=0.,
                alpha_db=0., beta_db=0., covariance_error=0., failure=None))
    frame = pd.DataFrame(rows)
    initial = {row["arm"]: row for row in controlled_gates(frame, configuration())}
    assert initial["vmd_projected_K3"]["controlled_development_gates_pass"]
    assert not initial["vmd_projected_K3"]["regional_integration_authorized"]
    # Its participant mean is only .0075, but one clean record exceeds .01.
    bad = (frame.evaluated_family == "vmd_projected_K3") & (frame.example_id == "e2")
    frame.loc[bad, "clean_change"] = .015
    reviewed = {row["arm"]: row for row in controlled_gates(frame, configuration())}
    assert not reviewed["vmd_projected_K3"]["gates"]["clean_modification"]
    dirty = (frame.evaluated_family == "vmd_projected_K3") & (frame.condition == "blink")
    frame.loc[dirty, "mse"] = np.nan
    reviewed = {row["arm"]: row for row in controlled_gates(frame, configuration())}
    assert not reviewed["vmd_projected_K3"]["gates"]["accuracy_coverage"]
    assert reviewed["vmd_projected_K3"]["mean_output_snr_db"] is None
    assert np.isnan(strict_mean([1., np.nan]))


def test_controlled_gates_refuse_missing_channel_rows():
    rows = []
    for arm in ("identity", "direct"):
        for condition in ("clean", "blink"):
            for channel in ("F3", "F4"):
                rows.append(dict(evaluated_family=arm, example_id=condition,
                    channel=channel, recipient="r1", condition=condition,
                    input_snr_db=0., snr_energy_db=16., mse=.01, rrmse_time=.1,
                    pearson_cc=.99, cc_deterioration=0., clean_change=0.,
                    alpha_db=0., beta_db=0., covariance_error=0., failure=None))
    frame = pd.DataFrame(rows[:-1])
    reviewed = {row["arm"]: row for row in controlled_gates(frame, configuration())}
    assert not reviewed["direct"]["gates"]["row_coverage"]


def test_interface_metric_parity_preserves_units_and_missing_metric_status():
    metrics = dict(mse=np.array([2e-12, 0.]), rrmse_time=np.array([.1, 0.]),
        pearson_cc=np.array([.99, 1.]), snr_energy_db=np.array([20., np.inf]),
        snr_improvement_db=np.array([25., np.nan]))
    saved = pd.DataFrame(dict(channel=["F3", "F4"], mse=[2e-12, 0.],
        rrmse_time=[.1, 0.], pearson_cc=[.99, 1.], snr_energy_db=[20., np.nan],
        snr_improvement_db=[25., np.nan]))
    check_metric_parity(metrics, saved.iloc[::-1], ["F3", "F4"])
    changed = saved.copy()
    changed.loc[0, "mse"] = 2e-10
    with pytest.raises(AssertionError, match="mse"):
        check_metric_parity(metrics, changed, ["F3", "F4"])
    with pytest.raises(ValueError, match="missing"):
        check_metric_parity(metrics, saved.iloc[:1], ["F3", "F4"])
