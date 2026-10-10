"""Durable per-channel paper tables computed from unchanged predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .io import atomic_json
from .paper_metrics import paired_channels, native_condition_channels, resting_spectrum


PROTOCOL_ID = "paper-grounded-v1-20261010"


def save_protocol(output):
    root = Path(__file__).resolve().parents[2]
    document = root/"docs/EVALUATION_PROTOCOL.md"
    (output/"EVALUATION_PROTOCOL.md").write_text(document.read_text(encoding="utf-8"), encoding="utf-8")
    atomic_json(output/"paper_evaluation_protocol.json", {
        "id": PROTOCOL_ID,
        "requested_sources": ["VMD papers", "OSF/EEGOAR-Net", "Klados"],
        "paired_primary": ["rrmse_time", "mse", "pearson_cc"],
        "explicit_project_target": "snr_energy_db",
        "native_primary": ["condition-specific zero-lag absolute Pearson", "rest_rmse", "rest_relative_band_power"],
        "supplementary": ["lag-maximized correlation", "joint_eog_r2", "clean_relative_error", "alpha_beta_db", "covariance_error"],
        "psnr_status": "unavailable: no justified 8-bit signal scaling",
        "snr_aggregation": "per channel first; never pooled channel energy",
        "native_unit": "5.12-second cached windows: adaptation, not whole-condition author concatenation",
        "chance_level_status": "not established: cache windows shorter than paper's eight seconds",
        "welch": {"duration_s": 2, "overlap_s": 1, "window": "hann", "detrend": "constant",
                  "relative_power_denominator_hz": [1, 40], "unspecified_choices_are_adaptations": True},
        "source_identity_contract": "unchanged campaign data/calibration/partitions; separate evaluation protocol identity",
    })


def append_table(output, filename, records):
    path = output/filename
    pd.DataFrame(records).to_csv(path, mode="a", index=False, header=not path.exists())


def write_paired(output, metadata, cleaned, target, raw, names, regions):
    metrics = paired_channels(cleaned, target, raw)
    records = []
    for channel, name in enumerate(names):
        record = {**metadata, "channel": str(name), "channel_index": channel,
                  "region_code": int(regions[channel]), "protocol_id": PROTOCOL_ID,
                  "rmse_mse_units": "verified source units only; consult corpus provenance"}
        for key, values in metrics.items():
            value = float(values[channel])
            record[key] = value
            if not np.isfinite(value):
                record[key+"_status"] = "infinite" if np.isinf(value) else "undefined"
            else:
                record[key+"_status"] = "finite"
        records.append(record)
    append_table(output, "paper_paired_channels.csv", records)


def write_native(output, metadata, cleaned, raw, references, names, regions, fs):
    condition = metadata["condition"]
    if condition not in ("rest", "lateral", "vertical", "blink"):
        raise ValueError("Unverified native condition cannot enter paper tables")
    metrics = native_condition_channels(cleaned, raw, references, condition)
    fields = ("rest_rmse", "eeg_eog_abs_r_before", "eeg_eog_abs_r_after")
    records = []
    for channel, name in enumerate(names):
        record = {**metadata, "channel": str(name), "channel_index": channel,
                  "region_code": int(regions[channel]), "protocol_id": PROTOCOL_ID,
                  "scoring_unit": "cached window (adapted)", "chance_level_claim": False}
        for key in fields:
            value = float(metrics[key][channel]) if key in metrics else np.nan
            record[key] = value
            record[key+"_status"] = ("not_applicable" if key not in metrics else
                                    "finite" if np.isfinite(value) else "undefined")
        records.append(record)
    append_table(output, "paper_native_channels.csv", records)
    if condition != "rest":
        return
    spectrum = resting_spectrum(cleaned, raw, fs, normalization_band=(1., 40.))
    directory = output/"paper_rest_spectra"
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/(metadata["example_id"]+"-"+metadata["method"]+".npz"),
                        channel_names=names, regions=regions, **spectrum)
    records = []
    for channel, name in enumerate(names):
        record = {**metadata, "channel": str(name), "channel_index": channel,
                  "region_code": int(regions[channel]), "protocol_id": PROTOCOL_ID}
        for key, value in spectrum.items():
            if "_relative_" in key:
                record[key] = float(value[channel])
        records.append(record)
    append_table(output, "paper_rest_bandpower_channels.csv", records)


def run_fixture(output):
    """Inspect an analytic oracle on Kaggle; these are not research results."""
    save_protocol(output)
    target = np.array([[-1., 1., -1., 1.], [-10., 10., -10., 10.]])
    metrics = paired_channels(target+1., target, target+2.)
    rows = [{"channel": channel, **{key: float(value[channel]) for key, value in metrics.items()}}
            for channel in range(2)]
    pd.DataFrame(rows).to_csv(output/"paper_metric_fixture.csv", index=False)
    atomic_json(output/"paper_metric_fixture_summary.json", {
        "passed": True, "scope": "analytical equation fixtures and full pytest suite; not scientific validation",
        "expected_channel_snr_db": [0., 20.], "expected_channel_mean_snr_db": 10.,
        "no_native_reconstruction_snr": True, "models_authorized": False,
    })
