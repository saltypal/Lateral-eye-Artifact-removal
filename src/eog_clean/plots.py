"""Figures generated from saved results, on Kaggle; no invented performance data."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch


def make_figures(output):
    output = Path(output)
    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .2})
    paths = []

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(figures/f"{name}.{extension}", bbox_inches="tight")
        plt.close(fig)
        paths.append(str(figures/f"{name}.png"))

    frame = pd.read_csv(output/"klados_per_record.csv")
    for index, (metric, label) in enumerate((("pooled_window_snr_db", "Output SNR (dB; pooled channels within window)"),
                                          ("rmse", "RMSE (source amplitude units)"),
                                          ("pearson_cc", "Ordinary Pearson CC")), 1):
        table = frame.groupby("method")[metric].agg(["mean", "std"])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(np.arange(len(table)), table["mean"], yerr=table["std"], capsize=3, color="#386cb0")
        ax.set_xticks(np.arange(len(table)), table.index, rotation=25, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"Klados nonregional comparison — {frame.record.nunique()} records\nBars: equal-record means; whiskers: record SD, not confidence intervals")
        if metric == "pearson_cc":
            ax.set_ylim(0, 1)
        save(fig, f"{index:02d}_klados_{metric}")
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, rows in frame.groupby("method"):
        ax.plot(rows.record, rows.pooled_window_snr_db, marker=".", label=method, alpha=.8)
    ax.set(xlabel="Klados exported record index (participant unknown)", ylabel="Record mean output SNR (dB)",
           title="Per-record variability; every eligible failure retained as pass-through")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "04_klados_record_snr")

    first = next((output/"klados").glob("record_*/predictions.npz"))
    with np.load(first, allow_pickle=False) as cache:
        start = min(int(name.split("_")[-1]) for name in cache.files if name.startswith("target_"))
        fig, ax = plt.subplots(figsize=(11, 4))
        for method, label in (("raw", "Contaminated input"), ("target", "Paired reference"),
                              ("direct_calibrated", "Calibrated regression"), ("ica_component_regression", "ICA regression")):
            ax.plot(np.arange(1024)/200, cache[f"{method}_{start}"][0], label=label, alpha=.8)
        ax.set(xlabel="Time within scoring window (s)", ylabel="Source amplitude units",
               title=f"Klados {first.parent.name}, row 0 — unknown electrode anatomy")
        ax.legend(ncol=2)
        save(fig, "05_klados_example")

    historical = json.loads((output/"historical_replay.json").read_text())
    hist = pd.read_csv(output/"historical_record_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(hist.record.astype(str), hist.snr_out_db, color="#8056a3")
    ax.axhline(historical["measured_record_macro_snr_db"], color="black", linestyle="--", label="Record mean")
    ax.set(xlabel="Reused historical Klados record index", ylabel="Output SNR (dB)",
           title="Historical VMD + EOG + learned correction replay\nDevelopment-exposed, all 19 unnamed rows; separate from the new benchmark")
    ax.legend()
    save(fig, "06_historical_replay")

    native = pd.read_csv(output/"osf_participant_macro.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, condition, metric in zip(axes, ["lateral", "blink"], ["absolute_heog_cc", "absolute_veog_cc"]):
        subset = native[native.condition == condition]
        table = subset.pivot(index="method", columns="region", values=metric)
        table.plot.bar(ax=ax, rot=35)
        ax.set(title=f"{condition}: {'HEOG' if condition == 'lateral' else 'VEOG'} association", ylabel="Mean absolute Pearson CC", ylim=(0, 1), xlabel="")
    fig.suptitle("Native OSF suppression proxies — equal participant means; lower alone does not prove preservation", y=1.03)
    save(fig, "07_osf_ocular_association")

    rest = pd.read_csv(output/"osf_rest_preservation.csv")
    record_rest = rest.groupby(["method", "record", "region"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric, label, threshold in zip(axes, ["relative_change", "alpha_error_db", "covariance_relative_error"],
                                          ["Rest relative RMS change", "Rest alpha error (dB)", "Rest relative covariance change"], [.01, .5, .02]):
        table = record_rest.pivot_table(index="method", columns="region", values=metric, aggfunc="max")
        table.plot.bar(ax=ax, rot=40)
        ax.axhline(threshold, color="black", linestyle="--", label="Engineering limit")
        ax.set(title=label+"\nMaximum eligible record mean", ylabel=label, xlabel="")
    save(fig, "08_osf_rest_preservation")

    diagnostics = pd.read_csv(output/"osf_mode_reference_evidence.csv")
    valid = diagnostics[diagnostics.valid]
    table = valid.assign(abs_heog=valid.heog_cc.abs(), abs_veog=valid.veog_cc.abs()).groupby(
        ["condition", "mode"])[["abs_heog", "abs_veog"]].mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(table.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks([0, 1], ["|mode–HEOG CC|", "|mode–VEOG CC|"])
    ax.set_yticks(range(len(table)), [f"{condition} — mode {mode}" for condition, mode in table.index])
    for row in range(len(table)):
        for column in range(2):
            ax.text(column, row, f"{table.iloc[row, column]:.3f}", ha="center", va="center", color="white" if table.iloc[row, column] < .6 else "black")
    fig.colorbar(image, ax=ax)
    ax.set_title("Every valid frontal VMD mode inspected against both EOG axes\nDisplay: mean absolute association; signed per-window values saved in CSV")
    save(fig, "09_vmd_mode_reference_evidence")

    example = next((output/"osf").glob("*/trial_*_frontal_modes.npz"))
    metadata = json.loads((example.parent/"source_and_split.json").read_text())
    frontal_index = next(i for i, group in enumerate(metadata["regions"]) if group == "frontal")
    trial_id = example.name.split("_")[1]
    with np.load(example, allow_pickle=False) as cache, np.load(example.parent/f"trial_{trial_id}_predictions.npz", allow_pickle=False) as predictions:
        fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
        for mode in range(3):
            axes[mode].plot(np.arange(1024)/200, cache["modes"][0, 0, mode])
            axes[mode].set_ylabel(f"Mode {mode+1}")
        for axis, label in zip([0, 1], ["HEOG", "VEOG"]):
            axes[3+axis].plot(np.arange(1024)/200, predictions["eyes"][axis, :1024])
            axes[3+axis].set_ylabel(label)
        axes[-1].set_xlabel("Time within trial (s)")
        axes[0].set_title(f"{example.parent.name}: {metadata['channel_names'][frontal_index]}, K=3, alpha=2000\nEach mode retains source amplitude; no claim that mode index alone identifies EOG")
        save(fig, "10_vmd_mode_vectors")

    rest_example = None
    correlations_frame = pd.read_csv(output/"osf_trial_channel_correlations.csv")
    rest_row = correlations_frame[correlations_frame.condition == "rest"].iloc[0]
    rest_example = output/"osf"/rest_row.record/f"trial_{rest_row.trial:03d}_predictions.npz"
    with np.load(rest_example, allow_pickle=False) as predictions:
        fig, ax = plt.subplots(figsize=(10, 5))
        for method in ("raw", "frontal_vmd_only", "direct_calibrated", "hybrid_vmd_ica_component_regression", "hybrid_vmd_mwf"):
            frequency, spectrum = welch(predictions[method], fs=200, nperseg=512, axis=-1)
            keep = (frequency >= .5) & (frequency <= 40)
            ax.semilogy(frequency[keep], spectrum.mean(axis=0)[keep], label=method)
        ax.set(xlabel="Frequency (Hz)", ylabel="PSD (source amplitude²/Hz)", title=f"Native resting EEG, {rest_row.record}, trial {rest_row.trial}\nOne common amplitude scale; whole-montage channel mean")
        ax.legend(fontsize=8)
        save(fig, "11_osf_rest_spectrum")

    return paths
