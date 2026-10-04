"""Source-backed notebook figures, generated on Kaggle with the experiments."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .provenance import save_json

BLUE = "#2369A8"
ORANGE = "#C87B25"
GREY = "#6B7280"


def save(figure, output, name, sources):
    figure.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / f"{name}.svg", bbox_inches="tight")
    run_path = output / "run_config.json"
    phase = json.loads(run_path.read_text()).get("phase", "unknown") if run_path.exists() else "unknown"
    save_json(output / f"{name}.sources.json", {"sources": sources, "phase": phase, "full_study_inference": False})
    plt.close(figure)


def grid_figures(output):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    centers = json.loads((output / "vmd_centers.json").read_text())
    if centers:
        first = centers[0]
        selected = [row for row in centers if row["record"] == first["record"] and row["channel"] == first["channel"]]
        alphas = sorted(set(row["alpha"] for row in selected))
        figure, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
        maximum = max(max(row["centers_hz"]) for row in selected)
        for axis, alpha in zip(axes.flat, alphas):
            for row in selected:
                if row["alpha"] != alpha:
                    continue
                frequency = np.asarray(row["centers_hz"])
                axis.scatter(np.full(len(frequency), row["K"]), frequency, s=35, color=BLUE, alpha=0.8)
            axis.axhline(40, color=GREY, linestyle="--", linewidth=1)
            axis.set(title=f"Bandwidth penalty α = {alpha}", xlabel="Number of modes K", ylabel="Optimizer center (Hz)",
                     xticks=list(range(3, 11)), ylim=(0, max(45, maximum + 2)))
            axis.grid(axis="y", alpha=0.18)
        for axis in list(axes.flat)[len(alphas):]:
            axis.set_visible(False)
        figure.suptitle(f"VMD center-frequency sweep · development record {first['record']}, channel {first['channel']}")
        figure.text(0.02, 0.01, "One fixed 1,024-sample segment at 200 Hz. Dots are separate modes; mode identities are not matched across K. Dashed line: 40-Hz analysis cutoff.", fontsize=9)
        figure.tight_layout(rect=(0, 0.05, 1, 0.95))
        save(figure, output, "vmd_center_frequency_sweep", ["vmd_centers.json", "grid_coverage.json"])
    summary = pd.read_csv(output / "vmd_grid_summary.csv")
    figure, axis = plt.subplots(figsize=(8, 6))
    for feasible, marker, color, label in [(False, "x", ORANGE, "Outside preservation guardrails"),
                                          (True, "o", BLUE, "Within preservation guardrails")]:
        rows = summary[summary.preservation_guard_pass == feasible]
        axis.scatter(rows.clean_relative_change * 100, rows.rmse_improvement_fraction * 100,
                     marker=marker, color=color, s=35, alpha=0.7, label=label)
    axis.axhline(0, color=GREY, linewidth=1)
    maximum_change = float((summary.clean_relative_change * 100).max())
    if maximum_change >= 16:
        axis.axvline(20, color=GREY, linestyle="--", linewidth=1)
    else:
        axis.set_xlim(left=0, right=max(1, maximum_change * 1.15))
    axis.set(xlabel="Modification to clean EEG (% relative norm)", ylabel="RMSE improvement over raw EEG (%)",
             title="VMD correction versus clean-signal modification")
    axis.legend(loc="best", fontsize=9)
    axis.grid(alpha=0.15)
    figure.text(0.02, 0.01, "Development subset only. Guardrails: modification ≤20%, clean alpha/beta errors ≤1 dB. This is not a noninferiority test.", fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    save(figure, output, "vmd_preservation_tradeoff", ["vmd_grid_summary.csv", "grid_coverage.json"])


def student_figures(output):
    """Fixed held-out examples and measured cap timing; no fitted trend claims."""
    arrays = np.load(output / "student_test_predictions.npz", allow_pickle=False)
    time = np.arange(arrays["raw"].shape[-1]) / 200
    count = len(arrays["record_ids"])
    figure, axes = plt.subplots(count, 1, figsize=(11, 3 * count), squeeze=False, sharex=True)
    for index, record in enumerate(arrays["record_ids"]):
        axis = axes[index, 0]
        axis.plot(time, arrays["raw"][index, 0], color=GREY, linewidth=1, alpha=0.7, label="Contaminated")
        axis.plot(time, arrays["target"][index, 0], color=BLUE, linewidth=1.3, label="Paired clean target")
        axis.plot(time, arrays["predictions"][index, 0], color=ORANGE, linewidth=1, label="Student")
        axis.set(title=f"Held-out record {record}, anonymous channel 0", ylabel="Source amplitude units")
        axis.grid(alpha=0.15)
        axis.legend(loc="upper right", ncol=3, fontsize=9)
    axes[-1, 0].set_xlabel("Time within scored window (seconds)")
    figure.suptitle("EEG-only model · fixed held-out channel examples")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save(figure, output, "student_heldout_waveforms", ["student_test_predictions.npz", "student_metrics.csv"])
    timing = pd.read_csv(output / "student_latency_scaling.csv")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(timing.channels, timing.median_ms, marker="o", color=BLUE, label="Measured median")
    axis.plot(timing.channels, timing.p95_ms, marker="s", color=ORANGE, label="Measured p95")
    axis.set(xscale="log", xlabel="Valid EEG channels", ylabel="Offline forward time (ms)",
             title=f"Cap-size timing · {timing.device.iloc[0]}, batch 1, 1,024 samples",
             xticks=timing.channels, xticklabels=[str(value) for value in timing.channels])
    axis.legend()
    axis.grid(alpha=0.15)
    figure.text(0.02, 0.01, "Synthetic tensors; 3 warmups and 10 timed forwards per size. Excludes acquisition/preprocessing. No accuracy or live deadline claim.", fontsize=8)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    save(figure, output, "student_channel_scaling", ["student_latency_scaling.csv", "training_summary.json"])


def neural_search_figures(output):
    grid = pd.read_csv(output / "neural_grid.csv")
    selected = json.loads((output / "selected_neural.json").read_text())
    figure, axis = plt.subplots(figsize=(8, 6))
    rows = grid[grid.clean_relative_change_worst_record <= 0.05]
    colors = axis.scatter(rows.clean_relative_change_worst_record * 100, rows.rmse_improvement_fraction * 100,
                 c=rows.identity_weight, cmap="viridis", s=12, alpha=0.4)
    figure.colorbar(colors, ax=axis, label="Clean identity loss weight")
    axis.axvline(1, color=GREY, linestyle="--", label="1% development preservation bound")
    axis.axhline(10, color=GREY, linestyle=":", label="10% development error reduction target")
    axis.scatter(selected["clean_relative_change_worst_record"] * 100, selected["rmse_improvement_fraction"] * 100,
                 color=ORANGE, marker="*", s=180, label="Validation selection")
    axis.set(xlabel="Worst record mean clean modification (%)", ylabel="Validation RMSE reduction (%)",
             title="Neural validation search · clean preservation versus correction")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.15)
    figure.text(0.02, 0.01, "8 validation records, 3 windows each. Epochs and strengths are development candidates. Only points within 5% modification shown; full grid saved.", fontsize=8)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    save(figure, output, "neural_validation_tradeoff", ["neural_grid.csv", "selected_neural.json"])
    records = pd.read_csv(output / "student_record_metrics.csv")
    comparison = records.pivot(index="record", columns="method", values="rmse")
    figure, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(comparison))
    axis.bar(positions - 0.18, comparison.raw, width=0.36, color=GREY, label="Contaminated")
    axis.bar(positions + 0.18, comparison.eeg_only_student, width=0.36, color=BLUE, label="EEG-only student")
    axis.set(xticks=positions, xticklabels=[str(value) + ("*" if value in [5, 14] else "") for value in comparison.index],
             xlabel="Held-out record (* previously inspected smoke record)", ylabel="Record mean RMSE (source units)",
             title="Frozen selected model · all eight held-out records, three windows each")
    axis.legend()
    axis.grid(axis="y", alpha=0.15)
    figure.tight_layout()
    save(figure, output, "neural_record_comparison", ["student_record_metrics.csv", "search_protocol.json"])


def example_modes(raw, clean, modes, residual, centers, output):
    time = np.arange(raw.size) / 200
    figure, axes = plt.subplots(len(modes) + 2, 1, figsize=(11, 1.5 * (len(modes) + 2)), sharex=True)
    axes[0].plot(time, raw, color=ORANGE, linewidth=1, label="Contaminated EEG")
    axes[0].plot(time, clean, color=BLUE, linewidth=1, label="Paired clean target")
    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    for index, (axis, vector, center) in enumerate(zip(axes[1:-1], modes, centers)):
        axis.plot(time, vector, color=BLUE, linewidth=1)
        axis.set_ylabel(f"Mode {index + 1}\n{center:.2f} Hz")
    axes[-1].plot(time, residual, color=GREY, linewidth=1)
    axes[-1].set(ylabel="Residual", xlabel="Time (seconds)")
    figure.suptitle("Actual VMD mode vectors · same sample alignment, independent vertical scales")
    figure.text(0.02, 0.01, "Original Klados amplitude units remain unverified. Mode order follows optimizer centers; modes are not fixed physiological bins.", fontsize=9)
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(figure, output, "vmd_mode_vectors", ["vmd_example.npz", "vmd_example_metadata.json"])
