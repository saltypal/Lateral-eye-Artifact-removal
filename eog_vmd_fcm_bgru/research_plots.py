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
    save_json(output / f"{name}.sources.json", {"sources": sources, "scope": "kaggle_smoke; no final-study inference"})
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
    axis.axvline(20, color=GREY, linestyle="--", linewidth=1)
    axis.set(xlabel="Modification to clean EEG (% relative norm)", ylabel="RMSE improvement over raw EEG (%)",
             title="VMD correction versus clean-signal modification")
    axis.legend(loc="best", fontsize=9)
    axis.grid(alpha=0.15)
    figure.text(0.02, 0.01, "Development subset only. Guardrails also require clean alpha and beta band errors ≤1 dB; this is not a noninferiority test.", fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    save(figure, output, "vmd_preservation_tradeoff", ["vmd_grid_summary.csv", "grid_coverage.json"])


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
