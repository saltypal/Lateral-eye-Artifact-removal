"""Saved Kaggle figures from measured neural selection/evaluation tables."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_snr_campaign(output, model_names):
    frame = pd.read_csv(output / "test_summary.csv")
    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    labels = frame.method.str.replace("_", " ")
    axis.barh(labels, frame.snr_db, color=["#247c96" if "safe" in name else "#8f969a" for name in frame.method])
    axis.axvline(15, color="#b84430", linestyle="--", label="15 dB target")
    axis.axvline(20, color="#b84430", linestyle=":", label="20 dB loss target")
    axis.set_xlabel("Mean record SNR (dB); 8 reused test records, 3 windows each")
    axis.set_title("Frozen validation-selected models: runtime EOG requirements differ")
    axis.legend(loc="lower right")
    for position, value in enumerate(frame.snr_db):
        axis.text(value + 0.1, position, f"{value:.2f}", va="center")
    fig.savefig(output / "snr_target_test_comparison.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 4), constrained_layout=True, squeeze=False)
    for axis, name in zip(axes[0], model_names):
        grid = pd.read_csv(output / name / "validation_grid.csv")
        for config, subset in grid[grid.strength == 1].groupby("config_id"):
            axis.plot(subset.epoch, subset.validation_snr_db, label=f"Config {config}")
        axis.axhline(15, color="#b84430", linestyle="--")
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("Training epoch")
        axis.set_ylabel("Validation mean record SNR (dB)")
        axis.legend(fontsize=8)
    fig.savefig(output / "snr_target_training_curves.png", dpi=170)
    plt.close(fig)
