"""Nine native diagnostic figures and a complete cached mode/reference audit.

Numerical execution is Kaggle-only through runner.py. No predictions change.
"""
import json

import numpy as np
import pandas as pd

from .broader_coupled_corpus import declared_parent
from .io import atomic_json, read_jsonl
from .reference import correlations, lag_matrix


METHODS = ("identity", "raw_shared", "vmd_shared", "vmd_anchored", "vmd_anchored_raw_fallback")
LABELS = ("Original", "Raw ridge", "VMD shared", "VMD anchored", "VMD + fallback")
COLORS = ("#555555", "#0072B2", "#009E73", "#D55E00", "#CC79A7")


def readiness(input_root, config):
    if (config.get("scope") != "native_cached_visual_audit" or config.get("autovmd_enabled") is not False
            or config.get("reserved_access") != "closed until verified freeze"):
        raise ValueError("Visualization-only policy differs")
    parent = declared_parent(input_root, "native_cached_frontal_summary.json", config["visual_parent"])
    summary = json.loads((parent / "native_cached_frontal_summary.json").read_text())
    if (summary["trials"] != 51 or summary["decompositions"] != 4284
            or summary["reserved_confirmation_opened"] is not False or summary["posterior_changed"] is not False):
        raise ValueError("Native visual parent scope differs")
    return parent


def associations(modes, refs):
    """Match the actual signed peak selector while also preserving peak lags."""
    zero = correlations(modes, refs, 0)
    best, peak_lag = np.zeros_like(zero), np.zeros_like(zero, dtype=int)
    for lag in range(-20, 21):
        design, valid = lag_matrix(refs, (lag,))
        x = modes[:, valid].astype(float)
        y = design[:, valid].astype(float)
        x -= x.mean(-1, keepdims=True)
        y -= y.mean(-1, keepdims=True)
        nx, ny = np.linalg.norm(x, axis=1, keepdims=True), np.linalg.norm(y, axis=1, keepdims=True)
        x = np.divide(x, nx, out=np.zeros_like(x), where=nx > 0)
        y = np.divide(y, ny, out=np.zeros_like(y), where=ny > 0)
        current = np.clip(x @ y.T, -1., 1.)
        improved = np.abs(current) > np.abs(best)
        best, peak_lag = np.where(improved, current, best), np.where(improved, lag, peak_lag)
    np.testing.assert_allclose(best, correlations(modes, refs, 20), rtol=1e-12, atol=1e-12)
    return zero, best, peak_lag


def audit_modes(parent, output):
    rows, loaded_trial, archive = [], None, None
    for path in sorted((parent / "modes").glob("*.json")):
        trial_key, window_text = path.stem.rsplit("-window-", 1)
        if trial_key != loaded_trial:
            with np.load(parent / "trial_predictions" / (trial_key + ".npz"), allow_pickle=False) as stored:
                archive = dict(stored)
            loaded_trial = trial_key
        info = json.loads(path.read_text())
        decisions = read_jsonl(parent / "decisions" / (trial_key + ".jsonl"))[int(window_text)]
        selected = {row["channel"]: row["selected_modes"] for row in decisions["details"]["vmd_shared"]}
        # This is the exact reflected window consumed by the inference selector.
        ref = np.pad(archive["inference_references"], ((0, 0), (512, 1536)), mode="reflect")
        start = int(window_text) * 512
        ref = ref[:, start:start + 1024]
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
            for channel, detail in info["channels"].items():
                name = str(archive["channel_names"][int(channel)])
                modes = saved[f"channel_{channel}_modes"]
                zero, peak, lags = associations(modes, ref)
                failed = bool(detail.get("failure") or detail.get("hit_iteration_limit"))
                expected = [] if failed else np.flatnonzero(np.max(np.abs(peak), axis=1) >= .2).tolist()
                if expected != selected[name]:
                    raise ValueError("Cached mode-selection replay differs")
                for mode in range(len(modes)):
                    rows.append({"trial_key": trial_key, "window": int(window_text), "channel": name,
                        "mode": mode + 1, "center_hz": detail.get("centers_hz", [np.nan] * 3)[mode],
                        "zero_heog": zero[mode, 0], "zero_veog": zero[mode, 1],
                        "peak_heog": peak[mode, 0], "peak_veog": peak[mode, 1],
                        "heog_peak_lag_samples": int(lags[mode, 0]), "veog_peak_lag_samples": int(lags[mode, 1]),
                        "selected": mode in expected, "solver_flagged": failed})
        if len(rows) % 2100 == 0:
            print("MODE_REFERENCE_AUDIT_ROWS", len(rows), flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 12852:
        raise ValueError("Expected all 4284 channel decompositions times three modes")
    frame.to_csv(output / "all_mode_reference_correlations.csv", index=False)
    return frame


def run(input_root, output, config, profile):
    if profile != "full":
        raise ValueError("Visual audit requires the complete native pilot")
    parent = readiness(input_root, config)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    figures = output / "figures"
    figures.mkdir()
    inventory = []

    def finish(fig, number, name, caption):
        stem = f"{number:02d}_{name}"
        fig.text(.5, .01, "Two development OSF participants; unqualified frontal methods; no native reconstruction SNR",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .95))
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(figures / (stem + "." + suffix), dpi=180, bbox_inches="tight")
        plt.close(fig)
        inventory.append({"figure": stem, "caption": caption, "source_parent": config["visual_parent"]})
        print("NATIVE_FIGURE_SAVED", stem, flush=True)

    table = pd.read_csv(parent / "native_condition_participants.csv")
    for number, condition in enumerate(("lateral", "blink", "vertical"), 1):
        fig, ax = plt.subplots(figsize=(10, 5))
        for method, label, color in zip(METHODS, LABELS, COLORS):
            part = table[(table.condition == condition) & (table.method == method)].sort_values("source")
            ax.plot(part.source, part.eeg_eog_abs_r_after, marker="o", label=label, color=color)
        ax.set_ylim(0, 1)
        axis = "HEOG" if condition == "lateral" else "VEOG"
        ax.set(title=f"{condition.title()}: unchanged-reference ocular association", xlabel="Participant",
               ylabel=f"Mean absolute EEG–{axis} Pearson correlation")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        finish(fig, number, condition, "Complete-condition participant means; lower association alone does not establish preservation.")

    rest = pd.read_csv(parent / "native_frontal_preservation_records.csv")
    for number, field, label, limit in ((4, "relative_error", "Relative RMS modification", .01),
                                       (5, "alpha_db", "Absolute alpha-band error (dB)", .5),
                                       (6, "beta_db", "Absolute beta-band error (dB)", .5),
                                       (7, "covariance", "Relative covariance change", .02)):
        fig, ax = plt.subplots(figsize=(10, 5))
        for method, name, color in zip(METHODS, LABELS, COLORS):
            part = rest[rest.method == method].sort_values("source")
            ax.plot(part.source, part[field], marker="o", color=color, label=name)
        ax.axhline(limit, color="black", linestyle="--", label=f"Required limit {limit:g}")
        ax.set_ylim(bottom=0)
        ax.set(title="Resting preservation: " + label, xlabel="Participant", ylabel=label)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        finish(fig, number, field, "Means over resting trials per recording. Rest is not proven artifact-free; unchanged project limit shown.")

    record = sorted(rest.record_id.unique())[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, label, color in zip(METHODS, LABELS, COLORS):
        with np.load(parent / "native_rest_spectra" / f"{record}-{method}.npz", allow_pickle=False) as spectrum:
            ax.semilogy(spectrum["frequencies_hz"], spectrum["psd_after"].mean(axis=0), label=label, color=color)
    ax.set(xlim=(.5, 40), xlabel="Frequency (Hz)", ylabel="PSD (original array units²/Hz)",
           title=f"Resting PSD, frontal-channel mean: {record}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    finish(fig, 8, "rest_psd", "First record by ID, all resting trials, archived segment-weighted PSD; physical amplitude units unverified.")

    audit = audit_modes(parent, output)
    example = audit.sort_values(["trial_key", "window", "channel", "mode"]).iloc[0]
    chosen = audit[(audit.trial_key == example.trial_key) & (audit.window == example.window)
                   & (audit.channel == example.channel)].sort_values("mode")
    fig, ax = plt.subplots(figsize=(9, 5))
    values = chosen[["zero_heog", "zero_veog", "peak_heog", "peak_veog"]].to_numpy()
    heat = ax.imshow(values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(4), ["HEOG, zero lag", "VEOG, zero lag", "HEOG, peak lag", "VEOG, peak lag"])
    ax.set_yticks(range(3), [f"Mode {row.mode}, {row.center_hz:.2f} Hz" for row in chosen.itertuples()])
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center",
                    color="white" if abs(values[i,j]) > .65 else "black")
    ax.set_title(f"Mode–EOG evidence: {example.trial_key}, window {example.window}, {example.channel}")
    fig.colorbar(heat, ax=ax, label="Signed Pearson association")
    finish(fig, 9, "mode_eog_correlations", "First trial/window/channel by ID, not chosen for favorable output. Peak searches ±20 samples; reflected inference window. Full CSV includes every mode and selection replay.")
    atomic_json(output / "figure_inventory.json", inventory)
    atomic_json(output / "native_visual_summary.json", {"figures": len(inventory), "formats": ["png", "pdf", "svg"],
        "mode_rows": len(audit), "all_cached_selections_reproduced": True,
        "predictions_changed": False, "new_vmd_decompositions": 0, "teacher_qualified": False,
        "reserved_confirmation_opened": False})
