"""
infer_epoch.py — Visualise model predictions vs MATLAB labels on a single epoch
================================================================================
Replicates the EEGLAB-style stacked IC plot from NewMethodTry.ipynb for one
chosen epoch, then overlays two label strips:

  ┌──────────────────────────────────────────────────────────────┐
  │  Model predictions  (colour per window, from CNN-LSTM)       │
  │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
  │  MATLAB labels  (per-sample colours from event latencies)    │
  │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
  │  [  IC1 waveform                                            ]│
  │  [  IC2 waveform                                            ]│
  │  ...                                                         │
  └──────────────────────────────────────────────────────────────┘

Usage (run from src/ directory):
    python infer_epoch.py                 # default: epoch 4 (0-indexed: 3)
    python infer_epoch.py --epoch 7       # any 1-indexed epoch number
    python infer_epoch.py --epoch 4 --n_ics 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")          # popup window; falls back to Agg if no display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── local imports ──────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent           # src/
sys.path.insert(0, str(_SRC / "preprocessing"))
sys.path.insert(0, str(_SRC / "model"))

from preprocess import (                          # noqa: E402
    load_eeg, fir_filter,
    load_ica_and_compute_activations,
    _build_sample_labels,
    ICANormalizer,
    CLASS_NAMES, CLASSES, N_CLASSES,
)
from cnn_lstm import LateralEyeCNNLSTM            # noqa: E402


# ─────────────────────────────────────────────────────────────
# Colour palette  (matches NewMethodTry.ipynb exactly)
# ─────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "eye-l":    "#DD0000",
    "eye-r":    "#CC00CC",
    "eye-u":    "#00AA00",
    "eye-d":    "#0000DD",
    "blink":    "#996600",
    "fixation": "#888888",
}
# Bar height is in "data units" (fraction of total y-axis)
BAR_HEIGHT_FRAC = 0.045  # each label strip is ~4.5% of the plot height


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise model predictions vs MATLAB labels on one EEG epoch"
    )
    p.add_argument("--epoch",    type=int, default=4,
                   help="1-indexed epoch to visualise (default: 4)")
    p.add_argument("--n_ics",    type=int, default=6,
                   help="number of ICs to show in the stacked plot (default: 6)")
    p.add_argument("--window",   type=float, default=1.0,
                   help="window length in seconds (must match trained model, default 1.0)")
    p.add_argument("--stride",   type=float, default=0.25,
                   help="window stride in seconds (default 0.25)")
    p.add_argument("--ckpt",     type=str,
                   default=str(_SRC / "checkpoints" / "best_model.pt"),
                   help="path to checkpoint file")
    p.add_argument("--out",      type=str,
                   default=str(_SRC / "plots" / "epoch_comparison.png"),
                   help="output image path")
    p.add_argument("--spacing",  type=int, default=50,
                   help="vertical spacing between IC traces (eegplot 'spacing')")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    epoch_1idx = args.epoch          # 1-indexed (same as EEGLAB/notebook)
    epoch_0idx = epoch_1idx - 1      # 0-indexed numpy

    # ── 1. Load & preprocess ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"INFERENCE ON EPOCH {epoch_1idx}")
    print(f"{'='*60}")

    meta, epochs_raw = load_eeg()
    epochs_filt      = fir_filter(epochs_raw)
    ica              = load_ica_and_compute_activations(epochs_filt, meta)

    ic_all       = ica["ic_all"]        # (trials, n_ics, pnts)
    trials       = meta["trials"]
    srate        = meta["srate"]
    pnts         = meta["pnts"]
    sample_labels = _build_sample_labels(meta)  # (trials, pnts) int64

    if epoch_0idx < 0 or epoch_0idx >= trials:
        raise ValueError(
            f"Epoch {epoch_1idx} is out of range (dataset has {trials} epochs)."
        )

    # ── 2. Load checkpoint ────────────────────────────────────
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run  python train.py  first to generate a checkpoint."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"\n[checkpoint] loaded {ckpt_path}")
    print(f"  trained for epoch {ckpt.get('epoch', '?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}  "
          f"val_acc={ckpt.get('val_acc', float('nan')):.3f}")

    saved_args  = ckpt.get("args", {})
    n_ics_model = saved_args.get("n_ics", 58)
    n_classes   = ckpt.get("n_classes", N_CLASSES)
    class_names = ckpt.get("class_names", CLASS_NAMES)

    model = LateralEyeCNNLSTM(
        n_ics=n_ics_model,
        n_classes=n_classes,
        window_len=int(args.window * srate),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Restore normalizer
    norm = ICANormalizer()
    norm.mean_ = ckpt["norm_mean"]
    norm.std_  = ckpt["norm_std"]

    print(f"  model: {n_ics_model} ICs → {n_classes} classes  "
          f"({sum(p.numel() for p in model.parameters() if p.requires_grad):,} params)")

    # ── 3. Extract epoch and build windows ────────────────────
    win    = int(args.window * srate)    # 200 samples
    stride = int(args.stride * srate)   # 50 samples

    ic_epoch = ic_all[epoch_0idx, :n_ics_model, :]   # (n_ics_model, pnts)

    windows  = []
    win_times = []       # centre time of each window (seconds)
    start = 0
    while start + win <= pnts:
        end = start + win
        x   = ic_epoch[:, start:end].T.astype(np.float32)  # (win, n_ics)
        windows.append(x)
        win_times.append((start + end) / 2 / srate)
        start += stride

    X_epoch = np.stack(windows)  # (n_windows, win, n_ics)
    X_norm  = norm.transform(X_epoch)

    print(f"\n[inference] epoch {epoch_1idx}  "
          f"pnts={pnts}  win={win}  stride={stride}  "
          f"→ {len(windows)} windows")

    # ── 4. Run inference ──────────────────────────────────────
    with torch.no_grad():
        X_t    = torch.from_numpy(X_norm).float().to(device)  # (N, win, n_ics)
        logits = model(X_t)                                     # (N, n_classes)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()     # (N, n_classes)
        preds  = logits.argmax(dim=1).cpu().numpy()             # (N,)

    pred_names = [class_names[p] for p in preds]

    # ── Per-window ground truth (majority vote, same as training) ──
    gt_per_sample = sample_labels[epoch_0idx]        # (pnts,) int64
    gt_per_window = []
    w_start = 0
    while w_start + win <= pnts:
        w_end     = w_start + win
        win_lbls  = gt_per_sample[w_start:w_end]
        gt_cls    = int(np.bincount(win_lbls, minlength=N_CLASSES).argmax())
        gt_per_window.append(gt_cls)
        w_start  += stride
    gt_per_window = np.array(gt_per_window, dtype=np.int64)

    accuracy = np.mean(preds == gt_per_window)

    print(f"\n  Model predictions vs MATLAB ground truth per window:")
    print(f"  {'Win':>4s}  {'t_centre':>8s}  {'GT':>10s}  {'pred':>10s}  {'conf':>8s}")
    for i, (t_c, pred_n, prob_row) in enumerate(zip(win_times, pred_names, probs)):
        gt_n  = class_names[gt_per_window[i]]
        conf  = prob_row[preds[i]]
        match = "✓" if preds[i] == gt_per_window[i] else "✗"
        print(f"  {i+1:>4d}  {t_c:>7.3f}s  {gt_n:>10s}  {pred_n:>10s}  {conf:>7.1%}  {match}")

    print(f"\n  Window-level accuracy: {accuracy:.1%}")

    # ── 5. Build stacked IC plot ───────────────────────────────
    N_ICS   = min(args.n_ics, n_ics_model)
    SPACING = args.spacing
    t_axis  = np.arange(pnts) / srate

    # EEGLAB Norm: divide each IC by its std (no mean subtraction)
    ic_norm = np.zeros((N_ICS, pnts), dtype=np.float32)
    for i in range(N_ICS):
        s = np.std(ic_epoch[i])
        ic_norm[i] = ic_epoch[i] / s if s > 0 else ic_epoch[i]

    # y-axis range
    y_top     = (N_ICS + 1) * SPACING
    bar_h     = y_top * BAR_HEIGHT_FRAC

    # gt strip sits above IC traces
    gt_y_bot  = y_top
    gt_y_top  = y_top + bar_h

    # pred strip sits above gt strip
    pred_y_bot = gt_y_top + bar_h * 0.3
    pred_y_top = pred_y_bot + bar_h

    # extend y limit
    y_max = pred_y_top + bar_h * 0.5

    # ── Helper: draw a coloured strip from a per-sample label array ──
    def _draw_strip(ax, per_sample, y_mid, strip_h, label_text):
        """Draw coloured segments for a per-sample label array."""
        prev_p    = None
        seg_start = 0.0
        for si in range(pnts + 1):
            p = int(per_sample[si]) if si < pnts else -1
            if p != prev_p:
                if prev_p is not None and prev_p >= 0:
                    seg_name = class_names[prev_p]
                    seg_col  = CLASS_COLORS.get(seg_name, "#888888")
                    seg_end  = si / srate
                    seg_w    = seg_end - seg_start
                    mid_t    = seg_start + seg_w / 2
                    ax.barh(y=y_mid, width=seg_w, left=seg_start,
                            height=strip_h, color=seg_col, alpha=0.85, zorder=3)
                    if seg_w > 0.3:
                        ax.text(mid_t, y_mid, seg_name,
                                fontsize=7, ha="center", va="center",
                                color="white", fontweight="bold", zorder=4,
                                clip_on=True)
                seg_start = si / srate
                prev_p    = p
        ax.text(-0.15, y_mid, label_text,
                fontsize=9, ha="right", va="center",
                color="black", fontweight="bold")

    # ── Build per-sample prediction array ─────────────────────
    pred_per_sample = np.full(pnts, CLASSES["fixation"], dtype=int)
    s = 0
    for i, p in enumerate(preds):
        pred_per_sample[s:s + win] = p
        s += stride
    if len(preds) > 0:
        pred_per_sample[s:] = preds[-1]

    # ── Figure ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 11))
    fig.patch.set_facecolor("#E0E0EC")
    ax.set_facecolor("white")

    # Ground-truth strip (per-sample colours from event latencies)
    _draw_strip(ax, gt_per_sample,
                (gt_y_bot + gt_y_top) / 2, bar_h, "MATLAB\nlabels")

    # Model-prediction strip (per-sample colours from windows)
    _draw_strip(ax, pred_per_sample,
                (pred_y_bot + pred_y_top) / 2, bar_h, "Model\npreds")

    # Separator line
    ax.axhline(y_top, color="black", lw=0.8, ls="--", alpha=0.4, zorder=5)

    # IC waveforms (EEGLAB eegplot style)
    for i in range(N_ICS):
        centre = (N_ICS - i) * SPACING
        ax.plot(t_axis, ic_norm[i] + centre, lw=0.6, color="black", zorder=2)
        ax.text(-0.15, centre, f"IC {i+1}",
                fontsize=10, ha="right", va="center", fontweight="bold")

    # Window boundary markers
    ws = 0
    for _ in preds:
        ax.axvline(ws / srate, color="gray", lw=0.4, ls=":", alpha=0.5, zorder=1)
        ws += stride

    # Axes cosmetics
    ax.set_xlim(0, pnts / srate)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_yticks([])
    ax.set_title(
        f"Epoch {epoch_1idx}  —  MATLAB event labels  vs  CNN-LSTM predictions\n"
        f"ICs shown: {N_ICS}  |  EEGLAB Norm (data/std)  |  "
        f"window={args.window}s  stride={args.stride}s  |  "
        f"window accuracy: {accuracy:.1%}",
        fontsize=12, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=CLASS_COLORS[cn], alpha=0.85, label=cn)
        for cn in class_names if cn in CLASS_COLORS
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              framealpha=0.9, ncol=3, title="Class colours", title_fontsize=8)

    # Save + show
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")
    plt.show()

    # ── 6. Print summary ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Epoch          : {epoch_1idx}")
    print(f"  Total windows  : {len(preds)}")
    correct = int(np.sum(preds == gt_per_window))
    print(f"  Correct windows: {correct}")
    print(f"  Window accuracy: {accuracy:.1%}")
    print(f"\n  Ground-truth distribution:")
    unique_gt, counts_gt = np.unique(gt_per_window, return_counts=True)
    for cls, cnt in zip(unique_gt, counts_gt):
        print(f"    {class_names[cls]:>10s}  {cnt:3d}/{len(gt_per_window)}")
    print(f"\n  Prediction distribution:")
    unique, counts = np.unique(preds, return_counts=True)
    for cls, cnt in zip(unique, counts):
        bar = "█" * int(cnt / len(preds) * 30)
        print(f"    {class_names[cls]:>10s}  {cnt:3d}/{len(preds)}  {cnt/len(preds):5.1%}  {bar}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
