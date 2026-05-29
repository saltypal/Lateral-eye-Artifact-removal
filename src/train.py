"""
train.py — Train CNN-LSTM for 6-class Eye Movement Detection
============================================================
Run from  src/  directory:

    # All study01 participants (default)
    python train.py

    # Specific participant only
    python train.py --participant p01

    # Different study
    python train.py --study study02

    # Full overrides
    python train.py --study study01 --epochs 80 --batch 64 --lr 5e-4

Outputs (per study)
-------------------
  checkpoints/{study}_best_model.pt
  checkpoints/{study}_final_model.pt
  plots/{study}_training_curves.png
  plots/{study}_confusion_matrix.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Local imports ──────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC / "preprocessing"))
sys.path.insert(0, str(_SRC / "model"))

from preprocess import run_preprocessing, EEG_DIR, ICA_DIR   # noqa: E402
from cnn_lstm   import LateralEyeCNNLSTM, build_loss          # noqa: E402


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN-LSTM eye movement classifier")
    p.add_argument("--study",       type=str,   default="study01",
                   help="study prefix to train on (e.g. study01, study02)")
    p.add_argument("--participant", type=str,   default="all",
                   help="participant id (e.g. p01) or 'all' for every participant in study")
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--batch",       type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--window",      type=float, default=1.0)
    p.add_argument("--stride",      type=float, default=0.25)
    p.add_argument("--n_ics",       type=int,   default=58)
    p.add_argument("--patience",    type=int,   default=15)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    """X: float32 (N,T,C)  |  y: int64 (N,) class indices for CrossEntropyLoss."""
    return TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y.astype(np.int64)),   # (N,)  — no unsqueeze
    )


def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb  = xb.to(device), yb.to(device)   # yb: (B,) int64
            logits   = model(xb)                       # (B, n_classes)
            loss     = criterion(logits, yb)
            total_loss += loss.item() * len(xb)
            preds    = logits.argmax(dim=1)            # (B,)
            correct += (preds == yb).sum().item()
            total   += len(xb)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────
# Train / val loop
# ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] device={device}  seed={args.seed}")

    # ── 1. Preprocess ──────────────────────────────────────────
    data = run_preprocessing(window_sec=args.window,
                             stride_sec=args.stride,
                             n_ics=args.n_ics)

    X_tr, y_tr = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_te,  y_te  = data["X_test"],  data["y_test"]

    train_loader = DataLoader(to_tensor_dataset(X_tr, y_tr),
                              batch_size=args.batch, shuffle=True,
                              drop_last=True)
    val_loader   = DataLoader(to_tensor_dataset(X_val, y_val),
                              batch_size=args.batch, shuffle=False)
    test_loader  = DataLoader(to_tensor_dataset(X_te,  y_te),
                              batch_size=args.batch, shuffle=False)

    print(f"\n  train batches : {len(train_loader)}"
          f"  val batches : {len(val_loader)}"
          f"  test batches : {len(test_loader)}")

    # ── 2. Model ───────────────────────────────────────────────
    _, window_len, n_ics = X_tr.shape
    n_classes = data["n_classes"]
    class_names = data["class_names"]
    model = LateralEyeCNNLSTM(n_ics=n_ics, n_classes=n_classes,
                               window_len=window_len).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  n_classes     : {n_classes}  {class_names}")
    print(f"  model params  : {n_params:,}")

    criterion = build_loss(y_tr, n_classes=n_classes).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # ── 3. Output dirs ─────────────────────────────────────────
    ckpt_dir  = _SRC / "checkpoints"
    plot_dir  = _SRC / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss  = float("inf")
    patience_count = 0
    history: dict = {"train_loss": [], "val_loss": [],
                     "train_acc":  [], "val_acc":  []}

    # ── 4. Training loop ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TRAINING  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            ep_loss    += loss.item() * len(xb)
            preds       = logits.argmax(dim=1)         # (B,)
            ep_correct += (preds == yb).sum().item()
            ep_total   += len(xb)

        tr_loss = ep_loss / ep_total
        tr_acc  = ep_correct / ep_total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:3d}/{args.epochs}"
              f"  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}"
              f"  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        # ── checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "n_classes":   n_classes,
                "class_names": class_names,
                "norm_mean":   data["normalizer"].mean_,
                "norm_std":    data["normalizer"].std_,
                "args":        vars(args),
            }, ckpt_dir / "best_model.pt")
            print(f"  ✓ saved best checkpoint  (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n  Early stopping triggered at epoch {epoch}")
                break

    # ── 5. Save final model ────────────────────────────────────
    torch.save({
        "model_state": model.state_dict(),
        "n_classes":   n_classes,
        "class_names": class_names,
        "norm_mean":   data["normalizer"].mean_,
        "norm_std":    data["normalizer"].std_,
        "args":        vars(args),
    }, ckpt_dir / "final_model.pt")

    # ── 6. Test evaluation ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("TEST EVALUATION  (loading best checkpoint)")
    print(f"{'='*60}")

    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"  Test loss : {test_loss:.4f}")
    print(f"  Test acc  : {test_acc:.3f}  ({test_acc*100:.1f}%)")

    # Detailed metrics
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            p  = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.extend(p.tolist())
            all_labels.extend(yb.numpy().tolist())

    _print_classification_report(all_labels, all_preds, class_names)

    # ── 7. Plots ───────────────────────────────────────────────
    _plot_training_curves(history, plot_dir)
    _plot_confusion_matrix(all_labels, all_preds, class_names, plot_dir)

    print(f"\nDone.  Checkpoints → {ckpt_dir}\n       Plots       → {plot_dir}")


# ─────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────

def _print_classification_report(labels: list, preds: list,
                                   class_names: list):
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    all_labels_idx = list(range(len(class_names)))
    print("\nClassification report:")
    print(classification_report(labels, preds,
                                 labels=all_labels_idx,
                                 target_names=class_names,
                                 zero_division=0))
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    print("Confusion matrix (rows=true, cols=pred):")
    header = "      " + "  ".join(f"{n[:5]:>5s}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {class_names[i][:5]:>5s}  " +
              "  ".join(f"{v:5d}" for v in row))
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"\n  Macro-F1 = {f1:.3f}")


def _plot_training_curves(history: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, history["train_loss"], label="train")
        axes[0].plot(epochs, history["val_loss"],   label="val")
        axes[0].set_title("Loss"); axes[0].legend()
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("CrossEntropy Loss")

        axes[1].plot(epochs, history["train_acc"], label="train")
        axes[1].plot(epochs, history["val_acc"],   label="val")
        axes[1].set_title("Accuracy"); axes[1].legend()
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")

        fig.tight_layout()
        path = out_dir / "training_curves.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved → {path}")
    except Exception as exc:
        print(f"  [warn] plot_training_curves: {exc}")


def _plot_confusion_matrix(labels: list, preds: list,
                            class_names: list, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        cm   = confusion_matrix(labels, preds,
                                 labels=list(range(len(class_names))))
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
        ax.set_title("Confusion Matrix — test set (6-class)")
        fig.tight_layout()
        path = out_dir / "confusion_matrix.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved → {path}")
    except Exception as exc:
        print(f"  [warn] plot_confusion_matrix: {exc}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(parse_args())
