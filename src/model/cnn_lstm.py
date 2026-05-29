"""
cnn_lstm.py — CNN-LSTM for Lateral Eye Movement Detection
==========================================================
Input  : (batch, T, n_ics)   T = window_samples (e.g. 200 @ 200 Hz = 1 s)
                               n_ics = 58 independent components

Architecture (6-class multiclass)
----------------------------------
  Permute   → (batch, n_ics, T)           for Conv1d (channels-first)
  Block 1   Conv1d(58→32, k=5) – BN – GELU – MaxPool(2)     → (B, 32, T/2)
  Block 2   Conv1d(32→64, k=3) – BN – GELU – MaxPool(2)     → (B, 64, T/4)
  Block 3   Conv1d(64→128, k=3) – BN – GELU – MaxPool(2)    → (B, 128, T/8)
  Permute   → (batch, T/8, 128)           for LSTM (batch_first)
  LSTM      (128→hidden=128, 1 layer)
  Take last hidden → (batch, 128)
  FC head   Dropout(0.4) → Linear(128→n_classes)  (raw logits)
  Train with CrossEntropyLoss

Output : (batch, n_classes)  raw logits
"""

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Building block
# ─────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → GELU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 pool: int = 2, p_drop: float = 0.1):
        super().__init__()
        pad = kernel // 2           # 'same' padding
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.MaxPool1d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────

class LateralEyeCNNLSTM(nn.Module):
    """
    Parameters
    ----------
    n_ics       : number of input features (ICs) — default 58
    n_classes   : number of output classes   — default 6
    window_len  : number of time steps in each window — default 200
    conv_drop   : dropout inside conv blocks
    lstm_hidden : hidden size of LSTM  (128 — compact)
    lstm_layers : stacked LSTM layers  (1 — avoids overfitting)
    lstm_drop   : dropout between LSTM layers (unused when layers=1)
    fc_drop     : dropout before output linear
    """

    def __init__(
        self,
        n_ics:       int   = 58,
        n_classes:   int   = 6,
        window_len:  int   = 200,
        conv_drop:   float = 0.15,
        lstm_hidden: int   = 128,
        lstm_layers: int   = 1,
        lstm_drop:   float = 0.0,
        fc_drop:     float = 0.40,
    ):
        super().__init__()
        self.n_ics      = n_ics
        self.n_classes  = n_classes
        self.window_len = window_len

        # ── CNN encoder (lightweight) ────────────────────────
        # Input for Conv1d must be (B, C, T)  →  permuted inside forward()
        self.cnn = nn.Sequential(
            ConvBlock(n_ics, 32, kernel=5, pool=2, p_drop=conv_drop),  # T→T/2
            ConvBlock(32,    64, kernel=3, pool=2, p_drop=conv_drop),  # T/2→T/4
            ConvBlock(64,   128, kernel=3, pool=2, p_drop=conv_drop),  # T/4→T/8
        )
        # After 3× MaxPool(2): T → T/8 time steps, 128 feature maps

        # ── LSTM temporal modelling (single layer) ────────────
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_drop if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        # ── Classification head (compact) ─────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(fc_drop),
            nn.Linear(lstm_hidden, n_classes),   # raw logits
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    # ── Forward pass ─────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, C)   T = window length, C = n_ics
        returns logits (B, n_classes)
        """
        # (B, T, C) → (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)           # (B, n_ics, T)

        # CNN
        x = self.cnn(x)                  # (B, 128, T/8)

        # (B, 128, T/8) → (B, T/8, 128) for LSTM
        x = x.permute(0, 2, 1)           # (B, T/8, 128)

        # LSTM — take output at final time step
        _, (h_n, _) = self.lstm(x)       # h_n: (layers, B, hidden)
        x = h_n[-1]                      # last layer: (B, hidden)

        # FC head
        return self.classifier(x)        # (B, n_classes)


# ─────────────────────────────────────────────────────────────
# Loss helper
# ─────────────────────────────────────────────────────────────

def build_loss(y_train: np.ndarray,
               n_classes: int = 6) -> nn.CrossEntropyLoss:
    """
    CrossEntropyLoss with per-class inverse-frequency weights to handle
    imbalanced class distribution across the 6 eye-movement types.
    """
    counts = np.bincount(y_train.astype(int), minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)   # avoid div-by-zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes  # scale so mean weight = 1
    print("[build_loss] class weights:")
    for i, w in enumerate(weights):
        print(f"  class {i}: n={int(counts[i]):4d}  weight={w:.3f}")
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32)
    )


# ─────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T, C, K = 8, 200, 58, 6
    x = torch.randn(B, T, C)

    model = LateralEyeCNNLSTM(n_ics=C, n_classes=K, window_len=T)
    out   = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}   (logits, 6 classes)")
    print(f"Preds  : {out.argmax(dim=1).tolist()}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {n_params:,}")
