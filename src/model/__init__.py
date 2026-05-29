"""
model package — CNN-LSTM architecture for 6-class eye movement classification.

Public API
----------
    LateralEyeCNNLSTM   — the model class
    build_loss          — factory for CrossEntropyLoss with class weights
"""

from .cnn_lstm import LateralEyeCNNLSTM, build_loss

__all__ = [
    "LateralEyeCNNLSTM",
    "build_loss",
]
