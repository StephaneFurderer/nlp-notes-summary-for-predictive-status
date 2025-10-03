"""
Claims Reserving Toolkit
========================

An LSTM-based toolkit for insurance claims reserving and ultimate loss prediction.

Main Components:
- Model architectures (v1, v2, ...)
- Dataset utilities for variable-length sequences
- Save/load functions with metadata tracking
- Data preprocessing for claim sequences

Example Usage:
    >>> from claims_reserving_toolkit import ClaimReservingLSTM_v1, save_claims_model
    >>> model = ClaimReservingLSTM_v1(input_size=2, hidden_size=64, num_layers=2)
    >>> # ... train model ...
    >>> save_claims_model(model, scaler, train_dataset, metrics, hyperparams, save_dir="./models")
"""

from .models.architectures import (
    ClaimReservingLSTM_v1,
    ClaimReservingLSTM_v2,
    ClaimReservingLSTM
)
from .data.dataset import ClaimDataset, collate_fn
from .data.preprocessing import prepare_claim_sequences
from .io.model_io import (
    save_claims_model,
    load_claims_model,
    list_saved_models
)

__version__ = "0.1.0"
__author__ = "SF"

__all__ = [
    # Models
    "ClaimReservingLSTM_v1",
    "ClaimReservingLSTM_v2",
    "ClaimReservingLSTM",
    # Data
    "ClaimDataset",
    "collate_fn",
    "prepare_claim_sequences",
    # I/O
    "save_claims_model",
    "load_claims_model",
    "list_saved_models",
]

