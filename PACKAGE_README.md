# Claims Reserving Toolkit

An LSTM-based Python package for insurance claims reserving and ultimate loss prediction.

## Features

- **Multiple Model Architectures**: v1 (baseline), v2 (with attention)
- **Variable-Length Sequences**: Handles claims that settle at different speeds
- **Flexible Input**: Supports single or multi-feature inputs (e.g., payments + reserves)
- **Model Versioning**: Save/load models with full metadata tracking
- **Platform Agnostic**: Works on local machines, cloud notebooks, or any Python environment

## Installation

### From Source (Development)

```bash
cd /path/to/nlp-notes-summary-for-predictive-status
pip install -e .
```

### From Wheel (Production)

```bash
pip install claims_reserving_toolkit-0.1.0-py3-none-any.whl
```

## Quick Start

```python
from claims_reserving_toolkit import (
    ClaimReservingLSTM_v1,
    prepare_claim_sequences,
    ClaimDataset,
    collate_fn,
    save_claims_model,
    load_claims_model
)
import torch
from torch.utils.data import DataLoader

# 1. Prepare your claims data (platform-specific)
claims = [
    {
        'claim_id': 123,
        'ultimate': 5000,
        'payments': [0, 1000, 2000, 1500, 500],
        'reserves': [8000, 5000, 3000, 1000, 0],
        'settlement_period': 4
    },
    # ... more claims
]

# 2. Use the toolkit's generic preprocessing
train_data, test_data = prepare_claim_sequences(claims, train_split=0.8)

# 3. Create datasets
train_dataset = ClaimDataset(train_data)
test_dataset = ClaimDataset(test_data, scaler=train_dataset.scaler)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

# 4. Initialize model
model = ClaimReservingLSTM_v1(
    input_size=2,  # payments + reserves
    hidden_size=64,
    num_layers=2,
    dropout=0.2
)

# 5. Train your model
# ... your training loop ...

# 6. Save the model (you provide the directory)
save_claims_model(
    model=model,
    scaler=train_dataset.scaler,
    train_dataset=train_dataset,
    test_metrics=results,
    hyperparameters={'input_size': 2, 'hidden_size': 64, ...},
    save_dir="./my_models",  # ← Platform-specific path
    model_name="my_experiment_v1"
)

# 7. Load the model later
loaded_model, scaler, metadata = load_claims_model(
    model_dir="./my_models/my_experiment_v1_20250103_120000",
    model_class=ClaimReservingLSTM_v1,
    device='cpu'
)
```

## Model Architectures

### ClaimReservingLSTM_v1 (Baseline)
- Simple LSTM with 2-layer FC prediction heads
- Fast training, good baseline performance
- Best for: Initial experiments, limited data

### ClaimReservingLSTM_v2 (With Attention)
- Attention mechanism over LSTM outputs
- Deeper prediction heads (3 layers)
- Best for: Large datasets (5000+ claims), performance optimization

## What's Platform-Specific (Not in Package)

The package intentionally does NOT include:
- Data loading from specific file formats (CSV, Parquet, databases)
- File path configurations (use your own BASE_DATA_DIR)
- Business logic for filtering claims
- Visualization code

You handle these in your notebooks/scripts, then pass the standardized `claims` list to the toolkit.

## Package Structure

```
claims_reserving_toolkit/
├── models/
│   └── architectures.py      # Model definitions
├── data/
│   ├── dataset.py             # ClaimDataset, collate_fn
│   └── preprocessing.py       # prepare_claim_sequences
├── io/
│   └── model_io.py            # save/load functions
└── utils/
    └── (future utilities)
```

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests (future)
pytest tests/

# Build wheel
python -m build
```

## License

MIT

## Author

SF

