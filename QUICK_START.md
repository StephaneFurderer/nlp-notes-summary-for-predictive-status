# Quick Start Guide - 5 Minutes

## 1. Install the Package (30 seconds)

```bash
cd /Users/sf/Applications/nlp-notes-summary-for-predictive-status
pip install -e .
```

## 2. Update Your Notebook Imports (1 minute)

**Replace this:**
```python
from model_architecture import ClaimReservingLSTM_v1
from save_model import save_claims_model, load_claims_model
```

**With this:**
```python
from claims_reserving_toolkit import (
    ClaimReservingLSTM_v1,
    ClaimDataset,
    collate_fn,
    prepare_claim_sequences,
    save_claims_model,
    load_claims_model
)
```

## 3. Your Data Format (1 minute)

The package expects this standard format:

```python
claims = [
    {
        'claim_id': 123,
        'ultimate': 5000.0,
        'payments': [0, 1000, 2000, 1500, 500],
        'reserves': [8000, 5000, 3000, 1000, 0],  # Optional
        'settlement_period': 4
    },
    # ... more claims
]
```

Convert YOUR data to this format, then pass to the package.

## 4. Basic Usage (2 minutes)

```python
# Prepare sequences
train_data, test_data = prepare_claim_sequences(claims, train_split=0.8)

# Create datasets
train_dataset = ClaimDataset(train_data)
test_dataset = ClaimDataset(test_data, scaler=train_dataset.scaler)

# Create data loaders
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Initialize model
model = ClaimReservingLSTM_v1(input_size=2, hidden_size=64, num_layers=2, dropout=0.2)

# Train (your loop)
# ...

# Save
save_claims_model(
    model=model,
    scaler=train_dataset.scaler,
    train_dataset=train_dataset,
    test_metrics=results,
    hyperparameters={'input_size': 2, ...},
    save_dir="./my_models",
    model_name="experiment_v1"
)

# Load later
loaded_model, scaler, metadata = load_claims_model(
    model_dir="./my_models/experiment_v1_20250103_120000",
    model_class=ClaimReservingLSTM_v1,
    device='cpu'
)
```

## 5. Done! 🎉

Your notebook now uses the package. Everything is cleaner and reusable.

## Common Questions

**Q: Do I need to rebuild after changing the package?**  
A: No! Editable install (`pip install -e .`) means changes reflect immediately.

**Q: Where do I save models?**  
A: Anywhere YOU want! Pass `save_dir="./my_models"` or `"s3://bucket/"` or wherever.

**Q: What if I need a new feature?**  
A: Add it to the package files, it'll work immediately (editable install).

**Q: Can I use this on Colab/Databricks?**  
A: Yes! Upload the `.whl` file and `pip install claims_reserving_toolkit-0.1.0-py3-none-any.whl`.

**Q: What stays in my notebook?**  
A: Data loading, filtering, training loop, plots - anything platform-specific.

---

**Need more details?** See:
- `USAGE_IN_NOTEBOOKS.md` - Full examples
- `PACKAGE_SUMMARY.md` - Architecture and design
- `PACKAGE_README.md` - Package documentation

