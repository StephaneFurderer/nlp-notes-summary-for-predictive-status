# Model Versioning Guide

## Overview

This guide explains how to manage multiple model architectures as your experiments evolve.

## File Structure

```
helpers/lab/
├── model_architecture.py    # All model versions defined here
├── save_model.py            # Save/load functions
└── MODEL_VERSIONING_GUIDE.md
```

## Available Model Versions

### `ClaimReservingLSTM_v1` (Current Default)
- Simple LSTM with 2-layer FC heads
- Good baseline performance
- Fast training

**Use when:**
- Starting a new experiment
- Need quick iterations
- Limited compute resources

### `ClaimReservingLSTM_v2`
- Adds attention mechanism over LSTM outputs
- Deeper FC heads (3 layers instead of 2)
- Separate dropout rates for LSTM vs FC layers

**Use when:**
- v1 plateaus in performance
- You have sufficient training data (>5000 claims)
- Want to identify which periods are most important

## Workflow Examples

### Example 1: Train with v1, Compare with v2

```python
# In your notebook
from model_architecture import ClaimReservingLSTM_v1, ClaimReservingLSTM_v2
from save_model import save_claims_model, load_claims_model, list_saved_models

# === Train v1 ===
model_v1 = ClaimReservingLSTM_v1(
    input_size=2,
    hidden_size=64,
    num_layers=2,
    dropout=0.2
)

# ... train model_v1 ...

# Save v1
save_path_v1 = save_claims_model(
    model=model_v1,
    scaler=train_dataset.scaler,
    train_dataset=train_dataset,
    test_metrics=results_v1,
    hyperparameters={'input_size': 2, 'hidden_size': 64, ...},
    model_name="011_feat_reserve_v1"
)

# === Train v2 ===
model_v2 = ClaimReservingLSTM_v2(
    input_size=2,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    fc_dropout=0.3  # v2-specific parameter
)

# ... train model_v2 ...

# Save v2
save_path_v2 = save_claims_model(
    model=model_v2,
    scaler=train_dataset.scaler,
    train_dataset=train_dataset,
    test_metrics=results_v2,
    hyperparameters={'input_size': 2, 'hidden_size': 64, ..., 'fc_dropout': 0.3},
    model_name="011_feat_reserve_v2"
)
```

### Example 2: List and Load Specific Version

```python
from model_architecture import ClaimReservingLSTM_v1, ClaimReservingLSTM_v2
from save_model import list_saved_models, load_claims_model

# List all saved models
models = list_saved_models()
for m in models:
    print(f"{m['folder']}")
    print(f"  Class: {m['model_class']}, Input: {m['input_size']}, Hidden: {m['hidden_size']}")
    print()

# Load a v1 model
model_v1, scaler_v1, metadata_v1 = load_claims_model(
    save_dir="models/011_feat_reserve_v1_20250102_143022",
    model_class=ClaimReservingLSTM_v1,
    device='cpu'
)

# Load a v2 model
model_v2, scaler_v2, metadata_v2 = load_claims_model(
    save_dir="models/011_feat_reserve_v2_20250102_151530",
    model_class=ClaimReservingLSTM_v2,
    device='cpu'
)

# Compare predictions
with torch.no_grad():
    claim = claims[0]
    X = prepare_input(claim, observe_until=5)  # Your preprocessing
    
    pred_remaining_v1, pred_ultimate_v1 = model_v1(X, lengths)
    pred_remaining_v2, pred_ultimate_v2 = model_v2(X, lengths)
    
    print(f"V1 prediction: ${pred_ultimate_v1.item():,.2f}")
    print(f"V2 prediction: ${pred_ultimate_v2.item():,.2f}")
    print(f"Actual: ${claim['ultimate']:,.2f}")
```

### Example 3: Load Without Knowing Version

```python
from model_architecture import ClaimReservingLSTM_v1, ClaimReservingLSTM_v2
from save_model import load_claims_model
import json

# Read metadata to determine version
save_dir = "models/011_feat_reserve_v1_20250102_143022"
with open(f"{save_dir}/metadata.json", 'r') as f:
    metadata = json.load(f)

# Map class name to actual class
MODEL_REGISTRY = {
    'ClaimReservingLSTM_v1': ClaimReservingLSTM_v1,
    'ClaimReservingLSTM_v2': ClaimReservingLSTM_v2,
    'ClaimReservingLSTM': ClaimReservingLSTM_v1,  # Backwards compatibility
}

model_class_name = metadata['architecture']['model_class']
model_class = MODEL_REGISTRY[model_class_name]

# Load with correct class
model, scaler, metadata = load_claims_model(save_dir, model_class)
```

## Adding a New Version

When you want to add `ClaimReservingLSTM_v3`:

1. **Add to `model_architecture.py`:**

```python
class ClaimReservingLSTM_v3(nn.Module):
    """
    Version 3: Your new improvements
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, new_param=0.5):
        super(ClaimReservingLSTM_v3, self).__init__()
        # Your new architecture
        pass
    
    def forward(self, x, lengths):
        # Your forward pass
        pass
```

2. **Optional: Update the default alias:**

```python
# At the bottom of model_architecture.py
ClaimReservingLSTM = ClaimReservingLSTM_v3  # Now v3 is default for new experiments
```

3. **Use in your notebook:**

```python
from model_architecture import ClaimReservingLSTM_v3

model = ClaimReservingLSTM_v3(
    input_size=2,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    new_param=0.5  # Your new parameter
)

# ... train ...

save_claims_model(
    model=model,
    scaler=train_dataset.scaler,
    train_dataset=train_dataset,
    test_metrics=results,
    hyperparameters={..., 'new_param': 0.5},  # Include new param
    model_name="011_feat_reserve_v3"
)
```

4. **Load it later:**

```python
from model_architecture import ClaimReservingLSTM_v3

model, scaler, metadata = load_claims_model(
    save_dir="models/011_feat_reserve_v3_20250103_120000",
    model_class=ClaimReservingLSTM_v3
)
```

## Best Practices

1. **Never delete old versions** - Keep v1, v2, etc. in `model_architecture.py` forever
2. **Document changes** - Add clear docstrings explaining what changed
3. **Version control** - Commit `model_architecture.py` changes with clear messages
4. **Name consistently** - Use pattern: `{experiment_name}_v{version}` (e.g., `011_feat_reserve_v2`)
5. **Track metadata** - Always save hyperparameters in the `save_claims_model` call
6. **Use the alias** - For new experiments, use `ClaimReservingLSTM` (points to latest version)
7. **Be explicit for production** - Use `ClaimReservingLSTM_v2` explicitly in production code

## Troubleshooting

### "Missing keys in state_dict"
- **Cause:** You're loading with the wrong model class
- **Fix:** Check `metadata.json` for `model_class` and use the correct version

### "Unexpected keys in state_dict"
- **Cause:** Same as above - architecture mismatch
- **Fix:** Load with the version that was used for training

### "Can't find old model version"
- **Cause:** Someone deleted the class from `model_architecture.py`
- **Fix:** Restore from git history: `git show HEAD~5:helpers/lab/model_architecture.py`

## Migration Guide

If you have old models saved before versioning:

```python
# Add this to load old models
def load_legacy_model(save_dir, device='cpu'):
    """Load models saved before versioning was implemented"""
    from model_architecture import ClaimReservingLSTM_v1
    
    # Assume old models were v1 architecture
    return load_claims_model(save_dir, ClaimReservingLSTM_v1, device)
```

