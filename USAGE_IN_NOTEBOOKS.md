# How to Use the Package in Your Notebooks

## Installation

### Option 1: Install in Development Mode (Recommended for now)

```bash
# In your terminal
cd /Users/sf/Applications/nlp-notes-summary-for-predictive-status
pip install -e .
```

This installs the package in "editable" mode, so any changes you make to the package code are immediately reflected.

### Option 2: Install from Wheel (For production or sharing)

```bash
pip install /Users/sf/Applications/nlp-notes-summary-for-predictive-status/dist/claims_reserving_toolkit-0.1.0-py3-none-any.whl
```

## Update Your Existing Notebook

Replace your old imports with the package imports:

### Old Way (Before Package)

```python
# Cell 1: Old imports from local files
from model_architecture import ClaimReservingLSTM_v1
from save_model import save_claims_model, load_claims_model

# Cell 2: Dataset class defined in notebook
class ClaimDataset(torch.utils.data.Dataset):
    # ... 50 lines of code ...

# Cell 3: Preprocessing function in notebook
def prepare_claim_sequences(claims_data, train_split=0.8):
    # ... 50 lines of code ...
```

### New Way (With Package)

```python
# Cell 1: Import from the package
from claims_reserving_toolkit import (
    ClaimReservingLSTM_v1,
    ClaimReservingLSTM_v2,
    ClaimDataset,
    collate_fn,
    prepare_claim_sequences,
    save_claims_model,
    load_claims_model,
    list_saved_models
)
import torch
from torch.utils.data import DataLoader

# That's it! All utilities are imported
```

## Full Example for Your Notebook

```python
# ==============================================================================
# CELL 1: IMPORTS
# ==============================================================================
from claims_reserving_toolkit import (
    ClaimReservingLSTM_v1,
    ClaimDataset,
    collate_fn,
    prepare_claim_sequences,
    save_claims_model,
    load_claims_model
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os

# Your platform-specific imports
from helpers.functions.CONST import BASE_DATA_DIR

# ==============================================================================
# CELL 2: LOAD DATA (Platform-Specific - YOU provide this)
# ==============================================================================
evaluation_date = "2025-09-21"
df_periods = pd.read_parquet(
    os.path.join(BASE_DATA_DIR, evaluation_date, "closed_txn_to_periods.parquet")
)

# Your filtering logic
df_filtered = df_periods[df_periods['clmStatus'].isin(['PAID', 'CLOSED'])]

# Convert to standard claims format expected by the package
claims = []
for claim_id, group in df_filtered.groupby("clmNum"):
    claims.append({
        'claim_id': claim_id,
        'ultimate': group["paid"].sum(),
        'payments': group["paid"].tolist(),
        'reserves': group["reserve"].tolist(),  # If you have reserves
        'settlement_period': group["period"].max()
    })

print(f"Loaded {len(claims)} claims")

# ==============================================================================
# CELL 3: PREPARE DATA (Package handles this generically)
# ==============================================================================
train_data, test_data = prepare_claim_sequences(claims, train_split=0.8)

train_dataset = ClaimDataset(train_data)
test_dataset = ClaimDataset(test_data, scaler=train_dataset.scaler)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# ==============================================================================
# CELL 4: INITIALIZE MODEL (From package)
# ==============================================================================
hyperparameters = {
    'input_size': 2,  # payments + reserves
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'epochs': 50,
    'batch_size': 32,
    'lr': 0.001,
    'optimizer': 'Adam',
}

model = ClaimReservingLSTM_v1(
    input_size=hyperparameters['input_size'],
    hidden_size=hyperparameters['hidden_size'],
    num_layers=hyperparameters['num_layers'],
    dropout=hyperparameters['dropout']
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# ==============================================================================
# CELL 5: TRAINING (Your existing code)
# ==============================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr'])
criterion_remaining = nn.MSELoss()
criterion_ultimate = nn.MSELoss()

train_losses = []
test_losses = []

for epoch in range(hyperparameters['epochs']):
    model.train()
    epoch_loss = 0
    
    for X_batch, y_remaining, y_ultimate, lengths, _ in train_loader:
        X_batch = X_batch.to(device)
        y_remaining = y_remaining.to(device)
        y_ultimate = y_ultimate.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        pred_remaining, pred_ultimate = model(X_batch, lengths)
        
        loss_remaining = criterion_remaining(pred_remaining, y_remaining)
        loss_ultimate = criterion_ultimate(pred_ultimate, y_ultimate)
        loss = loss_remaining + loss_ultimate
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss / len(train_loader))
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_remaining, y_ultimate, lengths, _ in test_loader:
            X_batch = X_batch.to(device)
            y_remaining = y_remaining.to(device)
            y_ultimate = y_ultimate.to(device)
            lengths = lengths.to(device)
            
            pred_remaining, pred_ultimate = model(X_batch, lengths)
            loss = criterion_remaining(pred_remaining, y_remaining) + \
                   criterion_ultimate(pred_ultimate, y_ultimate)
            val_loss += loss.item()
    
    test_losses.append(val_loss / len(test_loader))
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{hyperparameters['epochs']}: "
              f"Train Loss={train_losses[-1]:.4f}, Val Loss={test_losses[-1]:.4f}")

# ==============================================================================
# CELL 6: SAVE MODEL (Package handles this, YOU provide the directory)
# ==============================================================================
# Compute your test metrics
results = {
    2: {'r2_ult': 0.15, 'mae_ult': 12000},
    5: {'r2_ult': 0.42, 'mae_ult': 8500},
    # ... your metrics by period
}

save_path = save_claims_model(
    model=model,
    scaler=train_dataset.scaler,
    train_dataset=train_dataset,
    test_metrics=results,
    hyperparameters=hyperparameters,
    save_dir="./my_models",  # ← YOU specify where to save
    model_name="feat_reserve_paid_v1"
)

print(f"Model saved to: {save_path}")

# ==============================================================================
# CELL 7: LOAD MODEL LATER (Package handles this)
# ==============================================================================
# List available models
from claims_reserving_toolkit import list_saved_models

models = list_saved_models(models_dir="./my_models")
for m in models:
    print(f"{m['folder']}: {m['model_class']}, input={m['input_size']}")

# Load a specific model
loaded_model, loaded_scaler, metadata = load_claims_model(
    model_dir=save_path,
    model_class=ClaimReservingLSTM_v1,  # Must match the saved model
    device='cpu'
)

# Use for prediction
with torch.no_grad():
    claim = claims[0]
    observe_until = 5
    
    observed_payments = claim['payments'][:observe_until]
    observed_reserves = claim['reserves'][:observe_until]
    
    # Prepare input
    import numpy as np
    X = np.column_stack([observed_payments, observed_reserves])
    X = loaded_scaler.transform(X)
    X = torch.FloatTensor(X).unsqueeze(0)
    length = torch.LongTensor([observe_until])
    
    pred_remaining, pred_ultimate = loaded_model(X, length)
    
    print(f"Predicted ultimate: ${pred_ultimate.item():,.2f}")
    print(f"Actual ultimate: ${claim['ultimate']:,.2f}")
```

## What Changed vs Before

### ✅ What the Package Now Handles

- Model architectures (v1, v2, future versions)
- Dataset class and collate function
- Data preprocessing (prepare_claim_sequences)
- Save/load with metadata tracking

### 🎯 What YOU Still Handle (Platform-Specific)

- Loading data from parquet/CSV/database
- Filtering claims by status, cause, date, etc.
- Setting file paths (BASE_DATA_DIR, model save directories)
- Training loop details
- Visualization and reporting

This separation makes your code portable across platforms!

