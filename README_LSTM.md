## Insurance Claims LSTM (PyTorch) - Quick Start

This guide explains how to run the Streamlit app, how the per-period claims data is formatted, and why the flattened format is used for the LSTM workflow.

### Prerequisites

- Python 3.10+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch appropriate for your platform (CUDA, CPU, or MPS). This app is PyTorch-only and does not include scikit-learn fallbacks.

### Data Input

The app expects a per-period claims dataset in parquet format with at least these columns:

- `clmNum`: claim identifier
- `period`: integer development period (0-based)
- `paid`: payment amount for that period

By default, `claims_lstm_pytorch_only.py` reads from `helpers.functions.CONST.BASE_DATA_DIR` and a hardcoded `extraction_date` folder containing `closed_txn_to_periods.parquet`.

Update the `extraction_date` or source path in the code if your data lives elsewhere.

### Data Flattening

The app converts the long-form per-period rows into a single row per claim via:

- `flatten_claims_periods(df_periods, max_periods=60, claim_id_col='clmNum', period_col='period', payment_col='paid')`

This produces a wide schema per claim:

- `claim_id`, `total_payments`, `n_payments`, `first_payment_period`, `last_payment_period`
- `cumulative_period_0..59` (running totals)
- `payment_period_0..59` (per-period increments)

Notes:
- `first_payment_period` is set to `60` when a claim has no payments.
- `last_payment_period` is `0` when a claim has no payments.
- Period values are clipped into `[0, 59]` to maintain fixed length.

### Why flattened format?

- Fixed-length sequences: LSTMs (and batching on GPU) work best with uniform shapes; using 60 periods per claim eliminates ragged sequences/padding complexity.
- Clear targets: `payment_period_*` are increments, `cumulative_period_*` are running totals—both enable different training objectives.
- Efficient dataloading: Wide numeric arrays map directly to tensors with minimal runtime preprocessing.
- Simple masking/edge cases: Deterministic handling for claims with no payments (`first=60`, `last=0`).
- Easy feature engineering: Period-wise operations (lags, deltas, windows) are straightforward on the wide layout.

### Running the App

From the project root (PyTorch-only optimized app):

```bash
streamlit run claims_lstm_pytorch_only.py
```

In the UI:

1. The app loads and flattens your per-period parquet into the wide schema.
2. Set the lookback window and model parameters in the sidebar.
3. Click “Train Model” to run the PyTorch LSTM (AMP-enabled when CUDA is available).
4. Review training/validation curves, accuracy, and analysis tabs.
5. Optionally save and reload models from the UI.

### Programmatic Snippet

If you need the flattened dataframe outside the UI:

```python
import pandas as pd
from claims_lstm_pytorch_only import flatten_claims_periods

df_periods = pd.read_parquet("/path/to/closed_txn_to_periods.parquet")
df_flat = flatten_claims_periods(df_periods, max_periods=60)
```

### Troubleshooting

- Missing third-party packages: reinstall with `pip install -r requirements.txt` (ensure correct virtual env).
- PyTorch not available: install a compatible PyTorch build for your OS/accelerator.
- Different period horizons: pass a different `max_periods` to `flatten_claims_periods` and ensure the rest of the pipeline expects the same length.


