"""
Insurance Claims LSTM - PyTorch Only (Optimized)

This Streamlit app trains an LSTM to predict per-period payment increments on
flattened claims sequences. It removes scikit-learn fallbacks and focuses on
runtime efficiency:
 - Prebuilt tensors and efficient DataLoader
 - CUDA/MPS device support with AMP (autocast + GradScaler)
 - cuDNN benchmark and tuned DataLoader flags
 - Gradient clipping, scheduler, and early stopping
 - Reduced Streamlit UI overhead (batched updates)
"""

import os
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def flatten_claims_periods(
    df_periods: pd.DataFrame,
    max_periods: int = 60,
    claim_id_col: str = 'clmNum',
    period_col: str = 'period',
    payment_col: str = 'paid'
) -> pd.DataFrame:
    """Flatten per-period claim rows into wide per-claim rows.

    Output columns:
      claim_id,total_payments,n_payments,first_payment_period,last_payment_period,
      cumulative_period_0..{max-1},payment_period_0..{max-1}
    """
    if df_periods is None or len(df_periods) == 0:
        return pd.DataFrame(
            columns=(
                ['claim_id', 'total_payments', 'n_payments', 'first_payment_period', 'last_payment_period']
                + [f'cumulative_period_{i}' for i in range(max_periods)]
                + [f'payment_period_{i}' for i in range(max_periods)]
            )
        )

    required_cols = {claim_id_col, period_col, payment_col}
    missing = required_cols.difference(df_periods.columns)
    if missing:
        raise ValueError(f"Missing required columns in df_periods: {sorted(missing)}")

    df = df_periods[[claim_id_col, period_col, payment_col]].copy()
    df[period_col] = pd.to_numeric(df[period_col], errors='coerce').fillna(0).astype(int)
    df[period_col] = df[period_col].clip(lower=0, upper=max_periods - 1)
    df[payment_col] = pd.to_numeric(df[payment_col], errors='coerce').fillna(0.0).astype(float)

    claim_groups = df.groupby(claim_id_col, sort=False)
    num_claims = claim_groups.ngroups
    payments_matrix = np.zeros((num_claims, max_periods), dtype=float)

    for row_index, (_, g) in enumerate(claim_groups):
        per_period = g.groupby(period_col, as_index=False)[payment_col].sum()
        for _, r in per_period.iterrows():
            p_idx = int(r[period_col])
            if 0 <= p_idx < max_periods:
                payments_matrix[row_index, p_idx] += float(r[payment_col])

    cumulative_matrix = np.cumsum(payments_matrix, axis=1)
    total_payments = payments_matrix.sum(axis=1)
    n_payments = (payments_matrix > 0).sum(axis=1)

    first_payment_period = np.full(num_claims, max_periods, dtype=int)
    last_payment_period = np.zeros(num_claims, dtype=int)
    for i in range(num_claims):
        nz = np.nonzero(payments_matrix[i] > 0)[0]
        if nz.size > 0:
            first_payment_period[i] = int(nz.min())
            last_payment_period[i] = int(nz.max())

    columns = (
        ['claim_id', 'total_payments', 'n_payments', 'first_payment_period', 'last_payment_period']
        + [f'cumulative_period_{i}' for i in range(max_periods)]
        + [f'payment_period_{i}' for i in range(max_periods)]
    )
    out = np.concatenate(
        [
            np.arange(num_claims).reshape(-1, 1),
            total_payments.reshape(-1, 1),
            n_payments.reshape(-1, 1),
            first_payment_period.reshape(-1, 1),
            last_payment_period.reshape(-1, 1),
            cumulative_matrix,
            payments_matrix,
        ],
        axis=1,
    )
    df_flat = pd.DataFrame(out, columns=columns)
    df_flat['claim_id'] = df_flat['claim_id'].astype(int)
    df_flat['n_payments'] = df_flat['n_payments'].astype(int)
    df_flat['first_payment_period'] = df_flat['first_payment_period'].astype(int)
    df_flat['last_payment_period'] = df_flat['last_payment_period'].astype(int)
    return df_flat


@st.cache_data
def load_claims_data():
    from helpers.functions.CONST import BASE_DATA_DIR
    extraction_date = "2025-09-21"
    df_periods = pd.read_parquet(
        os.path.join(BASE_DATA_DIR, extraction_date, "closed_txn_to_periods.parquet")
    )
    df_flat = flatten_claims_periods(df_periods)
    return df_flat, None


def create_lstm_dataset(sequences, lookback=15):
    X, y = [], []
    for seq in sequences:
        for i in range(lookback, len(seq)):
            X.append(seq[i - lookback:i])
            y.append(seq[i] - seq[i - 1] if i > 0 else seq[i])
    return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = x.new_zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = x.new_zeros(self.num_layers, batch_size, self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def build_dataloaders(X_train, y_train, X_val, y_val, batch_size=128, device=torch.device('cpu')):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    num_workers = max(1, (os.cpu_count() or 4) - 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == 'cuda'),
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
    )
    return train_loader, val_loader


def train_lstm(
    model,
    train_loader,
    val_loader,
    device,
    epochs=50,
    learning_rate=1e-3,
    grad_clip=1.0,
    early_stop_patience=8,
):
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        steps = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out = model(xb).squeeze()
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            steps += 1
        train_loss = running / max(1, steps)

        # Validation
        model.eval()
        val_running = 0.0
        val_steps = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    out = model(xb).squeeze()
                    loss = criterion(out, yb)
                val_running += loss.item()
                val_steps += 1
        val_loss = val_running / max(1, val_steps)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Batched UI update
        st.write(f"Epoch {epoch+1}/{epochs} - train: {train_loss:.5f} - val: {val_loss:.5f} - time: {time.time()-t0:.1f}s")

        # Early stopping
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def main():
    st.set_page_config(page_title="Claims LSTM - PyTorch Only", page_icon="🏥", layout="wide")
    st.markdown('<h1>🏥 Insurance Claims LSTM - PyTorch Only</h1>', unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading claims data..."):
        claims_df, _ = load_claims_data()
    if claims_df is None or claims_df.empty:
        st.error("No data loaded.")
        st.stop()

    # Sidebar
    st.sidebar.header("Parameters")
    lookback = st.sidebar.slider("Lookback", 5, 30, 15, 5)
    train_split = st.sidebar.slider("Training Split", 0.6, 0.9, 0.8, 0.05)
    hidden_size = st.sidebar.slider("Hidden Size", 32, 256, 64, 32)
    num_layers = st.sidebar.slider("Layers", 1, 4, 2, 1)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
    epochs = st.sidebar.slider("Epochs", 10, 200, 50, 10)
    batch_size = st.sidebar.slider("Batch Size", 32, 512, 128, 32)
    learning_rate = st.sidebar.select_slider("LR", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)

    # Prepare sequences from cumulative payments
    cumulative_cols = [c for c in claims_df.columns if c.startswith('cumulative_period_')]
    sequences = [ [row[c] for c in cumulative_cols] for _, row in claims_df.iterrows() ]

    # Scale sequences claim-wise for stability
    scaler = MinMaxScaler()
    sequences_scaled = []
    for seq in sequences:
        seq_arr = np.array(seq).reshape(-1, 1)
        scaled_seq = scaler.fit_transform(seq_arr).flatten()
        sequences_scaled.append(scaled_seq)

    # Create supervised dataset
    X, y = create_lstm_dataset(sequences_scaled, lookback)
    if len(X) == 0:
        st.error("Insufficient sequence length for the chosen lookback.")
        st.stop()

    X = X.reshape(X.shape[0], X.shape[1], 1)
    split_idx = int(train_split * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    device = get_device()
    st.write(f"Device: {device}")

    # Build model and loaders
    model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    train_loader, val_loader = build_dataloaders(X_train, y_train, X_val, y_val, batch_size=batch_size, device=device)

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training..."):
            history = train_lstm(
                model, train_loader, val_loader, device,
                epochs=epochs, learning_rate=learning_rate
            )
        st.success("Training complete")
        st.line_chart({'train_loss': history['train_loss'], 'val_loss': history['val_loss']})

        # Quick validation metrics on last batch
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb).squeeze().detach().cpu().numpy()
                preds.append(out)
                trues.append(yb.numpy())
        preds = np.concatenate(preds) if preds else np.array([])
        trues = np.concatenate(trues) if trues else np.array([])
        if len(preds) and len(trues):
            mse = mean_squared_error(trues, preds)
            mae = mean_absolute_error(trues, preds)
            rmse = float(np.sqrt(mse))
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE", f"{mse:.6f}")
            col2.metric("MAE", f"{mae:.6f}")
            col3.metric("RMSE", f"{rmse:.6f}")


if __name__ == "__main__":
    main()


