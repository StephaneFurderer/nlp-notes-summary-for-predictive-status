"""
Insurance Claims LSTM - Paper-aligned Dual-Head Model (PyTorch)

Implements the architecture described in the attached paper:
 - Inputs X_{k,j}: dynamic only (for now): past standardized increments, past occurrences,
   development period j/n, observed flag.
 - Outputs per step j->j+1: p_hat (probability of non-zero payment), y_hat_star (standardized amount).
 - Loss: multitask with task-dependent uncertainty weighting (learnable), with masking on amount loss for I=1.
 - Horizon n=60; no sliding window; sequence model predicts next period at each step.
"""

import os
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import mean_squared_error, roc_auc_score, mean_absolute_error
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


HORIZON_N = 60


def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def flatten_claims_periods(
    df_periods: pd.DataFrame,
    max_periods: int = HORIZON_N,
    claim_id_col: str = 'clmNum',
    period_col: str = 'period',
    payment_col: str = 'paid'
) -> pd.DataFrame:
    """Flatten per-period rows into wide per-claim rows with payment and cumulative vectors."""
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
    return df_flat


def build_dynamic_inputs(df_flat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sequences: increments, occurrences, j/n, observed flags, and mask for targets.

    Returns
    -------
    increments : [B, N]
    occurrences : [B, N] in {0,1}
    j_over_n : [N]
    observed : [B, N] (currently ones; can be adapted)
    target_mask : [B, N-1] mask for j+1 targets (1 if j+1 target exists)
    """
    B = len(df_flat)
    N = HORIZON_N
    pay_cols = [f'payment_period_{i}' for i in range(N)]
    increments = df_flat[pay_cols].to_numpy(dtype=float)
    occurrences = (increments != 0.0).astype(float)

    # Assume all periods up to N are observed for now (can be adapted to evaluation date)
    observed = np.ones_like(increments, dtype=float)

    # Mask for targets at j+1: here all but last step
    target_mask = np.ones((B, N - 1), dtype=float)

    j_over_n = np.arange(N, dtype=float) / float(N)

    return increments, occurrences, j_over_n, observed, target_mask


def standardize_nonzero(train_increments: np.ndarray, increments: np.ndarray) -> Tuple[np.ndarray, float, float]:
    nonzero = train_increments[train_increments != 0.0]
    if nonzero.size == 0:
        mu, sigma = 0.0, 1.0
    else:
        mu = float(nonzero.mean())
        sigma = float(nonzero.std(ddof=0))
        if sigma == 0.0:
            sigma = 1.0
    standardized = (increments - mu) / sigma
    # For zero increments, keep standardized value consistent (still centered relative to mu)
    return standardized, mu, sigma


class DualHeadLSTM(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
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
        self.dropout = nn.Dropout(dropout)
        self.fc_amount = nn.Linear(hidden_size, 1)
        self.fc_prob = nn.Linear(hidden_size, 1)
        # task uncertainties (log variances) for weighting
        self.log_sigma_reg = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_cls = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # x: [B, T, F]
        B = x.size(0)
        h0 = x.new_zeros(self.num_layers, B, self.hidden_size)
        c0 = x.new_zeros(self.num_layers, B, self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        y_hat_star = self.fc_amount(out).squeeze(-1)  # [B, T]
        p_hat = torch.sigmoid(self.fc_prob(out)).squeeze(-1)  # [B, T]
        return y_hat_star, p_hat


def build_batches(
    inc_std: np.ndarray,
    occ: np.ndarray,
    j_over_n: np.ndarray,
    observed: np.ndarray,
    target_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
):
    # Build inputs X[:, :N-1, F] with features at steps j=0..N-2 to predict j+1
    B, N = inc_std.shape
    T = N - 1
    inc_feat = inc_std[:, :T]
    occ_feat = occ[:, :T]
    obs_feat = observed[:, :T]
    j_feat = np.tile(j_over_n[:T], (B, 1))
    X = np.stack([inc_feat, occ_feat, j_feat, obs_feat], axis=-1).astype(np.float32)

    # Targets at steps 1..N-1
    y_star = inc_std[:, 1:]
    I = occ[:, 1:]
    mask = target_mask  # [B, T]

    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y_star.astype(np.float32)),
        torch.from_numpy(I.astype(np.float32)),
        torch.from_numpy(mask.astype(np.float32)),
    )

    num_workers = max(1, (os.cpu_count() or 4) - 1)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
    )
    return loader


def train_model(
    model: DualHeadLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    alpha_cls: float = 1.0,
    teacher_force: bool = True,
):
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    bce = nn.BCELoss(reduction='none')
    mse = nn.MSELoss(reduction='none')

    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_losses = []
        # Probability to replace inputs with previous predictions increases over epochs
        replace_prob = min(0.8, (epoch + 1) / epochs * 0.8) if teacher_force else 0.0
        for X, y_star, I, mask in train_loader:
            X = X.to(device, non_blocking=True)
            y_star = y_star.to(device, non_blocking=True)
            I = I.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Optional replacement: use previous predictions for inc/occ channels
            if replace_prob > 0.0:
                with torch.no_grad():
                    y_hat_star_tmp, p_hat_tmp = model(X)
                B, T, F = X.shape
                # indices: 0=inc, 1=occ
                if T > 1:
                    # shift predictions to align with inputs at j (use preds of j-1)
                    inc_pred = y_hat_star_tmp[:, :-1]
                    occ_pred = p_hat_tmp[:, :-1]
                    # stochastic replacement mask
                    rep_mask = (torch.rand_like(inc_pred) < replace_prob).float()
                    X[:, 1:, 0] = rep_mask * inc_pred + (1 - rep_mask) * X[:, 1:, 0]
                    X[:, 1:, 1] = rep_mask * occ_pred + (1 - rep_mask) * X[:, 1:, 1]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                y_hat_star, p_hat = model(X)  # [B, T]
                # Losses
                ce = bce(p_hat, I) * mask
                rl = mse(y_hat_star, y_star) * I * mask  # amount loss only where I=1
                # Task uncertainty weighting
                inv_var_reg = torch.exp(-model.log_sigma_reg * 2)
                inv_var_cls = torch.exp(-model.log_sigma_cls * 2)
                loss = inv_var_reg * rl.mean() + alpha_cls * inv_var_cls * ce.mean() \
                       + model.log_sigma_reg * 2 + model.log_sigma_cls * 2

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.detach().item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y_star, I, mask in val_loader:
                X = X.to(device)
                y_star = y_star.to(device)
                I = I.to(device)
                mask = mask.to(device)
                y_hat_star, p_hat = model(X)
                ce = bce(p_hat, I) * mask
                rl = mse(y_hat_star, y_star) * I * mask
                inv_var_reg = torch.exp(-model.log_sigma_reg * 2)
                inv_var_cls = torch.exp(-model.log_sigma_cls * 2)
                loss = inv_var_reg * rl.mean() + alpha_cls * inv_var_cls * ce.mean() \
                       + model.log_sigma_reg * 2 + model.log_sigma_cls * 2
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        #st.write(f"Epoch {epoch+1}/{epochs} - train {train_loss:.5f} - val {val_loss:.5f} - {time.time()-t0:.1f}s")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def save_paper_model(model: DualHeadLSTM, params: dict) -> str:
    os.makedirs('models', exist_ok=True)
    ts = int(time.time())
    path = os.path.join('models', f'paper_dual_head_{ts}.pth')
    torch.save({'state_dict': model.state_dict(), 'params': params}, path)
    return path


def list_paper_models():
    if not os.path.exists('models'):
        return []
    files = [f for f in os.listdir('models') if f.startswith('paper_dual_head_') and f.endswith('.pth')]
    files.sort()
    return [os.path.join('models', f) for f in files]


def load_paper_model(path: str, device: torch.device):
    blob = torch.load(path, map_location=device)
    params = blob.get('params', {})
    model = DualHeadLSTM(
        input_size=4,
        hidden_size=params.get('hidden_size', 64),
        num_layers=params.get('num_layers', 2),
        dropout=params.get('dropout', 0.1),
    ).to(device)
    model.load_state_dict(blob['state_dict'])
    model.eval()
    return model, params

def main():
    st.set_page_config(page_title="Claims LSTM - Paper Dual-Head", page_icon="📄", layout="wide")
    st.markdown('<h1>📄 Claims LSTM - Paper Dual-Head</h1>', unsafe_allow_html=True)

    with st.spinner("Loading claims data..."):
        df_flat = load_claims_data()
    if df_flat is None or df_flat.empty:
        st.error("No data loaded")
        st.stop()

    # Build dynamic sequences
    increments, occurrences, j_over_n, observed, target_mask = build_dynamic_inputs(df_flat)

    # Sidebar controls
    st.sidebar.header("Parameters")
    train_split = st.sidebar.slider("Training Split", 0.6, 0.95, 0.8, 0.05)
    hidden_size = st.sidebar.slider("Hidden Size", 32, 256, 64, 32)
    num_layers = st.sidebar.slider("Layers", 1, 4, 2, 1)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
    batch_size = st.sidebar.slider("Batch Size", 32, 512, 128, 32)
    epochs = st.sidebar.slider("Epochs", 10, 200, 50, 10)
    lr = st.sidebar.select_slider("LR", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
    alpha_cls = st.sidebar.slider("Alpha (classification scale)", 0.1, 5.0, 1.0, 0.1)

    # Split train/val by claims
    num_claims = increments.shape[0]
    train_count = int(train_split * num_claims)
    idx = np.arange(num_claims)
    np.random.seed(42)
    np.random.shuffle(idx)
    train_idx, val_idx = idx[:train_count], idx[train_count:]

    inc_train, inc_val = increments[train_idx], increments[val_idx]
    occ_train, occ_val = occurrences[train_idx], occurrences[val_idx]
    obs_train, obs_val = observed[train_idx], observed[val_idx]
    mask_train, mask_val = target_mask[train_idx], target_mask[val_idx]

    # Standardize using train set non-zero increments
    inc_train_std, mu, sigma = standardize_nonzero(inc_train, inc_train)
    inc_val_std, _, _ = standardize_nonzero(inc_train, inc_val)

    device = get_device()
    st.write(f"Device: {device}")

    # Build loaders (features at j predict j+1)
    train_loader = build_batches(inc_train_std, occ_train, j_over_n, obs_train, mask_train, batch_size, device)
    val_loader = build_batches(inc_val_std, occ_val, j_over_n, obs_val, mask_val, batch_size, device)

    model = DualHeadLSTM(input_size=4, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)

    # Tabs layout
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Data Overview", "🔧 Data Preparation", "🤖 Model Training", "🎯 Predictions", "📊 Analysis", "🔍 Individual Claims"]) 

    with tab1:
        st.write({"claims": int(num_claims), "horizon": HORIZON_N})
        st.dataframe(df_flat.head(10))

        # Payment frequency distribution
        n_payments_series = df_flat['n_payments'].value_counts().sort_index()
        fig_freq = go.Figure(data=[go.Bar(x=n_payments_series.index, y=n_payments_series.values, name='Payment Count')])
        fig_freq.update_layout(title='Payment Frequency Distribution', xaxis_title='Number of Payments', yaxis_title='Count')
        st.plotly_chart(fig_freq, use_container_width=True)

        # Incremental payment distribution (non-zero)
        pay_cols = [f'payment_period_{i}' for i in range(HORIZON_N)]
        all_incs = df_flat[pay_cols].to_numpy().flatten()
        nonzero_incs = all_incs[all_incs != 0]
        fig_inc = go.Figure(data=[go.Histogram(x=nonzero_incs, nbinsx=60)])
        fig_inc.update_layout(title='Incremental Payment Distribution (Non-zero)', xaxis_title='Amount', yaxis_title='Frequency')
        st.plotly_chart(fig_inc, use_container_width=True)

        # Payment timing distribution (first payment period among paid claims)
        paid_claims = df_flat[df_flat['n_payments'] > 0]
        fig_time = go.Figure(data=[go.Histogram(x=paid_claims['first_payment_period'], nbinsx=20)])
        fig_time.update_layout(title='First Payment Timing Distribution', xaxis_title='Period', yaxis_title='Count')
        st.plotly_chart(fig_time, use_container_width=True)

        # Sample claims cumulative payments
        cum_cols = [f'cumulative_period_{i}' for i in range(HORIZON_N)]
        sample = df_flat.sample(min(10, len(df_flat)), random_state=42)
        fig_cum = go.Figure()
        for _, r in sample.iterrows():
            fig_cum.add_trace(go.Scatter(x=list(range(HORIZON_N)), y=[r[c] for c in cum_cols], mode='lines', name=f"Claim {int(r['claim_id'])}", opacity=0.6))
        fig_cum.update_layout(title='Sample Claims (Cumulative Payments)', xaxis_title='Period', yaxis_title='Cumulative Amount')
        st.plotly_chart(fig_cum, use_container_width=True)

    with tab2:
        st.write("Standardization μ, σ computed on train non-zero increments")
        st.write({"mu": float(mu), "sigma": float(sigma)})

    with tab3:
        if st.button("🚀 Train Dual-Head Model", type="primary"):
            with st.spinner("Training..."):
                history = train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr, alpha_cls=alpha_cls)
            st.success("Training complete")
            st.line_chart({'train_loss': history['train_loss'], 'val_loss': history['val_loss']})
            st.session_state.paper_model = model
            st.session_state.paper_params = {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'alpha_cls': alpha_cls,
                'mu': mu,
                'sigma': sigma,
                'train_split': train_split,
            }

        # Save/load controls
        col1, col2 = st.columns(2)
        with col1:
            if 'paper_model' in st.session_state and st.button("💾 Save Model"):
                path = save_paper_model(st.session_state.paper_model, st.session_state.paper_params)
                st.success(f"Saved: {path}")
        with col2:
            models = list_paper_models()
            if models:
                choice = st.selectbox("Saved models", models)
                if st.button("📂 Load Selected"):
                    model_loaded, params = load_paper_model(choice, device)
                    st.session_state.paper_model = model_loaded
                    st.session_state.paper_params = params
                    st.success("Model loaded")

    with tab4:
        if 'paper_model' in st.session_state:
            m = st.session_state.paper_model
            m.eval()
            all_I, all_p, all_yhat = [], [], []
            with torch.no_grad():
                for X, y_star, I, mask in val_loader:
                    X = X.to(device)
                    y_hat_star, p_hat = m(X)
                    all_I.append(I.numpy())
                    all_p.append(p_hat.cpu().numpy())
                    all_yhat.append(y_hat_star.cpu().numpy())
            I_mat = np.concatenate(all_I)
            P_mat = np.concatenate(all_p)
            YS_mat = np.concatenate(all_yhat)
            expected_inc = P_mat * (YS_mat * sigma + mu)
            st.write("Expected increment sample (val):")
            st.write(np.round(expected_inc[:5, :10], 2))

    with tab5:
        if 'paper_model' in st.session_state:
            # AUROC per period
            m = st.session_state.paper_model
            m.eval()
            all_I, all_p = [], []
            with torch.no_grad():
                for X, y_star, I, mask in val_loader:
                    X = X.to(device)
                    _, p_hat = m(X)
                    all_I.append(I.numpy())
                    all_p.append(p_hat.cpu().numpy())
            I_mat = np.concatenate(all_I)
            P_mat = np.concatenate(all_p)
            aurocs = []
            for t in range(P_mat.shape[1]):
                y_true = I_mat[:, t]
                y_pred = P_mat[:, t]
                if len(np.unique(y_true)) > 1:
                    aurocs.append(roc_auc_score(y_true, y_pred))
                else:
                    aurocs.append(np.nan)
            st.write({f"AUROC_period_{t+1}": (float(a) if not np.isnan(a) else None) for t, a in enumerate(aurocs)})

    with tab6:
        if 'paper_model' in st.session_state:
            m = st.session_state.paper_model
            m.eval()
            # Available validation claim IDs
            val_claim_ids = df_flat.iloc[val_idx]['claim_id'].tolist()
            selected_id = st.selectbox("Select validation claim_id", val_claim_ids)
            sel_pos = val_claim_ids.index(selected_id)

            # Build single-claim features X[1, T, F]
            T = HORIZON_N - 1
            inc_feat = inc_val_std[sel_pos, :T]
            occ_feat = occ_val[sel_pos, :T]
            obs_feat = obs_val[sel_pos, :T]
            j_feat = j_over_n[:T]
            X_single = np.stack([inc_feat, occ_feat, j_feat, obs_feat], axis=-1).astype(np.float32)[None, ...]

            with torch.no_grad():
                X_t = torch.from_numpy(X_single).to(device)
                y_hat_star, p_hat = m(X_t)
                y_hat_star = y_hat_star.cpu().numpy()[0]
                p_hat = p_hat.cpu().numpy()[0]
                expected_inc = p_hat * (y_hat_star * sigma + mu)

            periods = np.arange(1, HORIZON_N)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=periods, y=expected_inc, name='Expected Increment'))
            fig.update_layout(title=f"Claim {selected_id} - Expected Increments", xaxis_title="Period", yaxis_title="Amount")
            st.plotly_chart(fig, use_container_width=True)

            st.write({
                'mean_expected': float(np.mean(expected_inc)),
                'sum_expected': float(np.sum(expected_inc)),
                'nonzero_prob_mean': float(np.mean(p_hat)),
            })

            # Actual vs Predicted cumulative
            cum_cols = [f'cumulative_period_{i}' for i in range(HORIZON_N)]
            actual_cum = df_flat[df_flat['claim_id'] == selected_id][cum_cols].iloc[0].to_numpy(dtype=float)
            pred_cum = np.concatenate([[0.0], np.cumsum(expected_inc)])
            fig_ap = go.Figure()
            fig_ap.add_trace(go.Scatter(x=np.arange(HORIZON_N), y=actual_cum, mode='lines+markers', name='Actual Cumulative'))
            fig_ap.add_trace(go.Scatter(x=np.arange(HORIZON_N), y=pred_cum, mode='lines+markers', name='Predicted Cumulative'))
            fig_ap.update_layout(title='Actual vs Predicted Cumulative', xaxis_title='Period', yaxis_title='Amount')
            st.plotly_chart(fig_ap, use_container_width=True)

            # Over/Under reserved scatter for all val claims at final period
            with torch.no_grad():
                all_pred_final, all_actual_final, claim_ids = [], [], []
                for i, cid in enumerate(val_claim_ids):
                    inc_feat = inc_val_std[i, :T]
                    occ_feat = occ_val[i, :T]
                    obs_feat = obs_val[i, :T]
                    X_i = np.stack([inc_feat, occ_feat, j_over_n[:T], obs_feat], axis=-1).astype(np.float32)[None, ...]
                    X_ti = torch.from_numpy(X_i).to(device)
                    yhs, ph = m(X_ti)
                    yhs = yhs.cpu().numpy()[0]
                    ph = ph.cpu().numpy()[0]
                    exp_inc_i = ph * (yhs * sigma + mu)
                    pred_final = float(np.sum(exp_inc_i))
                    actual_final = float(df_flat[df_flat['claim_id'] == cid][cum_cols].iloc[0].to_numpy(dtype=float)[-1])
                    all_pred_final.append(pred_final)
                    all_actual_final.append(actual_final)
                    claim_ids.append(int(cid))
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(x=all_actual_final, y=all_pred_final, mode='markers', text=claim_ids, name='Claims'))
            min_v = float(min(all_actual_final + all_pred_final))
            max_v = float(max(all_actual_final + all_pred_final))
            fig_sc.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines', name='Perfect', line=dict(color='red', dash='dash')))
            fig_sc.update_layout(title='Final Cumulative: Actual vs Predicted', xaxis_title='Actual', yaxis_title='Predicted')
            st.plotly_chart(fig_sc, use_container_width=True)


if __name__ == "__main__":
    main()


