"""
Claims LSTM - Paper Dual-Head on Dummy CSV Data

Loads the flattened dummy CSV used by the original PyTorch app and runs the
paper-aligned dual-head LSTM (occurrence + amount) to report performance.
"""

import glob
import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import roc_auc_score
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from claims_lstm_pytorch_paper import (
    HORIZON_N,
    get_device,
    build_dynamic_inputs,
    standardize_nonzero,
    DualHeadLSTM,
    build_batches,
    train_model,
    save_paper_model,
    list_paper_models,
    load_paper_model,
)


@st.cache_data
def load_dummy_flattened():
    csv_files = glob.glob("claims_data_period_flattened_*.csv")
    if not csv_files:
        return None
    latest = max(csv_files, key=lambda x: x.split('_')[-1].split('.')[0])
    df = pd.read_csv(latest)
    return df


def main():
    st.set_page_config(page_title="Paper Dual-Head - Dummy Data", page_icon="🧪", layout="wide")
    st.markdown('<h1>🧪 Paper Dual-Head - Dummy Data</h1>', unsafe_allow_html=True)

    df_flat = load_dummy_flattened()
    if df_flat is None or df_flat.empty:
        st.error("No dummy flattened CSV found. Please run the generator first.")
        st.stop()

    st.sidebar.header("Parameters")
    lookback = st.sidebar.slider("Lookback Window", 5, 30, 15, 5)
    train_split = st.sidebar.slider("Training Split", 0.6, 0.95, 0.8, 0.05)
    hidden_size = st.sidebar.slider("Hidden Size", 32, 256, 64, 32)
    num_layers = st.sidebar.slider("Layers", 1, 4, 2, 1)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
    batch_size = st.sidebar.slider("Batch Size", 32, 512, 128, 32)
    epochs = st.sidebar.slider("Epochs", 10, 200, 50, 10)
    lr = st.sidebar.select_slider("LR", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
    alpha_cls = st.sidebar.slider("Alpha (classification scale)", 0.1, 5.0, 1.0, 0.1)

    # Build dynamic inputs
    increments, occurrences, j_over_n, observed, target_mask = build_dynamic_inputs(df_flat)

    # Split
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

    # Standardize by train non-zero
    inc_train_std, mu, sigma = standardize_nonzero(inc_train, inc_train)
    inc_val_std, _, _ = standardize_nonzero(inc_train, inc_val)

    device = get_device()
    st.write(f"Device: {device}")

    # Build lookback-based batches instead of full sequences
    def build_lookback_batches(inc_std, occ, j_over_n, observed, target_mask, batch_size, device, lookback):
        B, N = inc_std.shape
        X_list, y_star_list, I_list, mask_list = [], [], [], []
        
        for i in range(B):
            for j in range(lookback, N):
                # Input: last lookback periods
                inc_feat = inc_std[i, j-lookback:j]
                occ_feat = occ[i, j-lookback:j]
                obs_feat = observed[i, j-lookback:j]
                j_feat = j_over_n[j-lookback:j]
                X = np.stack([inc_feat, occ_feat, j_feat, obs_feat], axis=-1).astype(np.float32)
                
                # Target: next period
                y_star = inc_std[i, j]
                I = occ[i, j]
                mask = target_mask[i, j-1] if j > 0 else 1.0
                
                X_list.append(X)
                y_star_list.append(y_star)
                I_list.append(I)
                mask_list.append(mask)
        
        if not X_list:
            return None
            
        X_tensor = torch.from_numpy(np.array(X_list))
        y_star_tensor = torch.from_numpy(np.array(y_star_list).astype(np.float32))
        I_tensor = torch.from_numpy(np.array(I_list).astype(np.float32))
        mask_tensor = torch.from_numpy(np.array(mask_list).astype(np.float32))
        
        ds = TensorDataset(X_tensor, y_star_tensor, I_tensor, mask_tensor)
        num_workers = max(1, (os.cpu_count() or 4) - 1)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            pin_memory=(device.type == 'cuda'), num_workers=num_workers,
            persistent_workers=True, prefetch_factor=2, drop_last=False
        )
        return loader
    
    train_loader = build_lookback_batches(inc_train_std, occ_train, j_over_n, obs_train, mask_train, batch_size, device, lookback)
    val_loader = build_lookback_batches(inc_val_std, occ_val, j_over_n, obs_val, mask_val, batch_size, device, lookback)

    model = DualHeadLSTM(input_size=4, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)

    # Training function tailored for lookback windows (uses last timestep only)
    def train_model_lookback(model, train_loader, val_loader, device, epochs=50, lr=1e-3, alpha_cls=1.0):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        bce = nn.BCELoss(reduction='none')
        mse = nn.MSELoss(reduction='none')
        history = {'train_loss': [], 'val_loss': []}
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for X, y_star, I, mask in train_loader:
                X = X.to(device)
                y_star = y_star.to(device)
                I = I.to(device)
                mask = mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                y_hat_star, p_hat = model(X)
                y_last = y_hat_star[:, -1]
                p_last = p_hat[:, -1]
                ce = bce(p_last, I) * mask
                rl = mse(y_last, y_star) * I * mask
                inv_var_reg = torch.exp(-model.log_sigma_reg * 2)
                inv_var_cls = torch.exp(-model.log_sigma_cls * 2)
                loss = inv_var_reg * rl.mean() + alpha_cls * inv_var_cls * ce.mean() \
                       + model.log_sigma_reg * 2 + model.log_sigma_cls * 2
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
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
                    y_last = y_hat_star[:, -1]
                    p_last = p_hat[:, -1]
                    ce = bce(p_last, I) * mask
                    rl = mse(y_last, y_star) * I * mask
                    inv_var_reg = torch.exp(-model.log_sigma_reg * 2)
                    inv_var_cls = torch.exp(-model.log_sigma_cls * 2)
                    loss = inv_var_reg * rl.mean() + alpha_cls * inv_var_cls * ce.mean() \
                           + model.log_sigma_reg * 2 + model.log_sigma_cls * 2
                    val_losses.append(loss.item())
            scheduler.step()
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
        return history

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Data Overview", "🔧 Data Preparation", "🤖 Model Training", "🎯 Predictions", "📊 Analysis", "🔍 Individual Claims"]) 

    with tab1:
        st.write({"claims": int(len(df_flat)), "horizon": HORIZON_N})
        st.dataframe(df_flat.head(10))

        # Payment frequency
        n_pay = df_flat['n_payments'].value_counts().sort_index()
        st.plotly_chart(go.Figure(data=[go.Bar(x=n_pay.index, y=n_pay.values)]).update_layout(title='Payment Frequency', xaxis_title='Payments', yaxis_title='Count'), use_container_width=True)

        # Increment distribution (non-zero)
        pay_cols = [f'payment_period_{i}' for i in range(HORIZON_N)]
        nz = df_flat[pay_cols].to_numpy().flatten()
        nz = nz[nz != 0]
        st.plotly_chart(go.Figure(data=[go.Histogram(x=nz, nbinsx=60)]).update_layout(title='Increment Distribution (non-zero)'), use_container_width=True)

        # First payment timing
        paid = df_flat[df_flat['n_payments'] > 0]
        st.plotly_chart(go.Figure(data=[go.Histogram(x=paid['first_payment_period'], nbinsx=20)]).update_layout(title='First Payment Timing'), use_container_width=True)

        # Sample cumulative
        cum_cols = [f'cumulative_period_{i}' for i in range(HORIZON_N)]
        sample = df_flat.sample(min(10, len(df_flat)), random_state=42)
        fig = go.Figure()
        for _, r in sample.iterrows():
            fig.add_trace(go.Scatter(x=list(range(HORIZON_N)), y=[r[c] for c in cum_cols], mode='lines', name=f"Claim {int(r['claim_id'])}", opacity=0.6))
        fig.update_layout(title='Sample Claims (Cumulative)')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write({"mu": float(mu), "sigma": float(sigma)})

    with tab3:
        if st.button("🚀 Train on Dummy Data", type="primary"):
            with st.spinner("Training..."):
                history = train_model_lookback(model, train_loader, val_loader, device, epochs=epochs, lr=lr, alpha_cls=alpha_cls)
            st.success("Training complete")
            st.line_chart({'train_loss': history['train_loss'], 'val_loss': history['val_loss']})
            st.session_state.paper_dummy_model = model
            st.session_state.paper_dummy_mu = mu
            st.session_state.paper_dummy_sigma = sigma

        # Save / Load controls
        col1, col2 = st.columns(2)
        with col1:
            if 'paper_dummy_model' in st.session_state and st.button("💾 Save Model"):
                params = {
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'lr': lr,
                    'alpha_cls': alpha_cls,
                    'mu': float(mu),
                    'sigma': float(sigma),
                    'train_split': float(train_split),
                }
                path = save_paper_model(st.session_state.paper_dummy_model, params)
                st.success(f"Saved: {path}")
        with col2:
            models = list_paper_models()
            if models:
                choice = st.selectbox("Saved models", models)
                if st.button("📂 Load Selected"):
                    mdl, params = load_paper_model(choice, device)
                    st.session_state.paper_dummy_model = mdl
                    st.session_state.paper_dummy_mu = params.get('mu', mu)
                    st.session_state.paper_dummy_sigma = params.get('sigma', sigma)
                    st.success("Model loaded")

    with tab4:
        if 'paper_dummy_model' in st.session_state:
            m = st.session_state.paper_dummy_model
            m.eval()
            all_I, all_p, all_y = [], [], []
            with torch.no_grad():
                for X, y_star, I, mask in val_loader:
                    X = X.to(device)
                    y_hat_star, p_hat = m(X)
                    all_I.append(I.numpy())
                    all_p.append(p_hat.cpu().numpy())
                    all_y.append(y_hat_star.cpu().numpy())
            I_mat = np.concatenate(all_I)
            P_mat = np.concatenate(all_p)
            YS_mat = np.concatenate(all_y)
            exp_inc = P_mat * (YS_mat * st.session_state.paper_dummy_sigma + st.session_state.paper_dummy_mu)
            st.subheader('Expected increment preview (val)')
            st.write(np.round(exp_inc[:5, :10], 2))

    with tab5:
        if 'paper_dummy_model' in st.session_state:
            m = st.session_state.paper_dummy_model
            m.eval()
            # AUROC using last-timestep predictions for each window (lookback training)
            y_true_all, y_pred_all = [], []
            with torch.no_grad():
                for X, y_star, I, mask in val_loader:
                    X = X.to(device)
                    _, p_hat = m(X)
                    p_last = p_hat[:, -1].cpu().numpy()
                    y_true_all.append(I.numpy())
                    y_pred_all.append(p_last)
            y_true_all = np.concatenate(y_true_all)
            y_pred_all = np.concatenate(y_pred_all)
            if len(np.unique(y_true_all)) > 1:
                auc = float(roc_auc_score(y_true_all, y_pred_all))
            else:
                auc = None
            st.write({"AUROC_last_step": auc, "samples": int(y_true_all.shape[0])})

    with tab6:
        if 'paper_dummy_model' in st.session_state:
            m = st.session_state.paper_dummy_model
            m.eval()
            # Toggle: training vs validation pool
            use_validation = st.toggle("Use validation claims", value=True, help="Toggle to compare on training set as well")
            if use_validation:
                pool_claim_ids = df_flat.iloc[val_idx]['claim_id'].tolist()
                pool_inc_std = inc_val_std
                pool_occ = occ_val
                pool_obs = obs_val
            else:
                pool_claim_ids = df_flat.iloc[train_idx]['claim_id'].tolist()
                pool_inc_std = inc_train_std
                pool_occ = occ_train
                pool_obs = obs_train
            if not pool_claim_ids:
                st.info('No claims available in the selected split.')
                return
            selected_id = st.selectbox("Select claim_id", pool_claim_ids)
            sel_pos = pool_claim_ids.index(selected_id)
            # Use the same lookback as training
            T_full = HORIZON_N - 1
            inc_std_all = pool_inc_std[sel_pos, :T_full]
            occ_all = pool_occ[sel_pos, :T_full]
            obs_all = pool_obs[sel_pos, :T_full]

            mu_d = st.session_state.paper_dummy_mu
            sigma_d = st.session_state.paper_dummy_sigma

            expected_inc = np.zeros(T_full, dtype=float)
            # Predict each future period j using last lookback periods
            with torch.no_grad():
                for j in range(lookback, T_full + 0):
                    inc_feat = inc_std_all[j-lookback:j]
                    occ_feat = occ_all[j-lookback:j]
                    obs_feat = obs_all[j-lookback:j]
                    j_feat = j_over_n[j-lookback:j]
                    X_win = np.stack([inc_feat, occ_feat, j_feat, obs_feat], axis=-1).astype(np.float32)[None, ...]
                    y_star_win, p_hat_win = m(torch.from_numpy(X_win).to(device))
                    y_star_last = y_star_win[:, -1].cpu().numpy()[0]
                    p_last = p_hat_win[:, -1].cpu().numpy()[0]
                    expected_inc[j] = float(p_last * (y_star_last * sigma_d + mu_d))

            periods = np.arange(1, HORIZON_N)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=periods, y=expected_inc, name=f'Expected Increment (lookback={lookback})'))
            fig.update_layout(title=f"Claim {selected_id} - Expected Increments (Lookback={lookback})")
            st.plotly_chart(fig, use_container_width=True)

            # Plot observed vs predicted occurrence probabilities
            observed_I = pool_occ[sel_pos, 1:T_full+1]
            fig_prob = go.Figure()
            # Build p_series for periods 1..N-1
            p_series = np.zeros_like(expected_inc)
            with torch.no_grad():
                for j in range(lookback, T_full + 0):
                    inc_feat = inc_std_all[j-lookback:j]
                    occ_feat = occ_all[j-lookback:j]
                    obs_feat = obs_all[j-lookback:j]
                    j_feat = j_over_n[j-lookback:j]
                    X_win = np.stack([inc_feat, occ_feat, j_feat, obs_feat], axis=-1).astype(np.float32)[None, ...]
                    _, p_hat_win = m(torch.from_numpy(X_win).to(device))
                    p_series[j] = float(p_hat_win[:, -1].cpu().numpy()[0])
            fig_prob.add_trace(go.Scatter(x=periods, y=p_series, mode='lines+markers', name='Predicted p(no-zero)'))
            fig_prob.add_trace(go.Scatter(x=periods, y=observed_I, mode='lines+markers', name='Observed I', opacity=0.6))
            fig_prob.update_layout(title='Occurrence: Predicted probability vs Observed indicator', xaxis_title='Period', yaxis_title='Probability / Indicator')
            st.plotly_chart(fig_prob, use_container_width=True)

            # Actual vs Pred cumulative
            cum_cols = [f'cumulative_period_{i}' for i in range(HORIZON_N)]
            actual_cum = df_flat[df_flat['claim_id'] == selected_id][cum_cols].iloc[0].to_numpy(dtype=float)
            pred_cum = np.concatenate([[0.0], np.cumsum(expected_inc)])
            
            # Add anchor point to bridge the gap
            if lookback < HORIZON_N:
                anchor_period = lookback - 1
                anchor_value = actual_cum[anchor_period]
                pred_cum[lookback:] = pred_cum[lookback:] - pred_cum[lookback] + anchor_value
            
            fig_ap = go.Figure()
            fig_ap.add_trace(go.Scatter(x=np.arange(HORIZON_N), y=actual_cum, mode='lines+markers', name='Actual'))
            fig_ap.add_trace(go.Scatter(x=np.arange(HORIZON_N), y=pred_cum, mode='lines+markers', name='Predicted'))
            st.plotly_chart(fig_ap, use_container_width=True)


if __name__ == "__main__":
    main()


