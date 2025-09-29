"""
Insurance Claims LSTM App - PyTorch Version
Uses PyTorch LSTM to avoid TensorFlow mutex issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import glob
import warnings
import pickle
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
    print("PyTorch available")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available, using scikit-learn fallback")

# Fallback imports for when PyTorch is not available
if not PYTORCH_AVAILABLE:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

@st.cache_data
def load_claims_data():
    """Load the pre-generated claims data with per-period probabilities"""
    # Find the most recent claims data file
    csv_files = glob.glob("claims_data_period_flattened_*.csv")
    if not csv_files:
        st.error("No per-period claims data files found. Please run generate_claims_data_period.py first.")
        return None, None
    
    # Get the most recent file
    latest_file = max(csv_files, key=lambda x: x.split('_')[-1].split('.')[0])
    
    # Load the data
    df = pd.read_csv(latest_file)
    
    # Load metadata
    metadata_files = glob.glob("claims_metadata_period_*.json")
    if metadata_files:
        latest_metadata = max(metadata_files, key=lambda x: x.split('_')[-1].split('.')[0])
        with open(latest_metadata, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = None
    
    return df, metadata

def create_lstm_dataset(sequences, lookback=10):
    """Convert sequences to LSTM format - predicting payment increments"""
    X, y = [], []
    
    for seq in sequences:
        for i in range(lookback, len(seq)):
            X.append(seq[i-lookback:i])
            # Predict the payment increment (difference from previous period)
            if i > 0:
                y.append(seq[i] - seq[i-1])
            else:
                y.append(seq[i])  # First period has no previous period
    
    return np.array(X), np.array(y)

if PYTORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """PyTorch LSTM model for sequence prediction"""
        
        def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
            super(LSTMModel, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True
            )
            
            # Fully connected layers
            self.fc1 = nn.Linear(hidden_size, 25)
            self.fc2 = nn.Linear(25, 1)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            # Initialize hidden state
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            
            # LSTM forward pass
            out, _ = self.lstm(x, (h0, c0))
            
            # Take the last output
            out = out[:, -1, :]
            
            # Fully connected layers
            out = self.dropout(out)
            out = torch.relu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            
            return out
else:
    # Dummy class for when PyTorch is not available
    class LSTMModel:
        def __init__(self, *args, **kwargs):
            pass

def build_model(model_type, lookback=10, **kwargs):
    """Build the selected model"""
    if model_type == "LSTM (PyTorch)" and PYTORCH_AVAILABLE:
        return LSTMModel(
            input_size=1,
            hidden_size=kwargs.get('hidden_size', 50),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.2)
        )
    elif model_type == "Linear Regression":
        return LinearRegression()
    elif model_type == "Random Forest":
        return RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=42
        )
    else:
        # Fallback to Random Forest
        return RandomForestRegressor(n_estimators=100, random_state=42)

def train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001):
    """Train PyTorch LSTM model"""
    if not PYTORCH_AVAILABLE:
        return None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
        
        # Store history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss)
        
        # Early stopping
        if epoch > 10 and val_loss > min(history['val_loss'][-10:]):
            break
    
    return history

def train_sklearn_model(model, X_train, y_train):
    """Train scikit-learn model"""
    # Flatten for scikit-learn
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    model.fit(X_train_flat, y_train)
    return None

def make_pytorch_predictions(model, X_test):
    """Make predictions with PyTorch model"""
    if not PYTORCH_AVAILABLE:
        return None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        predictions = model(X_test_tensor)
        return predictions.cpu().numpy().flatten()

def make_sklearn_predictions(model, X_test):
    """Make predictions with scikit-learn model"""
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    return model.predict(X_test_flat)

def save_model(model, model_type, model_params, scaler, lookback, train_split):
    """Save the trained model and metadata"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    if model_type == "LSTM (PyTorch)" and PYTORCH_AVAILABLE:
        model_path = f'models/lstm_model_{timestamp}.pth'
        torch.save(model.state_dict(), model_path)
    else:
        model_path = f'models/sklearn_model_{timestamp}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        'model_type': model_type,
        'model_params': model_params,
        'lookback': lookback,
        'train_split': train_split,
        'timestamp': timestamp,
        'model_path': model_path,
        'scaler_min': scaler.data_min_[0],
        'scaler_max': scaler.data_max_[0]
    }
    
    metadata_path = f'models/model_metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_path, metadata_path

def load_model(model_path, metadata_path):
    """Load a saved model and metadata"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model_type = metadata['model_type']
    
    if model_type == "LSTM (PyTorch)" and PYTORCH_AVAILABLE:
        # Load PyTorch model
        model_params = metadata['model_params']
        model = LSTMModel(
            input_size=1,
            hidden_size=model_params.get('hidden_size', 50),
            num_layers=model_params.get('num_layers', 2),
            dropout=model_params.get('dropout', 0.2)
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        # Load scikit-learn model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    # Recreate scaler
    scaler = MinMaxScaler()
    scaler.data_min_ = np.array([metadata['scaler_min']])
    scaler.data_max_ = np.array([metadata['scaler_max']])
    scaler.scale_ = np.array([1.0 / (metadata['scaler_max'] - metadata['scaler_min'])])
    scaler.min_ = np.array([-metadata['scaler_min'] / (metadata['scaler_max'] - metadata['scaler_min'])])
    
    return model, metadata, scaler

def get_saved_models():
    """Get list of saved models"""
    if not os.path.exists('models'):
        return []
    
    metadata_files = glob.glob('models/model_metadata_*.json')
    models = []
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            models.append({
                'metadata_path': metadata_file,
                'model_path': metadata['model_path'],
                'model_type': metadata['model_type'],
                'timestamp': metadata['timestamp'],
                'lookback': metadata['lookback'],
                'train_split': metadata['train_split']
            })
        except:
            continue
    
    # Sort by timestamp (newest first)
    models.sort(key=lambda x: x['timestamp'], reverse=True)
    return models

def calculate_test_set_accuracy(model, model_type, scaler, test_claim_ids, claims_df, cumulative_cols, lookback):
    """Calculate aggregate accuracy metrics for all test claims"""
    if not test_claim_ids:
        return None
    
    test_claims = claims_df[claims_df['claim_id'].isin(test_claim_ids)]
    final_errors = []
    final_error_pcts = []
    
    for _, claim in test_claims.iterrows():
        # Get claim cumulative payments
        claim_cumulative = [claim[col] for col in cumulative_cols]
        
        # Scale the data
        claim_sequence_scaled = scaler.transform(
            np.array(claim_cumulative).reshape(-1, 1)
        ).flatten()
        
        # Make prediction for the last period
        if len(claim_sequence_scaled) > lookback:
            input_seq = claim_sequence_scaled[-lookback:].reshape(1, lookback, 1)
            
            # Make prediction (this predicts the payment increment)
            if model_type == "LSTM (PyTorch)" and PYTORCH_AVAILABLE:
                pred_increment = make_pytorch_predictions(model, input_seq)
            else:
                pred_increment = make_sklearn_predictions(model, input_seq)
            
            # Convert back to original scale
            pred_increment_orig = scaler.inverse_transform(
                pred_increment.reshape(-1, 1)
            ).flatten()[0]
            
            # Add the predicted increment to the previous cumulative payment
            previous_cumulative = claim_cumulative[-2] if len(claim_cumulative) > 1 else 0
            predicted_cumulative = previous_cumulative + pred_increment_orig
            
            # Calculate final error
            final_actual = claim_cumulative[-1]
            final_error = abs(final_actual - predicted_cumulative)
            final_error_pct = (final_error / final_actual * 100) if final_actual > 0 else 0
            
            final_errors.append(final_error)
            final_error_pcts.append(final_error_pct)
    
    if final_errors:
        return {
            'avg_final_error': np.mean(final_errors),
            'avg_final_error_pct': np.mean(final_error_pcts),
            'test_claims_count': len(final_errors),
            'max_final_error': np.max(final_errors),
            'min_final_error': np.min(final_errors)
        }
    
    return None

def main():
    st.set_page_config(
        page_title="Insurance Claims LSTM - PyTorch",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.5rem;
            color: #2e8b57;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .info-box {
            background-color: #e8f4fd;
            border-left: 4px solid #1f77b4;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Insurance Claims LSTM - PyTorch</h1>', unsafe_allow_html=True)
    st.markdown("Per-period payment probabilities with PyTorch LSTM")
    
    # Load data
    with st.spinner("Loading claims data..."):
        claims_df, metadata = load_claims_data()
    
    if claims_df is None:
        st.stop()
    
    # Sidebar parameters
    st.sidebar.header("🎛️ Model Parameters")
    
    lookback = st.sidebar.slider("LSTM Lookback Window", 5, 30, 15, 5)
    train_split = st.sidebar.slider("Training Split", 0.6, 0.9, 0.8, 0.05)
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    if PYTORCH_AVAILABLE:
        model_type = st.sidebar.selectbox(
            "Model Type",
            ["LSTM (PyTorch)", "Linear Regression", "Random Forest"],
            help="LSTM recommended for sequential data"
        )
    else:
        st.sidebar.warning("PyTorch not available. Using scikit-learn models.")
        model_type = st.sidebar.selectbox(
            "Model Type",
            ["Linear Regression", "Random Forest"]
        )
    
    # Model-specific parameters
    if model_type == "LSTM (PyTorch)" and PYTORCH_AVAILABLE:
        st.sidebar.subheader("🧠 LSTM Parameters")
        hidden_size = st.sidebar.slider("Hidden Size", 25, 200, 50, 25)
        num_layers = st.sidebar.slider("Number of Layers", 1, 4, 2, 1)
        dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.1)
        epochs = st.sidebar.slider("Epochs", 10, 200, 100, 10)
        batch_size = st.sidebar.slider("Batch Size", 16, 128, 32, 16)
        learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
        
        model_params = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    elif model_type == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 100, 10)
        max_depth = st.sidebar.slider("Max Depth", 3, 20, 10, 1)
        model_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
    else:
        model_params = {}
    
    # Display data info
    st.markdown('<h2 class="section-header">📊 Claims Data Summary</h2>', unsafe_allow_html=True)
    
    # Extract cumulative payments from columns
    cumulative_cols = [col for col in claims_df.columns if col.startswith('cumulative_period_')]
    n_periods = len(cumulative_cols)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Claims", f"{len(claims_df):,}")
    with col2:
        st.metric("Periods", f"{n_periods}")
    with col3:
        st.metric("Claims with Payments", f"{(claims_df['total_payments'] > 0).sum():,}")
    with col4:
        st.metric("Avg Total Payment", f"${claims_df['total_payments'].mean():,.0f}")
    
    # Display metadata if available
    if metadata:
        st.sidebar.subheader("📋 Data Info")
        st.sidebar.write(f"Generated: {metadata.get('generated_at', 'Unknown')}")
        st.sidebar.write(f"Seed: {metadata.get('seed', 'Unknown')}")
        st.sidebar.write(f"Per-Period Probs: {metadata.get('period_frequency_probs', 'Unknown')}")
    
    # Tab layout
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Data Overview", 
        "🔧 Data Preparation", 
        "🤖 Model Training", 
        "🎯 Predictions", 
        "📊 Analysis",
        "🔍 Individual Claims"
    ])
    
    with tab1:
        st.markdown('<h3 class="section-header">📈 Claims Data Overview</h3>', unsafe_allow_html=True)
        
        # Payment frequency distribution
        fig1 = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Payment Frequency Distribution", 
                "Total Payment Distribution",
                "Sample Claims (Cumulative Payments)",
                "Payment Timing Distribution"
            )
        )
        
        # Payment frequency
        payment_counts = claims_df['n_payments'].value_counts().sort_index()
        fig1.add_trace(go.Bar(
            x=payment_counts.index,
            y=payment_counts.values,
            name="Payment Count",
            marker_color='lightblue'
        ), row=1, col=1)
        
        # Total payment distribution (log scale)
        fig1.add_trace(go.Histogram(
            x=claims_df[claims_df['total_payments'] > 0]['total_payments'],
            nbinsx=50,
            name="Total Payments",
            marker_color='lightgreen'
        ), row=1, col=2)
        
        # Sample claims
        sample_claims = claims_df.sample(min(10, len(claims_df)))
        for i, (_, claim) in enumerate(sample_claims.iterrows()):
            cumulative_payments = [claim[col] for col in cumulative_cols]
            periods = list(range(len(cumulative_payments)))
            
            fig1.add_trace(go.Scatter(
                x=periods,
                y=cumulative_payments,
                mode='lines',
                name=f"Claim {claim['claim_id']}",
                showlegend=False,
                opacity=0.7
            ), row=2, col=1)
        
        # Payment timing
        first_payments = claims_df[claims_df['first_payment_period'] < n_periods]['first_payment_period']
        fig1.add_trace(go.Histogram(
            x=first_payments,
            nbinsx=20,
            name="First Payment Period",
            marker_color='orange'
        ), row=2, col=2)
        
        fig1.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Statistics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Payment Statistics:**")
            st.write(f"• Claims with no payments: {(claims_df['n_payments'] == 0).sum():,} ({(claims_df['n_payments'] == 0).mean()*100:.1f}%)")
            st.write(f"• Claims with 1-10 payments: {((claims_df['n_payments'] >= 1) & (claims_df['n_payments'] <= 10)).sum():,} ({((claims_df['n_payments'] >= 1) & (claims_df['n_payments'] <= 10)).mean()*100:.1f}%)")
            st.write(f"• Claims with 11-30 payments: {((claims_df['n_payments'] >= 11) & (claims_df['n_payments'] <= 30)).sum():,} ({((claims_df['n_payments'] >= 11) & (claims_df['n_payments'] <= 30)).mean()*100:.1f}%)")
            st.write(f"• Claims with 31+ payments: {(claims_df['n_payments'] >= 31).sum():,} ({(claims_df['n_payments'] >= 31).mean()*100:.1f}%)")
            st.write(f"• Average payments per claim: {claims_df['n_payments'].mean():.2f}")
        
        with col2:
            st.markdown("**Payment Amount Statistics:**")
            paid_claims = claims_df[claims_df['total_payments'] > 0]
            st.write(f"• Mean payment: ${paid_claims['total_payments'].mean():,.0f}")
            st.write(f"• Median payment: ${paid_claims['total_payments'].median():,.0f}")
            st.write(f"• 95th percentile: ${paid_claims['total_payments'].quantile(0.95):,.0f}")
            st.write(f"• Max payment: ${paid_claims['total_payments'].max():,.0f}")
        
        # Per-period statistics
        st.markdown("**Per-Period Statistics:**")
        all_payments = []
        for _, row in claims_df.iterrows():
            for col in cumulative_cols:
                if col.startswith('payment_period_'):
                    all_payments.append(row[col])
        
        all_payments = np.array(all_payments)
        periods_with_payments = np.sum(all_payments > 0)
        total_periods = len(all_payments)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"• Total periods across all claims: {total_periods:,}")
            st.write(f"• Periods with payments: {periods_with_payments:,} ({periods_with_payments/total_periods*100:.1f}%)")
            st.write(f"• Periods without payments: {total_periods - periods_with_payments:,} ({(total_periods - periods_with_payments)/total_periods*100:.1f}%)")
        
        with col2:
            st.write(f"• Average payment per period (when > 0): ${all_payments[all_payments > 0].mean():,.0f}")
            st.write(f"• Expected per-period payment rate: ~30% (P(N_t=0)=70%)")
            st.write(f"• Actual per-period payment rate: {periods_with_payments/total_periods*100:.1f}%")
    
    with tab2:
        st.markdown('<h3 class="section-header">🔧 Data Preparation</h3>', unsafe_allow_html=True)
        
        # Prepare sequences from cumulative payments
        sequences = []
        for _, row in claims_df.iterrows():
            cumulative_payments = [row[col] for col in cumulative_cols]
            sequences.append(cumulative_payments)
        
        # Scale the data
        scaler = MinMaxScaler()
        sequences_scaled = []
        for seq in sequences:
            scaled_seq = scaler.fit_transform(np.array(seq).reshape(-1, 1)).flatten()
            sequences_scaled.append(scaled_seq)
        
        # Create supervised learning dataset
        X, y = create_lstm_dataset(sequences_scaled, lookback)
        X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM
        
        # Split data
        split_idx = int(train_split * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Further split training into train/validation
        val_split = int(0.8 * len(X_train))
        X_train_final, X_val = X_train[:val_split], X_train[val_split:]
        y_train_final, y_val = y_train[:val_split], y_train[val_split:]
        
        # Store in session state
        st.session_state.X_train = X_train_final
        st.session_state.X_val = X_val
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train_final
        st.session_state.y_val = y_val
        st.session_state.y_test = y_test
        st.session_state.scaler = scaler
        st.session_state.train_split = train_split
        
        # Store test claim IDs for individual analysis
        # Calculate which claims are in test set based on the split
        total_claims = len(claims_df)
        train_claim_count = int(total_claims * train_split)
        test_claim_ids = set(claims_df['claim_id'].iloc[train_claim_count:].tolist())
        st.session_state.test_claim_ids = test_claim_ids
        
        # Display dataset info
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dataset Information:**")
            st.write(f"• Total sequences: {len(X):,}")
            st.write(f"• Training samples: {len(X_train_final):,}")
            st.write(f"• Validation samples: {len(X_val):,}")
            st.write(f"• Test samples: {len(X_test):,}")
            st.write(f"• Lookback window: {lookback}")
            st.write(f"• Features per timestep: 1")
        
        with col2:
            st.markdown("**Data Scaling:**")
            st.write(f"• Scaled to range: [0, 1]")
            st.write(f"• Original min: ${scaler.data_min_[0]:,.0f}")
            st.write(f"• Original max: ${scaler.data_max_[0]:,.0f}")
            st.write(f"• Scaled min: 0.0")
            st.write(f"• Scaled max: 1.0")
        
        # Show sample sequences
        st.markdown("**Sample Input Sequences:**")
        fig2 = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Sequence 1", "Sequence 2", "Sequence 3", "Sequence 4")
        )
        
        for i in range(4):
            row = i // 2 + 1
            col = i % 2 + 1
            
            sample_seq = X_train_final[i*100].flatten()
            
            fig2.add_trace(go.Scatter(
                x=list(range(len(sample_seq))),
                y=sample_seq,
                mode='lines+markers',
                name=f'Seq {i+1}',
                showlegend=False
            ), row=row, col=col)
        
        fig2.update_layout(
            title="Sample Input Sequences (Scaled)",
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.markdown('<h3 class="section-header">🤖 Model Training</h3>', unsafe_allow_html=True)
        
        # Model management section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🚀 Train New Model")
            
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model..."):
                    # Build model
                    model = build_model(model_type, lookback, **model_params)
                    
                    # Train model
                    if model_type == "LSTM (PyTorch)" and PYTORCH_AVAILABLE:
                        history = train_pytorch_model(
                            model, 
                            st.session_state.X_train, 
                            st.session_state.y_train,
                            st.session_state.X_val, 
                            st.session_state.y_val,
                            epochs=model_params.get('epochs', 100),
                            batch_size=model_params.get('batch_size', 32),
                            learning_rate=model_params.get('learning_rate', 0.001)
                        )
                    else:
                        history = train_sklearn_model(
                            model, 
                            st.session_state.X_train, 
                            st.session_state.y_train
                        )
                    
                    # Store model
                    st.session_state.model = model
                    st.session_state.history = history
                    st.session_state.model_trained = True
                    st.session_state.model_type = model_type
                    st.session_state.model_params = model_params
                    
                    # Calculate test set accuracy
                    test_accuracy = calculate_test_set_accuracy(
                        model, model_type, st.session_state.scaler, 
                        st.session_state.test_claim_ids, claims_df, 
                        cumulative_cols, lookback
                    )
                    if test_accuracy:
                        st.session_state.test_predictions_accuracy = test_accuracy
                    
                    st.success("Model trained successfully!")
        
        with col2:
            st.subheader("💾 Save Model")
            
            if 'model_trained' in st.session_state:
                if st.button("💾 Save Current Model"):
                    try:
                        model_path, metadata_path = save_model(
                            st.session_state.model,
                            st.session_state.model_type,
                            st.session_state.model_params,
                            st.session_state.scaler,
                            lookback,
                            train_split
                        )
                        st.success(f"Model saved successfully!")
                        st.write(f"Saved to: {model_path}")
                    except Exception as e:
                        st.error(f"Error saving model: {str(e)}")
            else:
                st.info("Train a model first to save it")
        
        # Load saved models section
        st.subheader("📂 Load Saved Models")
        
        saved_models = get_saved_models()
        
        if saved_models:
            # Create model selection options
            model_options = []
            for model_info in saved_models:
                timestamp_str = datetime.strptime(model_info['timestamp'], '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
                option_text = f"{model_info['model_type']} - {timestamp_str} (Lookback: {model_info['lookback']})"
                model_options.append((option_text, model_info))
            
            if model_options:
                selected_model_text = st.selectbox(
                    "Select a saved model to load:",
                    [opt[0] for opt in model_options],
                    help="Choose from previously saved models"
                )
                
                if st.button("📂 Load Selected Model"):
                    # Find the selected model info
                    selected_model_info = None
                    for option_text, model_info in model_options:
                        if option_text == selected_model_text:
                            selected_model_info = model_info
                            break
                    
                    if selected_model_info:
                        try:
                            with st.spinner("Loading model..."):
                                model, metadata, scaler = load_model(
                                    selected_model_info['model_path'],
                                    selected_model_info['metadata_path']
                                )
                                
                                # Store loaded model in session state
                                st.session_state.model = model
                                st.session_state.history = None  # No training history for loaded models
                                st.session_state.model_trained = True
                                st.session_state.model_type = metadata['model_type']
                                st.session_state.model_params = metadata['model_params']
                                st.session_state.scaler = scaler
                                st.session_state.train_split = metadata['train_split']
                                
                                # Recalculate test claim IDs and test accuracy for loaded model
                                total_claims = len(claims_df)
                                train_claim_count = int(total_claims * metadata['train_split'])
                                test_claim_ids = set(claims_df['claim_id'].iloc[train_claim_count:].tolist())
                                st.session_state.test_claim_ids = test_claim_ids
                                
                                # Calculate test set accuracy
                                test_accuracy = calculate_test_set_accuracy(
                                    model, metadata['model_type'], scaler, 
                                    test_claim_ids, claims_df, 
                                    cumulative_cols, metadata['lookback']
                                )
                                if test_accuracy:
                                    st.session_state.test_predictions_accuracy = test_accuracy
                                
                                st.success("Model loaded successfully!")
                                st.write(f"Model Type: {metadata['model_type']}")
                                st.write(f"Lookback: {metadata['lookback']}")
                                st.write(f"Train Split: {metadata['train_split']}")
                                st.write(f"Saved: {datetime.strptime(metadata['timestamp'], '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')}")
                        except Exception as e:
                            st.error(f"Error loading model: {str(e)}")
        else:
            st.info("No saved models found. Train and save a model first.")
        
        if 'model_trained' in st.session_state:
            # Display training history if available
            if st.session_state.history is not None and model_type == "LSTM (PyTorch)":
                fig3 = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Training Loss', 'Validation Loss')
                )
                
                fig3.add_trace(go.Scatter(
                    y=st.session_state.history['train_loss'],
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='blue')
                ), row=1, col=1)
                
                fig3.add_trace(go.Scatter(
                    y=st.session_state.history['val_loss'],
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='red')
                ), row=1, col=1)
                
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)
            
            # Model summary
            st.markdown("**Model Information:**")
            st.write(f"• Model Type: {st.session_state.model_type}")
            st.write(f"• Parameters: {model_params}")
            
            if model_type == "LSTM (PyTorch)" and PYTORCH_AVAILABLE:
                st.write(f"• Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
                st.write(f"• Hidden Size: {model_params.get('hidden_size', 50)}")
                st.write(f"• Number of Layers: {model_params.get('num_layers', 2)}")
                st.write(f"• Dropout Rate: {model_params.get('dropout', 0.2)}")
            elif model_type == "LSTM (PyTorch)" and not PYTORCH_AVAILABLE:
                st.warning("PyTorch is not available. Please install PyTorch to use LSTM models.")
    
    with tab4:
        st.markdown('<h3 class="section-header">🎯 Predictions</h3>', unsafe_allow_html=True)
        
        if 'model_trained' not in st.session_state:
            st.warning("Please train a model first in the 'Model Training' tab.")
        else:
            # Make predictions
            if st.session_state.model_type == "LSTM (PyTorch)" and PYTORCH_AVAILABLE:
                y_pred = make_pytorch_predictions(st.session_state.model, st.session_state.X_test)
            else:
                y_pred = make_sklearn_predictions(st.session_state.model, st.session_state.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(st.session_state.y_test, y_pred)
            mae = mean_absolute_error(st.session_state.y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MSE", f"{mse:.6f}")
            with col2:
                st.metric("MAE", f"{mae:.6f}")
            with col3:
                st.metric("RMSE", f"{rmse:.6f}")
            
            # Convert back to original scale
            y_test_orig = st.session_state.scaler.inverse_transform(
                st.session_state.y_test.reshape(-1, 1)
            ).flatten()
            y_pred_orig = st.session_state.scaler.inverse_transform(
                y_pred.reshape(-1, 1)
            ).flatten()
            
            # Predictions plot
            fig4 = go.Figure()
            
            # Sample of predictions
            n_samples = min(100, len(y_test_orig))
            sample_idx = np.random.choice(len(y_test_orig), n_samples, replace=False)
            
            fig4.add_trace(go.Scatter(
                x=y_test_orig[sample_idx],
                y=y_pred_orig[sample_idx],
                mode='markers',
                name='Predictions vs Actual',
                marker=dict(
                    color='blue',
                    size=8,
                    opacity=0.6
                ),
                hovertemplate='Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>'
            ))
            
            # Perfect prediction line
            min_val = min(y_test_orig.min(), y_pred_orig.min())
            max_val = max(y_test_orig.max(), y_pred_orig.max())
            fig4.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig4.update_layout(
                title="Predictions vs Actual Values",
                xaxis_title="Actual Cumulative Payments ($)",
                yaxis_title="Predicted Cumulative Payments ($)",
                height=500
            )
            
            st.plotly_chart(fig4, use_container_width=True)
            
            # Error analysis
            errors = np.abs(y_test_orig - y_pred_orig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Error Statistics:**")
                st.write(f"• Mean Absolute Error: ${np.mean(errors):,.0f}")
                st.write(f"• Median Error: ${np.median(errors):,.0f}")
                st.write(f"• 95th Percentile Error: ${np.percentile(errors, 95):,.0f}")
                st.write(f"• Max Error: ${np.max(errors):,.0f}")
            
            with col2:
                st.markdown("**Error Distribution:**")
                st.write(f"• Error Std Dev: ${np.std(errors):,.0f}")
                st.write(f"• Errors < $1,000: {(errors < 1000).mean()*100:.1f}%")
                st.write(f"• Errors < $5,000: {(errors < 5000).mean()*100:.1f}%")
                st.write(f"• Errors < $10,000: {(errors < 10000).mean()*100:.1f}%")
            
            # Error histogram
            fig5 = go.Figure()
            fig5.add_trace(go.Histogram(
                x=errors,
                nbinsx=50,
                name='Error Distribution',
                marker_color='lightcoral'
            ))
            
            fig5.update_layout(
                title="Prediction Error Distribution",
                xaxis_title="Absolute Error ($)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig5, use_container_width=True)
    
    with tab5:
        st.markdown('<h3 class="section-header">📊 Analysis</h3>', unsafe_allow_html=True)
        
        # Claims development patterns
        st.markdown("**Claims Development Patterns:**")
        
        # Group claims by development pattern
        claims_df['development_pattern'] = 'No Payments'
        claims_df.loc[claims_df['n_payments'] <= 10, 'development_pattern'] = 'Low Activity (1-10 payments)'
        claims_df.loc[(claims_df['n_payments'] > 10) & (claims_df['n_payments'] <= 30), 'development_pattern'] = 'Medium Activity (11-30 payments)'
        claims_df.loc[claims_df['n_payments'] > 30, 'development_pattern'] = 'High Activity (31+ payments)'
        
        pattern_counts = claims_df['development_pattern'].value_counts()
        
        fig6 = go.Figure()
        fig6.add_trace(go.Pie(
            labels=pattern_counts.index,
            values=pattern_counts.values,
            hole=0.3
        ))
        
        fig6.update_layout(
            title="Claims Development Patterns",
            height=400
        )
        
        st.plotly_chart(fig6, use_container_width=True)
        
        # Payment timing analysis
        st.markdown("**Payment Timing Analysis:**")
        
        paid_claims = claims_df[claims_df['total_payments'] > 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**First Payment Timing:**")
            first_payment_stats = paid_claims['first_payment_period'].describe()
            for stat, value in first_payment_stats.items():
                st.write(f"• {stat.capitalize()}: {value:.1f} periods")
        
        with col2:
            st.markdown("**Payment Duration:**")
            paid_claims['payment_duration'] = paid_claims['last_payment_period'] - paid_claims['first_payment_period'] + 1
            duration_stats = paid_claims['payment_duration'].describe()
            for stat, value in duration_stats.items():
                st.write(f"• {stat.capitalize()}: {value:.1f} periods")
        
        # Summary insights
        st.markdown("**Key Insights:**")
        
        insights = [
            f"• Per-period payment probability: P(N_t=0)=70%, P(N_t=1)=10%, etc.",
            f"• Average of {claims_df['n_payments'].mean():.2f} payments per claim",
            f"• Payment amounts follow a log-normal distribution (highly skewed)",
            f"• {(paid_claims['first_payment_period'] < 10).mean()*100:.1f}% of paid claims start within 10 periods",
            f"• Average total payment: ${paid_claims['total_payments'].mean():,.0f}",
            f"• 95th percentile payment: ${paid_claims['total_payments'].quantile(0.95):,.0f}"
        ]
        
        for insight in insights:
            st.write(insight)
        
        # Model performance summary
        if 'model_trained' in st.session_state:
            st.markdown("**Model Performance Summary:**")
            st.write(f"• Model Type: {st.session_state.model_type}")
            st.write(f"• Lookback Window: {lookback} periods")
            st.write(f"• Training Samples: {len(st.session_state.X_train):,}")
            st.write(f"• Test Samples: {len(st.session_state.X_test):,}")
            if 'y_pred' in locals():
                st.write(f"• Test MAE: ${mae:.0f}")
                st.write(f"• Test RMSE: ${rmse:.0f}")
    
    with tab6:
        st.markdown('<h3 class="section-header">🔍 Individual Claims Analysis</h3>', unsafe_allow_html=True)
        
        # Training/Test toggle
        col1, col2 = st.columns([1, 3])
        
        with col1:
            use_test_claims = st.toggle(
                "Use Test Claims",
                value=False,
                help="Toggle to analyze test claims (not used in training) vs training claims"
            )
        
        with col2:
            if use_test_claims:
                st.info("🔍 Analyzing test claims - these were not used during model training")
            else:
                st.info("📚 Analyzing training claims - these were used during model training")
        
        # Claim selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Filter claims based on training/test split
            if 'model_trained' in st.session_state and use_test_claims:
                # Get test claim IDs from the test data
                test_claim_ids = set()
                if hasattr(st.session_state, 'test_claim_ids'):
                    test_claim_ids = st.session_state.test_claim_ids
                else:
                    # If test_claim_ids not stored, we need to identify test claims
                    # This is a fallback - ideally we should store this during training
                    total_claims = len(claims_df)
                    train_size = int(total_claims * st.session_state.train_split)
                    test_claim_ids = set(claims_df['claim_id'].iloc[train_size:].tolist())
                
                available_claim_ids = [cid for cid in claims_df['claim_id'].tolist() if cid in test_claim_ids]
                if not available_claim_ids:
                    st.warning("No test claims available. Please train a model first.")
                    available_claim_ids = claims_df['claim_id'].tolist()[:10]  # Fallback
            else:
                # Training claims or no model trained yet
                if 'model_trained' in st.session_state:
                    total_claims = len(claims_df)
                    train_size = int(total_claims * st.session_state.train_split)
                    available_claim_ids = claims_df['claim_id'].iloc[:train_size].tolist()
                else:
                    available_claim_ids = claims_df['claim_id'].tolist()
            
            # Claim ID selector
            selected_claim_id = st.selectbox(
                f"Select Claim ID ({'Test' if use_test_claims else 'Training'}):",
                available_claim_ids,
                help=f"Choose a {'test' if use_test_claims else 'training'} claim to analyze its lifetime and predictions"
            )
        
        with col2:
            # Show claim info
            selected_claim = claims_df[claims_df['claim_id'] == selected_claim_id].iloc[0]
            st.metric("Total Payments", f"${selected_claim['total_payments']:,.0f}")
            st.metric("Number of Payments", f"{selected_claim['n_payments']}")
        
        # Get claim data
        claim_cumulative = [selected_claim[col] for col in cumulative_cols]
        claim_payments = [selected_claim[col.replace('cumulative_period_', 'payment_period_')] for col in cumulative_cols]
        periods = list(range(len(claim_cumulative)))
        
        # Create lifetime visualization
        fig_lifetime = go.Figure()
        
        # Cumulative payments
        fig_lifetime.add_trace(go.Scatter(
            x=periods,
            y=claim_cumulative,
            mode='lines+markers',
            name='Cumulative Payments',
            line=dict(color='blue', width=3),
            marker=dict(size=6),
            hovertemplate='Period: %{x}<br>Cumulative: $%{y:,.0f}<extra></extra>'
        ))
        
        # Individual payments (as bars)
        fig_lifetime.add_trace(go.Bar(
            x=periods,
            y=claim_payments,
            name='Period Payments',
            marker_color='lightblue',
            opacity=0.7,
            hovertemplate='Period: %{x}<br>Payment: $%{y:,.0f}<extra></extra>'
        ))
        
        fig_lifetime.update_layout(
            title=f"Claim {selected_claim_id} Lifetime Development",
            xaxis_title="Development Period",
            yaxis_title="Payment Amount ($)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_lifetime, use_container_width=True)
        
        # Model predictions for this claim
        if 'model_trained' in st.session_state:
            st.markdown("**Model Predictions for This Claim:**")
            
            # Prepare the claim's sequence for prediction
            claim_sequence_scaled = st.session_state.scaler.transform(
                np.array(claim_cumulative).reshape(-1, 1)
            ).flatten()
            
            # Create prediction sequence
            predictions = []
            prediction_periods = []
            
            # Make predictions for each period after the lookback window
            for i in range(lookback, len(claim_sequence_scaled)):
                # Get the input sequence
                input_seq = claim_sequence_scaled[i-lookback:i].reshape(1, lookback, 1)
                
                # Make prediction (this predicts the payment increment)
                if st.session_state.model_type == "LSTM (PyTorch)" and PYTORCH_AVAILABLE:
                    pred_increment = make_pytorch_predictions(st.session_state.model, input_seq)
                else:
                    pred_increment = make_sklearn_predictions(st.session_state.model, input_seq)
                
                # Convert back to original scale
                pred_increment_orig = st.session_state.scaler.inverse_transform(
                    pred_increment.reshape(-1, 1)
                ).flatten()[0]
                
                # Add the predicted increment to the previous cumulative payment
                previous_cumulative = claim_cumulative[i-1] if i > 0 else 0
                predicted_cumulative = previous_cumulative + pred_increment_orig
                
                predictions.append(predicted_cumulative)
                prediction_periods.append(i)
            
            # Create prediction visualization
            fig_predictions = go.Figure()
            
            # Actual cumulative payments
            fig_predictions.add_trace(go.Scatter(
                x=periods,
                y=claim_cumulative,
                mode='lines+markers',
                name='Actual Cumulative',
                line=dict(color='blue', width=3),
                marker=dict(size=6)
            ))
            
            # Predictions
            fig_predictions.add_trace(go.Scatter(
                x=prediction_periods,
                y=predictions,
                mode='lines+markers',
                name='Model Predictions',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            fig_predictions.update_layout(
                title=f"Claim {selected_claim_id} - Actual vs Predicted",
                xaxis_title="Development Period",
                yaxis_title="Cumulative Payment Amount ($)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_predictions, use_container_width=True)
            
            # Prediction accuracy for this claim
            if len(predictions) > 0:
                actual_values = [claim_cumulative[i] for i in prediction_periods]
                errors = [abs(actual - pred) for actual, pred in zip(actual_values, predictions)]
                
                # Calculate final cumulative payment accuracy (last period)
                final_actual = claim_cumulative[-1]  # Last period actual
                final_predicted = predictions[-1] if predictions else 0  # Last prediction
                final_error = abs(final_actual - final_predicted)
                final_error_pct = (final_error / final_actual * 100) if final_actual > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Absolute Error", f"${np.mean(errors):,.0f}")
                with col2:
                    st.metric("Max Error", f"${np.max(errors):,.0f}")
                with col3:
                    st.metric("Final Payment Error", f"${final_error:,.0f}")
                with col4:
                    st.metric("Final Error %", f"{final_error_pct:.1f}%")
                
                # Test claim accuracy analysis
                if use_test_claims:
                    st.markdown("**🎯 Test Claim Accuracy Analysis:**")
                    
                    # Calculate accuracy metrics for this test claim
                    accuracy_metrics = {
                        'Final Cumulative Payment': {
                            'Actual': f"${final_actual:,.0f}",
                            'Predicted': f"${final_predicted:,.0f}",
                            'Error': f"${final_error:,.0f}",
                            'Error %': f"{final_error_pct:.1f}%"
                        },
                        'Overall Prediction': {
                            'Mean Error': f"${np.mean(errors):,.0f}",
                            'Max Error': f"${np.max(errors):,.0f}",
                            'Predictions Made': f"{len(predictions)}"
                        }
                    }
                    
                    # Display accuracy metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Final Cumulative Payment Accuracy:**")
                        for metric, value in accuracy_metrics['Final Cumulative Payment'].items():
                            st.write(f"• {metric}: {value}")
                    
                    with col2:
                        st.markdown("**Overall Prediction Accuracy:**")
                        for metric, value in accuracy_metrics['Overall Prediction'].items():
                            st.write(f"• {metric}: {value}")
                    
                    # Aggregate test set accuracy (if we have access to all test predictions)
                    if hasattr(st.session_state, 'test_predictions_accuracy'):
                        st.markdown("**📊 Aggregate Test Set Accuracy:**")
                        agg_metrics = st.session_state.test_predictions_accuracy
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Avg Final Error", f"${agg_metrics['avg_final_error']:,.0f}")
                        with col2:
                            st.metric("Avg Final Error %", f"{agg_metrics['avg_final_error_pct']:.1f}%")
                        with col3:
                            st.metric("Test Claims", f"{agg_metrics['test_claims_count']}")
                        
                        # Plot actual vs predicted for all test claims
                        st.markdown("**📈 Test Claims: Actual vs Predicted Final Cumulative Payments:**")
                        
                        # Calculate predictions for all test claims
                        test_claims = claims_df[claims_df['claim_id'].isin(st.session_state.test_claim_ids)]
                        actuals = []
                        predictions = []
                        claim_ids = []
                        errors = []
                        
                        for _, claim in test_claims.iterrows():
                            # Get claim cumulative payments
                            claim_cumulative = [claim[col] for col in cumulative_cols]
                            
                            # Scale the data
                            claim_sequence_scaled = st.session_state.scaler.transform(
                                np.array(claim_cumulative).reshape(-1, 1)
                            ).flatten()
                            
                            # Make prediction for the last period
                            if len(claim_sequence_scaled) > lookback:
                                input_seq = claim_sequence_scaled[-lookback:].reshape(1, lookback, 1)
                                
                                # Make prediction (this predicts the payment increment)
                                if st.session_state.model_type == "LSTM (PyTorch)" and PYTORCH_AVAILABLE:
                                    pred_increment = make_pytorch_predictions(st.session_state.model, input_seq)
                                else:
                                    pred_increment = make_sklearn_predictions(st.session_state.model, input_seq)
                                
                                # Convert back to original scale
                                pred_increment_orig = st.session_state.scaler.inverse_transform(
                                    pred_increment.reshape(-1, 1)
                                ).flatten()[0]
                                
                                # Add the predicted increment to the previous cumulative payment
                                previous_cumulative = claim_cumulative[-2] if len(claim_cumulative) > 1 else 0
                                predicted_cumulative = previous_cumulative + pred_increment_orig
                                
                                # Store results
                                final_actual = claim_cumulative[-1]
                                actuals.append(final_actual)
                                predictions.append(predicted_cumulative)
                                claim_ids.append(claim['claim_id'])
                                errors.append(abs(final_actual - predicted_cumulative))
                        
                        # Create scatter plot
                        if actuals and predictions:
                            fig_test_scatter = go.Figure()
                            
                            # Color points by error magnitude
                            colors = ['red' if error > np.percentile(errors, 75) else 
                                    'orange' if error > np.percentile(errors, 50) else 
                                    'green' for error in errors]
                            
                            fig_test_scatter.add_trace(go.Scatter(
                                x=actuals,
                                y=predictions,
                                mode='markers',
                                name='Test Claims',
                                marker=dict(
                                    size=8,
                                    color=colors,
                                    opacity=0.7,
                                    line=dict(width=1, color='black')
                                ),
                                text=[f"Claim {cid}<br>Actual: ${act:,.0f}<br>Predicted: ${pred:,.0f}<br>Error: ${err:,.0f}" 
                                      for cid, act, pred, err in zip(claim_ids, actuals, predictions, errors)],
                                hovertemplate='%{text}<extra></extra>'
                            ))
                            
                            # Perfect prediction line
                            min_val = min(min(actuals), min(predictions))
                            max_val = max(max(actuals), max(predictions))
                            fig_test_scatter.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='blue', dash='dash', width=2)
                            ))
                            
                            # Over-reserved line (predicted > actual)
                            fig_test_scatter.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val * 1.1, max_val * 1.1],
                                mode='lines',
                                name='10% Over-reserved',
                                line=dict(color='red', dash='dot', width=1),
                                opacity=0.5
                            ))
                            
                            # Under-reserved line (predicted < actual)
                            fig_test_scatter.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val * 0.9, max_val * 0.9],
                                mode='lines',
                                name='10% Under-reserved',
                                line=dict(color='orange', dash='dot', width=1),
                                opacity=0.5
                            ))
                            
                            fig_test_scatter.update_layout(
                                title="Test Claims: Actual vs Predicted Final Cumulative Payments",
                                xaxis_title="Actual Final Cumulative Payment ($)",
                                yaxis_title="Predicted Final Cumulative Payment ($)",
                                height=600,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_test_scatter, use_container_width=True)
                            
                            # Summary statistics
                            over_reserved = sum(1 for pred, actual in zip(predictions, actuals) if pred > actual * 1.1)
                            under_reserved = sum(1 for pred, actual in zip(predictions, actuals) if pred < actual * 0.9)
                            well_reserved = len(predictions) - over_reserved - under_reserved
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Test Claims", len(predictions))
                            with col2:
                                st.metric("Over-reserved (>10%)", over_reserved, 
                                        delta=f"{over_reserved/len(predictions)*100:.1f}%")
                            with col3:
                                st.metric("Under-reserved (<10%)", under_reserved,
                                        delta=f"{under_reserved/len(predictions)*100:.1f}%")
                            with col4:
                                st.metric("Well-reserved (±10%)", well_reserved,
                                        delta=f"{well_reserved/len(predictions)*100:.1f}%")
                            
                            # Show worst predictions
                            st.markdown("**🔍 Worst Predictions (Highest Errors):**")
                            worst_indices = np.argsort(errors)[-10:]  # Top 10 worst
                            worst_data = []
                            for idx in reversed(worst_indices):
                                worst_data.append({
                                    'Claim ID': claim_ids[idx],
                                    'Actual': f"${actuals[idx]:,.0f}",
                                    'Predicted': f"${predictions[idx]:,.0f}",
                                    'Error': f"${errors[idx]:,.0f}",
                                    'Error %': f"{(errors[idx]/actuals[idx]*100):.1f}%",
                                    'Status': 'Over-reserved' if predictions[idx] > actuals[idx] else 'Under-reserved'
                                })
                            
                            worst_df = pd.DataFrame(worst_data)
                            st.dataframe(worst_df, use_container_width=True)
                
                # Show detailed predictions table
                st.markdown("**Detailed Predictions:**")
                pred_data = []
                for i, (period, actual, pred, error) in enumerate(zip(prediction_periods, actual_values, predictions, errors)):
                    pred_data.append({
                        'Period': period,
                        'Actual': f"${actual:,.0f}",
                        'Predicted': f"${pred:,.0f}",
                        'Error': f"${error:,.0f}",
                        'Error %': f"{(error/actual*100):.1f}%" if actual > 0 else "N/A"
                    })
                
                pred_df = pd.DataFrame(pred_data)
                st.dataframe(pred_df, use_container_width=True)
        else:
            st.info("Please train a model first in the 'Model Training' tab to see predictions for individual claims.")
        
        # Claim statistics
        st.markdown("**Claim Statistics:**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("First Payment Period", f"{selected_claim['first_payment_period']}")
        with col2:
            st.metric("Last Payment Period", f"{selected_claim['last_payment_period']}")
        with col3:
            duration = selected_claim['last_payment_period'] - selected_claim['first_payment_period'] + 1
            st.metric("Payment Duration", f"{duration} periods")
        with col4:
            avg_payment = selected_claim['total_payments'] / selected_claim['n_payments'] if selected_claim['n_payments'] > 0 else 0
            st.metric("Avg Payment", f"${avg_payment:,.0f}")

if __name__ == "__main__":
    main()
