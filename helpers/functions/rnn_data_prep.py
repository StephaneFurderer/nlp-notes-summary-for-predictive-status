"""
Prepare data for RNN model - LSTM-based individual claim reserving

Implementation based on "Micro-level Reserving for General Insurance Claims
using a Long Short-Term Memory Network" (Chaoubi et al., 2022)

Core methodology:
- LSTM network with dual tasks: classification (payment probability) + regression (payment amount)
- Sequential processing of claim development periods
- Multi-task learning with uncertainty-weighted loss balancing
- Individual claim-level predictions vs aggregate methods
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def prepare_lstm_sequences(df: pd.DataFrame,
                          claim_id_col: str = 'claim_id',
                          period_col: str = 'period',
                          payment_col: str = 'payment_amount',
                          expense_col: str = 'expense_amount',
                          status_col: str = 'claim_status',
                          split_col: str = 'dataset_split',
                          max_periods: int = 24,
                          min_periods: int = 3) -> Dict[str, Any]:
    """
    Convert claims dataframe to LSTM sequences for individual claim reserving.

    Based on paper methodology:
    - Each claim becomes a sequence of development periods
    - Static features (Sk): claim characteristics at opening
    - Dynamic features (Dk,j): time-varying information per period
    - Targets: (Ik,j, Yk,j) = (payment_indicator, payment_amount)

    Args:
        df: Input dataframe with claims data
        claim_id_col: Column name for unique claim identifier
        period_col: Column name for development period
        payment_col: Column name for payment amounts
        expense_col: Column name for expense amounts
        status_col: Column name for claim status
        split_col: Column name indicating train/val/test split
        max_periods: Maximum sequence length (n in paper)
        min_periods: Minimum periods required for a claim

    Returns:
        Dict containing processed sequences and metadata
    """
    print(f"Preparing LSTM sequences for {df[claim_id_col].nunique()} unique claims...")

    # Group by claim and ensure proper sorting
    claim_groups = df.sort_values([claim_id_col, period_col]).groupby(claim_id_col)

    sequences = []
    static_features = []
    targets = []
    claim_metadata = []

    for claim_id, claim_data in claim_groups:
        # Skip claims with insufficient periods
        if len(claim_data) < min_periods:
            continue

        # Extract static features (first period characteristics)
        static_feat = extract_static_features(claim_data.iloc[0])

        # Create sequence up to max_periods
        sequence_data = claim_data.head(max_periods)

        # Build dynamic features and targets
        dynamic_seq, target_seq = build_sequence_features(
            sequence_data, payment_col, period_col, max_periods
        )

        sequences.append(dynamic_seq)
        static_features.append(static_feat)
        targets.append(target_seq)

        # Store metadata
        claim_metadata.append({
            'claim_id': claim_id,
            'dataset_split': claim_data.iloc[0][split_col],
            'actual_periods': len(sequence_data),
            'total_paid': claim_data[payment_col].sum()
        })

    print(f"Created {len(sequences)} sequences")

    return {
        'sequences': np.array(sequences),
        'static_features': np.array(static_features),
        'targets': np.array(targets),
        'metadata': pd.DataFrame(claim_metadata),
        'max_periods': max_periods
    }

def extract_static_features(first_period_data: pd.Series) -> np.ndarray:
    """
    Extract static claim characteristics (Sk in paper notation).

    These are features known at claim opening that don't change over time.
    Examples: accident date, claimant age, policy type, etc.
    """
    static_cols = []

    static_cols.append(first_period_data['clmCause'])
    

    return np.array(static_cols) if static_cols else np.array([0])

def build_sequence_features(claim_data: pd.DataFrame,
                          payment_col: str,
                          period_col: str,
                          max_periods: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build dynamic feature sequences and targets for LSTM.

    Returns:
        dynamic_seq: [max_periods, n_dynamic_features]
        target_seq: [max_periods, 2] where targets are [payment_indicator, payment_amount]
    """
    # Initialize sequences
    dynamic_features = np.zeros((max_periods, get_dynamic_feature_dim()))
    targets = np.zeros((max_periods, 2))  # [indicator, amount]

    for idx, (_, row) in enumerate(claim_data.iterrows()):
        if idx >= max_periods:
            break

        # Dynamic features (Dk,j in paper)
        dynamic_features[idx] = build_dynamic_feature_vector(row, idx + 1)

        # Targets (Ik,j, Yk,j in paper)
        payment = row[payment_col] if pd.notna(row[payment_col]) else 0.0
        targets[idx, 0] = 1.0 if payment != 0.0 else 0.0  # Payment indicator
        targets[idx, 1] = payment  # Payment amount

    return dynamic_features, targets

def get_dynamic_feature_dim() -> int:
    """
    Calculate dimension of dynamic features.

    Based on paper: includes development period, observed indicator,
    payment history, and other time-varying characteristics.
    """
    return 5  # [period, is_observed, cum_paid, periods_since_last_payment, claim_open]

def build_dynamic_feature_vector(row: pd.Series, period: int) -> np.ndarray:
    """
    Build dynamic feature vector for a single period (Dk,j in paper).

    Features include:
    - Development period j
    - Observation indicator rk,j (1 if period j <= tk)
    - Cumulative payments
    - Periods since last payment
    - Claim status indicators
    """
    features = np.zeros(get_dynamic_feature_dim())

    features[0] = period  # Development period
    features[1] = 1.0  # Observed indicator (assuming all input periods are observed)

    # Cumulative paid (if available)
    if 'cumulative_paid' in row.index:
        features[2] = row['cumulative_paid']
    elif 'payment_amount' in row.index:
        features[2] = row['payment_amount']  # Fallback to incremental

    # Periods since last payment (simplified)
    features[3] = period if row.get('payment_amount', 0) == 0 else 0

    # Claim open indicator
    features[4] = 1.0 if row.get('claim_status', 'OPEN') == 'OPEN' else 0.0

    return features

def censor_large_payments(sequences_data: Dict[str, Any],
                         threshold_percentile: float = 0.995) -> Dict[str, Any]:
    """
    Censor extremely large payments to improve LSTM training stability.

    Based on paper Section 4.2: payments above threshold u are censored
    to min(payment, u) to reduce variance and improve learning.

    Args:
        sequences_data: Output from prepare_lstm_sequences
        threshold_percentile: Percentile for censoring threshold

    Returns:
        Modified sequences_data with censored payments
    """
    targets = sequences_data['targets']

    # Calculate threshold from payment amounts (excluding zeros)
    all_payments = targets[:, :, 1].flatten()
    non_zero_payments = all_payments[all_payments > 0]

    if len(non_zero_payments) > 0:
        threshold = np.percentile(non_zero_payments, threshold_percentile * 100)
        print(f"Censoring payments above ${threshold:,.2f} ({threshold_percentile*100}th percentile)")

        # Apply censoring
        censored_targets = targets.copy()
        censored_targets[:, :, 1] = np.minimum(censored_targets[:, :, 1], threshold)

        sequences_data['targets'] = censored_targets
        sequences_data['censoring_threshold'] = threshold

        n_censored = np.sum(targets[:, :, 1] > threshold)
        print(f"Censored {n_censored} payment observations")

    return sequences_data

def scale_features(sequences_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scale features for LSTM training stability.

    Based on paper Eq. (1): Y*k,j = (Yk,j - μT) / σT
    Features are centered and scaled using training set statistics.
    """
    metadata = sequences_data['metadata']
    train_mask = metadata['dataset_split'] == 'train'

    # Scale payment amounts using training data statistics
    targets = sequences_data['targets']
    train_targets = targets[train_mask]

    # Calculate training set statistics for payment amounts
    train_payments = train_targets[:, :, 1].flatten()
    payment_mean = np.mean(train_payments)
    payment_std = np.std(train_payments)

    if payment_std > 0:
        targets[:, :, 1] = (targets[:, :, 1] - payment_mean) / payment_std
        sequences_data['targets'] = targets
        sequences_data['payment_scaler'] = {'mean': payment_mean, 'std': payment_std}
        print(f"Scaled payments: mean={payment_mean:.2f}, std={payment_std:.2f}")

    # Scale dynamic features
    sequences = sequences_data['sequences']
    train_sequences = sequences[train_mask]

    # Scale each feature dimension separately
    for feat_idx in range(sequences.shape[-1]):
        feat_values = train_sequences[:, :, feat_idx].flatten()
        feat_mean = np.mean(feat_values)
        feat_std = np.std(feat_values)

        if feat_std > 0:
            sequences[:, :, feat_idx] = (sequences[:, :, feat_idx] - feat_mean) / feat_std

    sequences_data['sequences'] = sequences

    return sequences_data

def split_sequences_by_dataset(sequences_data: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Split sequences into train/validation/test sets based on dataset_split column.

    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing sequences and targets
    """
    metadata = sequences_data['metadata']
    splits = {}

    for split_name in ['train', 'val', 'test']:
        mask = metadata['dataset_split'] == split_name

        if mask.any():
            splits[split_name] = {
                'sequences': sequences_data['sequences'][mask],
                'static_features': sequences_data['static_features'][mask],
                'targets': sequences_data['targets'][mask],
                'metadata': metadata[mask].reset_index(drop=True)
            }
            print(f"{split_name}: {mask.sum()} claims")

    return splits

class LSTMClaimReservingModel(nn.Module):
    """
    LSTM-based individual claim reserving model.

    Based on paper architecture:
    - LSTM backbone with n-1 modules
    - Dual tasks: classification (payment probability) + regression (payment amount)
    - Static context embedding + dynamic sequence processing
    - Multi-task loss with uncertainty weighting
    """

    def __init__(self,
                 static_dim: int,
                 dynamic_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 context_dim: int = 32,
                 max_periods: int = 24):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_periods = max_periods

        # Static feature embedding (Ck,0 in paper)
        self.static_embedding = nn.Sequential(
            nn.Linear(static_dim, context_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=dynamic_dim + context_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # Classification head (payment probability)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Regression head (payment amount)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # Task uncertainty parameters (Eq. 4 in paper)
        self.log_var_class = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))

    def forward(self, static_features: torch.Tensor,
                dynamic_sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM model.

        Args:
            static_features: [batch_size, static_dim]
            dynamic_sequences: [batch_size, seq_len, dynamic_dim]

        Returns:
            payment_probs: [batch_size, seq_len, 1]
            payment_amounts: [batch_size, seq_len, 1]
        """
        batch_size, seq_len, _ = dynamic_sequences.shape

        # Process static features
        context = self.static_embedding(static_features)  # [batch_size, context_dim]

        # Expand context to match sequence length
        context_expanded = context.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, context_dim]

        # Combine static context with dynamic features
        combined_input = torch.cat([dynamic_sequences, context_expanded], dim=-1)

        # LSTM processing
        lstm_out, _ = self.lstm(combined_input)  # [batch_size, seq_len, hidden_dim]

        # Dual task heads
        payment_probs = self.classifier(lstm_out)  # [batch_size, seq_len, 1]
        payment_amounts = self.regressor(lstm_out)  # [batch_size, seq_len, 1]

        return payment_probs, payment_amounts

def multi_task_loss(pred_probs: torch.Tensor,
                   pred_amounts: torch.Tensor,
                   target_indicators: torch.Tensor,
                   target_amounts: torch.Tensor,
                   log_var_class: torch.Tensor,
                   log_var_reg: torch.Tensor,
                   alpha: float = 1.0) -> torch.Tensor:
    """
    Multi-task loss with uncertainty weighting (Eq. 4 in paper).

    Args:
        pred_probs: Predicted payment probabilities
        pred_amounts: Predicted payment amounts
        target_indicators: Target payment indicators (0/1)
        target_amounts: Target payment amounts
        log_var_class: Log variance for classification task
        log_var_reg: Log variance for regression task
        alpha: Scale parameter for classification loss

    Returns:
        Combined loss
    """
    # Classification loss (binary cross-entropy)
    bce_loss = nn.BCELoss()(pred_probs.squeeze(), target_indicators)

    # Regression loss (mean squared error, only for non-zero payments)
    non_zero_mask = target_indicators > 0
    if non_zero_mask.sum() > 0:
        mse_loss = nn.MSELoss()(
            pred_amounts.squeeze()[non_zero_mask],
            target_amounts[non_zero_mask]
        )
    else:
        mse_loss = torch.tensor(0.0, device=pred_amounts.device)

    # Uncertainty-weighted multi-task loss (Eq. 4)
    precision_class = torch.exp(-log_var_class)
    precision_reg = torch.exp(-log_var_reg)

    loss = (alpha * precision_class * bce_loss + log_var_class +
            precision_reg * mse_loss + log_var_reg)

    return loss

def split_data_for_rnn(data, split_ratio: List[float] = [0.8, 0.1, 0.1]):
    """
    Elegantly split data into training, validation, and testing sets.

    Args:
        data (array-like or pd.DataFrame): The dataset to split.
        split_ratio (list of float): Proportions for train, val, and test splits. Must sum to 1.

    Returns:
        tuple: (train_data, val_data, test_data)
    """
    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError("Split ratios must sum to 1.")
    n = len(data)
    train_end = int(split_ratio[0] * n)
    val_end = train_end + int(split_ratio[1] * n)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data

def main():
    """
    Main function demonstrating the LSTM data preparation pipeline.
    """
    # This would be called with your closed claims dataframe
    # df_closed = pd.read_parquet("path_to_closed_claims.parquet")

    print("LSTM Data Preparation Pipeline")
    print("==============================")

    # Example usage (you'll replace this with your actual dataframe)
    # sequences_data = prepare_lstm_sequences(df_closed)
    # sequences_data = censor_large_payments(sequences_data)
    # sequences_data = scale_features(sequences_data)
    # splits = split_sequences_by_dataset(sequences_data)

    print("Pipeline complete. Ready for LSTM training.")

if __name__ == "__main__":
    main()