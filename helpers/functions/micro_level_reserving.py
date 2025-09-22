"""
Micro-Level Reserving Framework using LSTM Neural Networks

This module implements the micro-level reserving approach described in:
"Micro-level reserving for general insurance claims using a long short-term memory network"

Key Features:
1. Dual-task LSTM (classification + regression) for payment prediction
2. Sequential feature extraction for individual claims over time
3. Extreme payment handling using Generalized Pareto Distribution
4. Integration with existing NLP features from claim notes
5. Comparison with traditional reserving methods
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score
from scipy import stats
import warnings
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import os

warnings.filterwarnings('ignore')

class ClaimSequenceDataset(Dataset):
    """
    PyTorch Dataset for individual claim sequences
    """
    
    def __init__(self, sequences: List[np.ndarray], 
                 classification_targets: List[int],
                 regression_targets: List[float],
                 sequence_lengths: List[int]):
        self.sequences = sequences
        self.classification_targets = classification_targets
        self.regression_targets = regression_targets
        self.sequence_lengths = sequence_lengths
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': torch.FloatTensor(self.sequences[idx]),
            'classification_target': torch.LongTensor([self.classification_targets[idx]]),
            'regression_target': torch.FloatTensor([self.regression_targets[idx]]),
            'sequence_length': self.sequence_lengths[idx]
        }

class DualTaskLSTM(nn.Module):
    """
    Dual-task LSTM model for micro-level reserving
    
    Architecture:
    - Shared LSTM layers for sequence processing
    - Classification head for payment occurrence prediction
    - Regression head for payment amount prediction
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(DualTaskLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Shared LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Classification head (binary: payment occurs or not)
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)  # Binary classification
        )
        
        # Regression head (payment amount)
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x, sequence_lengths=None):
        # Pack padded sequences for efficient LSTM processing
        if sequence_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, sequence_lengths, 
                                                batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Unpack sequences
        if sequence_lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Use the last valid output for each sequence
        batch_size = lstm_out.size(0)
        last_outputs = []
        
        for i in range(batch_size):
            if sequence_lengths is not None:
                last_idx = sequence_lengths[i] - 1
            else:
                last_idx = lstm_out.size(1) - 1
            last_outputs.append(lstm_out[i, last_idx, :])
        
        final_output = torch.stack(last_outputs)
        
        # Dual-task outputs
        classification_output = self.classification_head(final_output)
        regression_output = self.regression_head(final_output)
        
        return classification_output, regression_output

class ExtremePaymentHandler:
    """
    Handles extreme payments using Generalized Pareto Distribution
    """
    
    def __init__(self, threshold_percentile: float = 95.0):
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.gpd_params = None
        self.is_fitted = False
        
    def fit(self, payments: np.ndarray):
        """
        Fit GPD to excess payments above threshold
        """
        # Calculate threshold
        self.threshold = np.percentile(payments, self.threshold_percentile)
        
        # Get excess payments
        excess_payments = payments[payments > self.threshold] - self.threshold
        
        if len(excess_payments) > 10:  # Need sufficient data for GPD fitting
            try:
                # Fit Generalized Pareto Distribution
                shape, loc, scale = stats.genpareto.fit(excess_payments, floc=0)
                self.gpd_params = {'shape': shape, 'loc': loc, 'scale': scale}
                self.is_fitted = True
            except:
                # Fallback to normal distribution if GPD fitting fails
                self.gpd_params = {'shape': 0, 'loc': 0, 'scale': np.std(excess_payments)}
                self.is_fitted = True
        else:
            self.gpd_params = {'shape': 0, 'loc': 0, 'scale': np.std(payments)}
            self.is_fitted = True
            
    def predict_tail_probability(self, amount: float) -> float:
        """
        Predict probability of exceeding given amount
        """
        if not self.is_fitted or amount <= self.threshold:
            return 0.0
            
        excess = amount - self.threshold
        try:
            prob = 1 - stats.genpareto.cdf(excess, 
                                         self.gpd_params['shape'],
                                         loc=self.gpd_params['loc'],
                                         scale=self.gpd_params['scale'])
            return max(0.0, min(1.0, prob))
        except:
            return 0.0

class MicroLevelReservingModel:
    """
    Main class for micro-level reserving using LSTM networks
    """
    
    def __init__(self, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, learning_rate: float = 0.001,
                 extreme_payment_threshold: float = 95.0):
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Components
        self.model = None
        self.sequence_scaler = StandardScaler()
        self.amount_scaler = MinMaxScaler()
        self.extreme_handler = ExtremePaymentHandler(extreme_payment_threshold)
        
        # Training history
        self.training_history = {
            'classification_loss': [],
            'regression_loss': [],
            'total_loss': [],
            'classification_accuracy': [],
            'regression_mae': []
        }
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_sequential_features(self, df_txn: pd.DataFrame, 
                                  nlp_features_df: pd.DataFrame = None,
                                  max_sequence_length: int = 50) -> Tuple[List, List, List, List]:
        """
        Prepare sequential features for individual claims
        
        Args:
            df_txn: Transaction dataframe with claim sequences
            nlp_features_df: Optional NLP features dataframe
            max_sequence_length: Maximum sequence length for padding
            
        Returns:
            Tuple of (sequences, classification_targets, regression_targets, sequence_lengths)
        """
        
        sequences = []
        classification_targets = []
        regression_targets = []
        sequence_lengths = []
        
        # Get unique claims
        unique_claims = df_txn['clmNum'].unique()
        
        for claim_num in unique_claims:
            claim_data = df_txn[df_txn['clmNum'] == claim_num].sort_values('datetxn')
            
            if len(claim_data) < 2:  # Need at least 2 transactions
                continue
                
            # Extract sequential features for each time step
            claim_sequence = []
            
            for idx, row in claim_data.iterrows():
                # Time-based features
                time_features = [
                    row.get('days_since_first_txn', 0),
                    row.get('num_transactions', 0),
                    row.get('development_stage', 0),
                ]
                
                # Financial features
                financial_features = [
                    row.get('paid_cumsum', 0),
                    row.get('expense_cumsum', 0),
                    row.get('recovery_cumsum', 0),
                    row.get('reserve_cumsum', 0),
                    row.get('incurred_cumsum', 0),
                    row.get('paid', 0),
                    row.get('expense', 0),
                    row.get('reserve', 0),
                ]
                
                # Change features
                change_features = [
                    row.get('paid_change', 0),
                    row.get('expense_change', 0),
                    row.get('reserve_change', 0),
                    row.get('incurred_change', 0),
                ]
                
                # Combine features
                step_features = time_features + financial_features + change_features
                claim_sequence.append(step_features)
            
            # Pad sequence to max_length
            if len(claim_sequence) > max_sequence_length:
                claim_sequence = claim_sequence[-max_sequence_length:]  # Keep last N steps
            
            # Pad with zeros if shorter
            while len(claim_sequence) < max_sequence_length:
                claim_sequence.insert(0, [0] * len(claim_sequence[0]))
            
            sequences.append(np.array(claim_sequence))
            sequence_lengths.append(len(claim_data))
            
            # Determine targets (for next period prediction)
            if len(claim_data) > 1:
                # Classification target: will there be a payment in the next period?
                last_paid = claim_data['paid_cumsum'].iloc[-1]
                second_last_paid = claim_data['paid_cumsum'].iloc[-2] if len(claim_data) > 1 else 0
                has_payment = 1 if (last_paid - second_last_paid) > 0 else 0
                classification_targets.append(has_payment)
                
                # Regression target: amount of payment (if any)
                payment_amount = max(0, last_paid - second_last_paid)
                regression_targets.append(payment_amount)
            else:
                classification_targets.append(0)
                regression_targets.append(0.0)
        
        return sequences, classification_targets, regression_targets, sequence_lengths
    
    def train(self, df_txn: pd.DataFrame, nlp_features_df: pd.DataFrame = None,
              epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2):
        """
        Train the micro-level reserving model
        """
        
        print("Preparing sequential features...")
        sequences, classification_targets, regression_targets, sequence_lengths = self.prepare_sequential_features(
            df_txn, nlp_features_df
        )
        
        if len(sequences) == 0:
            raise ValueError("No valid claim sequences found")
        
        print(f"Prepared {len(sequences)} claim sequences")
        
        # Convert to numpy arrays
        sequences = np.array(sequences)
        classification_targets = np.array(classification_targets)
        regression_targets = np.array(regression_targets)
        
        # Fit scalers
        # Flatten sequences for scaling
        sequences_flat = sequences.reshape(-1, sequences.shape[-1])
        sequences_scaled_flat = self.sequence_scaler.fit_transform(sequences_flat)
        sequences_scaled = sequences_scaled_flat.reshape(sequences.shape)
        
        # Scale regression targets
        regression_targets_scaled = self.amount_scaler.fit_transform(
            regression_targets.reshape(-1, 1)
        ).flatten()
        
        # Fit extreme payment handler
        all_payments = []
        for claim_data in [df_txn[df_txn['clmNum'] == claim] for claim in df_txn['clmNum'].unique()]:
            if len(claim_data) > 1:
                payments = claim_data['paid'].values
                all_payments.extend(payments[payments > 0])
        
        if len(all_payments) > 0:
            self.extreme_handler.fit(np.array(all_payments))
        
        # Split data
        n_samples = len(sequences_scaled)
        n_train = int(n_samples * (1 - validation_split))
        
        train_indices = np.random.choice(n_samples, n_train, replace=False)
        val_indices = np.setdiff1d(np.arange(n_samples), train_indices)
        
        # Create datasets
        train_dataset = ClaimSequenceDataset(
            sequences_scaled[train_indices].tolist(),
            classification_targets[train_indices].tolist(),
            regression_targets_scaled[train_indices].tolist(),
            [sequence_lengths[i] for i in train_indices]
        )
        
        val_dataset = ClaimSequenceDataset(
            sequences_scaled[val_indices].tolist(),
            classification_targets[val_indices].tolist(),
            regression_targets_scaled[val_indices].tolist(),
            [sequence_lengths[i] for i in val_indices]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = sequences.shape[-1]
        self.model = DualTaskLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss functions and optimizer
        classification_criterion = nn.CrossEntropyLoss()
        regression_criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_class_loss = 0
            train_reg_loss = 0
            train_total_loss = 0
            train_class_acc = 0
            train_reg_mae = 0
            train_batches = 0
            
            for batch in train_loader:
                sequences_batch = batch['sequence'].to(self.device)
                class_targets = batch['classification_target'].to(self.device).squeeze()
                reg_targets = batch['regression_target'].to(self.device).squeeze()
                seq_lengths = batch['sequence_length']
                
                optimizer.zero_grad()
                
                # Forward pass
                class_output, reg_output = self.model(sequences_batch, seq_lengths)
                
                # Calculate losses
                class_loss = classification_criterion(class_output, class_targets)
                reg_loss = regression_criterion(reg_output.squeeze(), reg_targets)
                total_loss = class_loss + reg_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Track metrics
                train_class_loss += class_loss.item()
                train_reg_loss += reg_loss.item()
                train_total_loss += total_loss.item()
                
                # Accuracy
                _, predicted = torch.max(class_output, 1)
                train_class_acc += (predicted == class_targets).float().mean().item()
                
                # MAE
                train_reg_mae += torch.mean(torch.abs(reg_output.squeeze() - reg_targets)).item()
                train_batches += 1
            
            # Validation phase
            self.model.eval()
            val_class_loss = 0
            val_reg_loss = 0
            val_total_loss = 0
            val_class_acc = 0
            val_reg_mae = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    sequences_batch = batch['sequence'].to(self.device)
                    class_targets = batch['classification_target'].to(self.device).squeeze()
                    reg_targets = batch['regression_target'].to(self.device).squeeze()
                    seq_lengths = batch['sequence_length']
                    
                    class_output, reg_output = self.model(sequences_batch, seq_lengths)
                    
                    class_loss = classification_criterion(class_output, class_targets)
                    reg_loss = regression_criterion(reg_output.squeeze(), reg_targets)
                    total_loss = class_loss + reg_loss
                    
                    val_class_loss += class_loss.item()
                    val_reg_loss += reg_loss.item()
                    val_total_loss += total_loss.item()
                    
                    _, predicted = torch.max(class_output, 1)
                    val_class_acc += (predicted == class_targets).float().mean().item()
                    val_reg_mae += torch.mean(torch.abs(reg_output.squeeze() - reg_targets)).item()
                    val_batches += 1
            
            # Record metrics
            self.training_history['classification_loss'].append(train_class_loss / train_batches)
            self.training_history['regression_loss'].append(train_reg_loss / train_batches)
            self.training_history['total_loss'].append(train_total_loss / train_batches)
            self.training_history['classification_accuracy'].append(train_class_acc / train_batches)
            self.training_history['regression_mae'].append(train_reg_mae / train_batches)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: "
                      f"Train Loss: {train_total_loss/train_batches:.4f}, "
                      f"Train Acc: {train_class_acc/train_batches:.4f}, "
                      f"Train MAE: {train_reg_mae/train_batches:.4f}, "
                      f"Val Loss: {val_total_loss/val_batches:.4f}, "
                      f"Val Acc: {val_class_acc/val_batches:.4f}, "
                      f"Val MAE: {val_reg_mae/val_batches:.4f}")
        
        print("Training completed!")
        
        return {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'final_train_loss': self.training_history['total_loss'][-1],
            'final_val_loss': val_total_loss / val_batches,
            'final_train_acc': self.training_history['classification_accuracy'][-1],
            'final_val_acc': val_class_acc / val_batches,
            'final_train_mae': self.training_history['regression_mae'][-1],
            'final_val_mae': val_reg_mae / val_batches
        }
    
    def predict_reserves(self, df_txn: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
        """
        Predict individual claim reserves using the trained model
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Prepare features for all claims
        sequences, _, _, sequence_lengths = self.prepare_sequential_features(df_txn)
        
        if len(sequences) == 0:
            return pd.DataFrame()
        
        # Scale sequences
        sequences_flat = np.array(sequences).reshape(-1, sequences[0].shape[-1])
        sequences_scaled_flat = self.sequence_scaler.transform(sequences_flat)
        sequences_scaled = sequences_scaled_flat.reshape(len(sequences), -1, sequences[0].shape[-1])
        
        predictions = []
        
        with torch.no_grad():
            for i, (seq, seq_len) in enumerate(zip(sequences_scaled, sequence_lengths)):
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                
                # Predict next period
                class_output, reg_output = self.model(seq_tensor, [seq_len])
                
                # Convert predictions
                class_proba = torch.softmax(class_output, dim=1)
                payment_prob = class_proba[0, 1].item()  # Probability of payment
                
                reg_amount_scaled = reg_output[0, 0].item()
                reg_amount = self.amount_scaler.inverse_transform([[reg_amount_scaled]])[0, 0]
                
                # Apply extreme payment adjustment
                if self.extreme_handler.is_fitted:
                    extreme_prob = self.extreme_handler.predict_tail_probability(reg_amount)
                    reg_amount *= (1 + extreme_prob * 0.5)  # Increase by up to 50% for extreme cases
                
                predictions.append({
                    'clmNum': df_txn['clmNum'].unique()[i],
                    'payment_probability': payment_prob,
                    'predicted_payment': reg_amount,
                    'adjusted_reserve': reg_amount * payment_prob,
                    'extreme_adjustment': extreme_prob if self.extreme_handler.is_fitted else 0.0
                })
        
        return pd.DataFrame(predictions)
    
    def save_model(self, filepath: str):
        """
        Save the trained model and components
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            },
            'sequence_scaler': self.sequence_scaler,
            'amount_scaler': self.amount_scaler,
            'extreme_handler': self.extreme_handler,
            'training_history': self.training_history
        }
        
        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = torch.load(filepath, map_location=self.device)
        
        # Load configuration
        config = model_data['model_config']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.learning_rate = config['learning_rate']
        
        # Load scalers
        self.sequence_scaler = model_data['sequence_scaler']
        self.amount_scaler = model_data['amount_scaler']
        self.extreme_handler = model_data['extreme_handler']
        self.training_history = model_data['training_history']
        
        # Initialize and load model
        input_size = len(self.sequence_scaler.feature_names_in_) if hasattr(self.sequence_scaler, 'feature_names_in_') else 15
        self.model = DualTaskLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {filepath}")

def compare_with_traditional_reserving(micro_level_results: pd.DataFrame, 
                                     traditional_results: pd.DataFrame) -> Dict[str, float]:
    """
    Compare micro-level reserving with traditional methods
    
    Args:
        micro_level_results: Results from micro-level model
        traditional_results: Results from traditional chain-ladder or similar
        
    Returns:
        Dictionary with comparison metrics
    """
    
    if len(micro_level_results) == 0 or len(traditional_results) == 0:
        return {}
    
    # Merge results on claim number
    comparison = micro_level_results.merge(
        traditional_results, 
        on='clmNum', 
        suffixes=('_micro', '_traditional'),
        how='inner'
    )
    
    if len(comparison) == 0:
        return {}
    
    # Calculate comparison metrics
    micro_reserves = comparison['adjusted_reserve']
    traditional_reserves = comparison['reserve_traditional'] if 'reserve_traditional' in comparison.columns else comparison['reserve']
    
    metrics = {
        'total_micro_reserves': micro_reserves.sum(),
        'total_traditional_reserves': traditional_reserves.sum(),
        'difference_absolute': abs(micro_reserves.sum() - traditional_reserves.sum()),
        'difference_percentage': abs(micro_reserves.sum() - traditional_reserves.sum()) / traditional_reserves.sum() * 100,
        'correlation': micro_reserves.corr(traditional_reserves),
        'mean_absolute_error': mean_absolute_error(traditional_reserves, micro_reserves),
        'root_mean_squared_error': np.sqrt(mean_squared_error(traditional_reserves, micro_reserves))
    }
    
    return metrics
