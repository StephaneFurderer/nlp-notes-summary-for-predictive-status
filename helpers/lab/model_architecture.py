import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ClaimReservingLSTM_v1(nn.Module):
    """
    LSTM for predicting remaining claim payments and ultimate loss
    
    Key features:
    - Handles variable-length sequences
    - Outputs both remaining payments and ultimate loss
    - Uses packed sequences for efficiency
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(ClaimReservingLSTM_v1, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Prediction heads
        self.fc_remaining = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        self.fc_ultimate = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, lengths):
        """
        x: (batch, max_seq_len, input_size) - padded sequences
        lengths: (batch,) - actual lengths before padding
        """
        # Pack padded sequences for efficient processing
        packed = pack_padded_sequence(
            x, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Process through LSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Use last hidden state (from last layer)
        # Shape: (batch, hidden_size)
        last_hidden = hidden[-1]
        
        # Predict remaining and ultimate
        remaining = self.fc_remaining(last_hidden)
        ultimate = self.fc_ultimate(last_hidden)
        
        return remaining, ultimate


class ClaimReservingLSTM_v2(nn.Module):
    """
    Version 2: Added attention mechanism and deeper prediction heads
    
    Improvements over v1:
    - Attention layer over LSTM outputs (captures important periods)
    - Deeper prediction heads (64 -> 32 -> 16 -> 1)
    - Separate dropout rates for LSTM and FC layers
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, fc_dropout=0.3):
        super(ClaimReservingLSTM_v2, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Deeper prediction heads
        self.fc_remaining = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(32, 1)
        )
        
        self.fc_ultimate = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, lengths):
        """
        x: (batch, max_seq_len, input_size) - padded sequences
        lengths: (batch,) - actual lengths before padding
        """
        # Pack padded sequences
        packed = pack_padded_sequence(
            x, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Process through LSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply attention over sequence outputs
        # Mask out padded positions
        batch_size, max_len, hidden_size = output.shape
        mask = torch.arange(max_len).expand(batch_size, max_len).to(output.device) < lengths.unsqueeze(1)
        
        # Compute attention scores
        attention_scores = self.attention(output).squeeze(-1)  # (batch, max_len)
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, max_len)
        
        # Weighted sum of outputs
        context = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)  # (batch, hidden_size)
        
        # Predict remaining and ultimate
        remaining = self.fc_remaining(context)
        ultimate = self.fc_ultimate(context)
        
        return remaining, ultimate


# Convenience alias: Always points to the latest recommended version
ClaimReservingLSTM = ClaimReservingLSTM_v1

