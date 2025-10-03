import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler


class ClaimDataset(Dataset):
    """
    Custom Dataset for variable-length claim sequences
    
    Args:
        data: List of dicts, each containing:
            - 'input_sequence': numpy array of shape [seq_len, n_features]
            - 'target_remaining': float (remaining payments to be made)
            - 'target_ultimate': float (ultimate loss amount)
            - 'observed_periods': int (number of periods observed)
        scaler: Optional pre-fitted scaler (for test/val sets)
    """
    
    def __init__(self, data, scaler=None):
        self.data = data
        
        # Fit scaler on training data
        if scaler is None:
            # Stack all 2D sequences vertically
            all_sequences = np.vstack([d['input_sequence'] for d in data])
            # all_sequences shape: [total_timesteps, n_features]
            self.scaler = StandardScaler()
            self.scaler.fit(all_sequences)
        else:
            self.scaler = scaler
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Normalize input sequence (2D: [seq_len, n_features])
        X = self.scaler.transform(sample['input_sequence'])
        X = torch.FloatTensor(X)
        
        # Targets (we'll predict both remaining and ultimate)
        y_remaining = torch.FloatTensor([sample['target_remaining']])
        y_ultimate = torch.FloatTensor([sample['target_ultimate']])
        
        # Also return the length for padding/packing
        seq_length = torch.LongTensor([len(sample['input_sequence'])])

        # return the as of date to track the performance 
        observed_periods = torch.LongTensor([sample['observed_periods']]) 
        
        return X, y_remaining, y_ultimate, seq_length, observed_periods


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences
    
    Args:
        batch: List of tuples from __getitem__
    
    Returns:
        Padded and stacked tensors ready for model input
    """
    X_list, y_remaining_list, y_ultimate_list, lengths, obs_periods = zip(*batch)
    
    # Pad sequences to same length
    X_padded = pad_sequence(X_list, batch_first=True, padding_value=0)
    
    # Stack targets
    y_remaining = torch.stack(y_remaining_list)
    y_ultimate = torch.stack(y_ultimate_list)
    lengths = torch.stack(lengths).squeeze()
    obs_periods = torch.stack(obs_periods).squeeze()
    
    return X_padded, y_remaining, y_ultimate, lengths, obs_periods

