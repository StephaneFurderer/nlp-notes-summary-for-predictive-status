import torch
import pickle
from datetime import datetime
import os

# ============================================================================
# SAVING THE MODEL
# ============================================================================

def save_claims_model(model, scaler, train_dataset, test_metrics, hyperparameters, 
                      model_name="lstm_claims_reserving"):
    """
    Save a complete PyTorch model with all necessary metadata
    
    Args:
        model: trained PyTorch model
        scaler: fitted sklearn scaler
        train_dataset: the training dataset object
        test_metrics: dict of evaluation metrics (R², MAE, etc.)
        hyperparameters: dict of model hyperparameters
        model_name: base name for the saved files
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/{model_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Save model weights (state_dict - MOST IMPORTANT)
    torch.save(model.state_dict(), f"{save_dir}/model_weights.pt")
    
    # 2. Save complete checkpoint (optional - for resuming training)
    checkpoint = {
        'epoch': hyperparameters.get('epochs', None),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,  # Add if you saved optimizer
        'train_loss': hyperparameters.get('final_train_loss', None),
        'test_loss': hyperparameters.get('final_test_loss', None),
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pt")
    
    # 3. Save scaler (CRITICAL for predictions)
    with open(f"{save_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # 4. Save metadata as JSON
    import json
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'pytorch_version': torch.__version__,
        
        # Model architecture
        'architecture': {
            'model_type': model.__class__.__name__,
            'input_size': hyperparameters.get('input_size', 1),
            'hidden_size': hyperparameters.get('hidden_size', 64),
            'num_layers': hyperparameters.get('num_layers', 2),
            'dropout': hyperparameters.get('dropout', 0.2),
            'output_size': hyperparameters.get('output_size', 2),  # remaining + ultimate
        },
        
        # Training config
        'training': {
            'epochs': hyperparameters.get('epochs', 100),
            'batch_size': hyperparameters.get('batch_size', 32),
            'learning_rate': hyperparameters.get('lr', 0.001),
            'optimizer': hyperparameters.get('optimizer', 'Adam'),
        },
        
        # Dataset info
        'data': {
            'num_train_claims': len(train_dataset),
            'num_features': hyperparameters.get('input_size', 1),
            'scaler_type': scaler.__class__.__name__,
            'scaler_min': float(scaler.data_min_[0]) if hasattr(scaler, 'data_min_') else None,
            'scaler_max': float(scaler.data_max_[0]) if hasattr(scaler, 'data_max_') else None,
        },
        
        # Performance metrics
        'performance': test_metrics,
    }
    
    with open(f"{save_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 5. Save a human-readable summary
    with open(f"{save_dir}/README.txt", 'w') as f:
        f.write(f"LSTM Claims Reserving Model\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Saved: {timestamp}\n")
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Hidden Size: {hyperparameters.get('hidden_size')}\n")
        f.write(f"Num Layers: {hyperparameters.get('num_layers')}\n\n")
        f.write(f"Test Performance:\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value}\n")
    
    print(f"✅ Model saved to: {save_dir}")
    return save_dir


# ============================================================================
# LOADING THE MODEL
# ============================================================================

def load_claims_model(save_dir, device='cpu'):
    """
    Load a complete PyTorch model with all metadata
    
    Args:
        save_dir: directory containing saved model files
        device: 'cpu' or 'cuda'
    
    Returns:
        model, scaler, metadata
    """
    import json
    
    # 1. Load metadata
    with open(f"{save_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # 2. Recreate model architecture
    from torch import nn
    
    class ClaimReservingLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, output_size=2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
            self.fc_remaining = nn.Linear(hidden_size, 1)
            self.fc_ultimate = nn.Linear(hidden_size, 1)
        
        def forward(self, x, lengths):
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            
            packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, (hidden, cell) = self.lstm(packed_input)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            
            last_outputs = output[torch.arange(output.size(0)), lengths - 1]
            
            remaining = self.fc_remaining(last_outputs)
            ultimate = self.fc_ultimate(last_outputs)
            
            return remaining, ultimate
    
    arch = metadata['architecture']
    model = ClaimReservingLSTM(
        input_size=arch['input_size'],
        hidden_size=arch['hidden_size'],
        num_layers=arch['num_layers'],
        dropout=arch['dropout'],
        output_size=arch['output_size']
    )
    
    # 3. Load model weights
    model.load_state_dict(torch.load(f"{save_dir}/model_weights.pt", map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # 4. Load scaler
    with open(f"{save_dir}/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✅ Model loaded from: {save_dir}")
    print(f"   Architecture: {arch['hidden_size']}-hidden, {arch['num_layers']}-layer LSTM")
    print(f"   Performance: R²={metadata['performance'].get('r2_ultimate', 'N/A'):.3f}")
    
    return model, scaler, metadata


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
""""
# After training your model:
test_metrics = {
    'r2_ultimate': 0.85,
    'r2_remaining': 0.78,
    'mae_ultimate': 5432.10,
    'mae_remaining': 3210.45,
}

hyperparameters = {
    'input_size': 1,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'output_size': 2,
    'epochs': 100,
    'batch_size': 32,
    'lr': 0.001,
    'optimizer': 'Adam',
}

# Save
save_path = save_claims_model(
    model=model,
    scaler=train_dataset.scaler,
    train_dataset=train_dataset,
    test_metrics=test_metrics,
    hyperparameters=hyperparameters,
    model_name="lstm_claims_v1"
)

# Later, load it back
loaded_model, loaded_scaler, loaded_metadata = load_claims_model(save_path, device='cpu')

# Use for prediction
with torch.no_grad():
    observed_payments = [1000, 2000, 1500]
    X = loaded_scaler.transform(np.array(observed_payments).reshape(-1, 1))
    X = torch.FloatTensor(X).unsqueeze(0)
    length = torch.LongTensor([len(observed_payments)])
    
    pred_remaining, pred_ultimate = loaded_model(X, length)
    print(f"Predicted remaining: ${pred_remaining.item():,.2f}")
    print(f"Predicted ultimate: ${pred_ultimate.item():,.2f}")

"""