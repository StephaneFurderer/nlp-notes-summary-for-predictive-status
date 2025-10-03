import torch
import pickle
from datetime import datetime
import os
import json


def save_claims_model(
    model: torch.nn.Module,
    scaler: object,
    train_dataset: object,
    test_metrics: dict,
    hyperparameters: dict,
    save_dir: str,
    model_name: str = "lstm_claims_reserving"
) -> str:
    """
    Save a complete PyTorch model with all necessary metadata
    
    Args:
        model: trained PyTorch model
        scaler: fitted sklearn scaler
        train_dataset: the training dataset object
        test_metrics: dict of evaluation metrics (R², MAE, etc.)
        hyperparameters: dict of model hyperparameters
        save_dir: base directory where models will be saved (e.g., './models')
        model_name: base name for the saved files
    
    Returns:
        str: Full path to the saved model directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_save_dir = os.path.join(save_dir, f"{model_name}_{timestamp}")
    os.makedirs(full_save_dir, exist_ok=True)
    
    # 1. Save model weights (state_dict - MOST IMPORTANT)
    torch.save(model.state_dict(), os.path.join(full_save_dir, "model_weights.pt"))
    
    # 2. Save complete checkpoint (optional - for resuming training)
    checkpoint = {
        'epoch': hyperparameters.get('epochs', None),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,  # Add if you saved optimizer
        'train_loss': hyperparameters.get('final_train_loss', None),
        'test_loss': hyperparameters.get('final_test_loss', None),
    }
    torch.save(checkpoint, os.path.join(full_save_dir, "checkpoint.pt"))
    
    # 3. Save scaler (CRITICAL for predictions)
    with open(os.path.join(full_save_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    
    # 4. Save metadata as JSON
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'pytorch_version': torch.__version__,
        
        # Model architecture
        'architecture': {
            'model_class': model.__class__.__name__,  # e.g., 'ClaimReservingLSTM_v1'
            'input_size': hyperparameters.get('input_size', 1),
            'hidden_size': hyperparameters.get('hidden_size', 64),
            'num_layers': hyperparameters.get('num_layers', 2),
            'dropout': hyperparameters.get('dropout', 0.2),
            'output_size': hyperparameters.get('output_size', 2),  # remaining + ultimate
            'fc_dropout': hyperparameters.get('fc_dropout', None),  # For v2+
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
    
    with open(os.path.join(full_save_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 5. Save a human-readable summary
    with open(os.path.join(full_save_dir, "README.txt"), 'w') as f:
        f.write(f"LSTM Claims Reserving Model\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Saved: {timestamp}\n")
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Hidden Size: {hyperparameters.get('hidden_size')}\n")
        f.write(f"Num Layers: {hyperparameters.get('num_layers')}\n\n")
        f.write(f"Test Performance:\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value}\n")
    
    print(f"✅ Model saved to: {full_save_dir}")
    return full_save_dir


def list_saved_models(models_dir: str):
    """
    List all saved models with their metadata
    
    Args:
        models_dir: directory containing saved model folders
    
    Returns:
        list of dicts with model info
    """
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for folder in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, folder)
        metadata_path = os.path.join(folder_path, 'metadata.json')
        
        if os.path.isdir(folder_path) and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            models.append({
                'folder': folder,
                'path': folder_path,
                'timestamp': metadata.get('timestamp'),
                'model_class': metadata['architecture'].get('model_class', 'Unknown'),
                'input_size': metadata['architecture']['input_size'],
                'hidden_size': metadata['architecture']['hidden_size'],
                'num_layers': metadata['architecture']['num_layers'],
            })
    
    return sorted(models, key=lambda x: x['timestamp'], reverse=True)


def load_claims_model(model_dir: str, model_class, device='cpu'):
    """
    Load a complete PyTorch model with all metadata
    
    Args:
        model_dir: directory containing saved model files (full path)
        model_class: The model class (e.g., ClaimReservingLSTM_v1) - NOT an instance
        device: 'cpu' or 'cuda'
    
    Returns:
        tuple: (model, scaler, metadata)
    """
    # 1. Load metadata
    with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # 2. Recreate model architecture using provided class
    arch = metadata['architecture']
    
    # Build model kwargs dynamically (handles different versions)
    model_kwargs = {
        'input_size': arch['input_size'],
        'hidden_size': arch['hidden_size'],
        'num_layers': arch['num_layers'],
        'dropout': arch['dropout']
    }
    
    # Add fc_dropout if present (for v2+)
    if arch.get('fc_dropout') is not None:
        model_kwargs['fc_dropout'] = arch['fc_dropout']
    
    model = model_class(**model_kwargs)
    
    # 3. Load model weights
    model.load_state_dict(torch.load(os.path.join(model_dir, "model_weights.pt"), map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # 4. Load scaler
    with open(os.path.join(model_dir, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✅ Model loaded from: {model_dir}")
    print(f"   Model class: {arch.get('model_class', 'Unknown')}")
    print(f"   Architecture: {arch['hidden_size']}-hidden, {arch['num_layers']}-layer LSTM")
    print(f"   Input features: {arch['input_size']}")
    
    # Show performance metrics if available
    perf = metadata.get('performance', {})
    if perf:
        # Handle nested dict (metrics by period) or flat dict
        if isinstance(next(iter(perf.values()), None), dict):
            print(f"   Performance: Metrics saved by observation period")
        else:
            print(f"   Performance: R²={perf.get('r2_ultimate', 'N/A')}")
    
    return model, scaler, metadata

