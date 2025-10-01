import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

"""
LSTM FOR INDIVIDUAL CLAIM DEVELOPMENT FORECASTING
==================================================

Problem: Predict future claim payments given historical payment patterns
Structure: Each claim is ONE row, columns are payment development periods

Example Data Structure:
Claim_ID | Period_1 | Period_2 | Period_3 | Period_4 | ... | Period_N | Ultimate
---------|----------|----------|----------|----------|-----|----------|----------
   001   |   5000   |   2000   |   1500   |    500   | ... |    0     |  15000
   002   |  12000   |   8000   |   3000   |   1000   | ... |    0     |  28000
   003   |   3000   |   1000   |    500   |      ?   | ... |    ?     |    ?

Key Difference from Time Series:
- Each ROW is a claim (not a time window)
- COLUMNS are development periods (inherently sequential)
- Variable length: Claims settle at different speeds
- We want to predict remaining payments AND ultimate loss
"""

np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# PART 1: GENERATE REALISTIC CLAIM DEVELOPMENT DATA
# ============================================================================

def generate_claim_development_data(n_claims=500, max_periods=12):
    """
    Generate synthetic individual claim development patterns
    
    Mimics real insurance claims:
    - Large initial payment (reported loss)
    - Declining incremental payments (development)
    - Some claims settle early, others develop slowly
    - Ultimate loss varies by claim severity
    """
    claims_data = []
    
    for claim_id in range(n_claims):
        # Initial severity (lognormal distribution)
        severity = np.random.lognormal(mean=8.5, sigma=1.2)
        
        # Settlement speed (some claims close fast, others slow)
        speed = np.random.uniform(0.3, 0.9)
        
        # Generate payment pattern
        payments = []
        cumulative = 0
        
        for period in range(max_periods):
            if period == 0:
                # Initial payment (40-70% of ultimate)
                payment = severity * np.random.uniform(0.4, 0.7)
            else:
                # Declining incremental payments
                remaining = severity - cumulative
                if remaining > 100:  # Still open
                    payment = remaining * speed * np.exp(-0.3 * period) * np.random.uniform(0.5, 1.5)
                    payment = min(payment, remaining)  # Can't exceed remaining
                else:
                    payment = remaining  # Close out
            
            payment = max(0, payment)  # No negative payments
            cumulative += payment
            payments.append(payment)
            
            # Claim settles when remaining is negligible
            if (severity - cumulative) < 100:
                break
        
        # Pad with zeros if settled early
        while len(payments) < max_periods:
            payments.append(0)
        
        claims_data.append({
            'claim_id': claim_id,
            'severity': severity,
            'ultimate': cumulative,
            'payments': payments,
            'settlement_period': len([p for p in payments if p > 0])
        })
    
    return claims_data

# Generate claims
claims = generate_claim_development_data(n_claims=500, max_periods=12)

# Convert to DataFrame for easier viewing
df_claims = pd.DataFrame([
    {
        'claim_id': c['claim_id'],
        'ultimate': c['ultimate'],
        'settlement_period': c['settlement_period'],
        **{f'period_{i+1}': c['payments'][i] for i in range(len(c['payments']))}
    }
    for c in claims
])

print("="*80)
print("SAMPLE CLAIM DEVELOPMENT DATA")
print("="*80)
print(df_claims.head(10))
print(f"\nTotal Claims: {len(df_claims)}")
print(f"Average Ultimate Loss: ${df_claims['ultimate'].mean():,.2f}")
print(f"Average Settlement Period: {df_claims['settlement_period'].mean():.1f} periods")

# Visualize sample claims
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ax in enumerate(axes.flat):
    claim = claims[idx]
    periods = range(1, len(claim['payments']) + 1)
    
    # Plot incremental payments
    ax.bar(periods, claim['payments'], alpha=0.7, color='steelblue', label='Incremental')
    
    # Plot cumulative
    cumulative = np.cumsum(claim['payments'])
    ax.plot(periods, cumulative, 'r-', linewidth=2, marker='o', label='Cumulative')
    ax.axhline(y=claim['ultimate'], color='g', linestyle='--', linewidth=2, label='Ultimate')
    
    ax.set_xlabel('Development Period', fontsize=11)
    ax.set_ylabel('Payment Amount ($)', fontsize=11)
    ax.set_title(f"Claim {claim['claim_id']}: Ultimate = ${claim['ultimate']:,.0f}", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sample_claim_developments.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PART 2: DATA PREPARATION FOR LSTM (THE KEY DIFFERENCE)
# ============================================================================

def prepare_claim_sequences(claims_data, train_split=0.8):
    """
    Prepare data for LSTM claim forecasting
    
    KEY STRUCTURE:
    - Each claim is ONE sequence
    - We observe first N periods, predict remaining periods + ultimate
    - Training: Use various "as-of" dates (partial observations)
    """
    
    sequences = []
    
    for claim in claims_data:
        payments = np.array(claim['payments'])
        ultimate = claim['ultimate']
        
        # Create multiple training samples per claim
        # Observe first k periods, predict remainder
        # This is like evaluating the claim at different development points
        
        settlement = claim['settlement_period']
        
        for observe_until in range(2, min(settlement + 1, len(payments))):
            # Input: payments from period 1 to observe_until
            X = payments[:observe_until].copy()
            
            # Target: remaining payments + ultimate
            remaining_periods = payments[observe_until:].sum()
            
            sequences.append({
                'claim_id': claim['claim_id'],
                'observed_periods': observe_until,
                'input_sequence': X,
                'target_ultimate': ultimate,
                'target_remaining': remaining_periods,
                'paid_to_date': X.sum()
            })
    
    # Split into train/test
    np.random.shuffle(sequences)
    split_idx = int(len(sequences) * train_split)
    train_data = sequences[:split_idx]
    test_data = sequences[split_idx:]
    
    return train_data, test_data

train_data, test_data = prepare_claim_sequences(claims, train_split=0.8)

print("\n" + "="*80)
print("DATA PREPARATION - THE NEW STRUCTURE")
print("="*80)
print(f"Total training sequences: {len(train_data)}")
print(f"Total test sequences: {len(test_data)}")
print("\nSample Training Record:")
sample = train_data[0]
print(f"  Claim ID: {sample['claim_id']}")
print(f"  Observed until period: {sample['observed_periods']}")
print(f"  Input sequence: {sample['input_sequence']}")
print(f"  Paid to date: ${sample['paid_to_date']:,.2f}")
print(f"  Target remaining: ${sample['target_remaining']:,.2f}")
print(f"  Target ultimate: ${sample['target_ultimate']:,.2f}")

print("\n" + "="*80)
print("KEY INSIGHT: VARIABLE LENGTH SEQUENCES")
print("="*80)
print("""
Unlike time series forecasting where all samples have the same seq_length:
- Claim A observed for 3 periods: [5000, 2000, 1500]
- Claim B observed for 5 periods: [12000, 8000, 3000, 1000, 500]
- Claim C observed for 2 periods: [3000, 1000]

All are valid training examples at different stages of development!
We need to handle this with PADDING and MASKING.
""")

# ============================================================================
# PART 3: PYTORCH DATASET FOR VARIABLE LENGTH SEQUENCES
# ============================================================================

# Part 3 — PyTorch dataset for variable-length claim sequences (what it does and why)
# Part 3 sets up the data pipeline that lets an LSTM learn from claim-level payment histories where each claim has a different number of observed periods. It solves three practical problems: scaling, variable sequence lengths, and batching.
# What lives here:
# ClaimDataset: wraps your per-claim records and handles scaling and tensor conversion
# collate_fn: batches variable-length sequences by padding them to the same length at runtime
# DataLoaders: efficient minibatching with the custom collate_fn

# --------------------------------------------------------------
# 1) Dataset: claim-level sequence, with consistent scaling
# Each item is one claim observed up to some development point. It returns the scaled input sequence, targets, and the sequence length.
# Why scale here: Actuarial payment sizes vary by orders of magnitude across claims. Fitting a scaler on the training set once and reusing it avoids leakage and stabilizes optimization.
# Outputs:
# X: shape [T, 1] (T periods observed), scaled
# y_remaining: scalar target for the sum of future payments
# y_ultimate: scalar target for ultimate loss
# seq_length: actual observed length T (needed to avoid learning from padding)
# --------------------------------------------------------------
# 2) Collate function: pad sequences and preserve true lengths
# Batches variable-length claims by padding sequences to the same max length in the batch and returns their true lengths.
# Padding value is 0 (post-padding). Crucially, you also return lengths so the model can ignore the padded tail (via packed sequences in Part 4).
# Why actuaries should care: different settlement speeds create different T per claim. This design faithfully represents “as-of” observation while allowing efficient GPU batching.
# --------------------------------------------------------------
# 3) DataLoaders: efficient minibatching with the custom collate_fn
# Batches variable-length claims by padding sequences to the same max length in the batch and returns their true lengths.
# Reusing the train scaler for test maintains proper out-of-sample evaluation (no peeking at test).
# Shuffling the training set is standard to de-correlate batches and improve convergence.
# --------------------------------------------------------------
# --------------------------------------------------------------
# Why this matters for reserving
# Respecting variable development: A claim observed for 3 periods and another for 7 periods are both valid training examples; the model learns from both without forcing a fixed lookback.
# No triangle aggregation needed: The pipeline works from raw per-claim sequences.
# Clean separation of concerns:
# Dataset: per-claim transformation and scaling
# Collate: batch-time padding/masking info
# Model (Part 4): uses packed sequences and lengths so the LSTM “sees” only real observations, not padding
# --------------------------------------------------------------
# Common pitfalls this avoids
# Data leakage: fitting scalers on all data (train+test) or on full future sequences. Here, train-only scaler is reused downstream.
# Padding bias: averaging over padded zeros without masking. Here, lengths enables packing in the LSTM so padded tail is ignored.
# Length confounding: You don’t truncate to a fixed window; the model sees the true observed horizon per claim.
# --------------------------------------------------------------

class ClaimDataset(torch.utils.data.Dataset):
    """Custom Dataset for variable-length claim sequences"""
    
    def __init__(self, data, scaler=None):
        self.data = data
        
        # Fit scaler on training data
        if scaler is None:
            all_payments = np.concatenate([d['input_sequence'] for d in data])
            self.scaler = StandardScaler()
            self.scaler.fit(all_payments.reshape(-1, 1))
        else:
            self.scaler = scaler
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Normalize input sequence
        X = self.scaler.transform(sample['input_sequence'].reshape(-1, 1))
        X = torch.FloatTensor(X)
        
        # Targets (we'll predict both remaining and ultimate)
        y_remaining = torch.FloatTensor([sample['target_remaining']])
        y_ultimate = torch.FloatTensor([sample['target_ultimate']])
        
        # Also return the length for padding/packing
        seq_length = torch.LongTensor([len(sample['input_sequence'])])
        
        return X, y_remaining, y_ultimate, seq_length

def collate_fn(batch):
    """Custom collate function to handle variable lengths"""
    X_list, y_remaining_list, y_ultimate_list, lengths = zip(*batch)
    
    # Pad sequences to same length
    X_padded = pad_sequence(X_list, batch_first=True, padding_value=0)
    
    # Stack targets
    y_remaining = torch.stack(y_remaining_list)
    y_ultimate = torch.stack(y_ultimate_list)
    lengths = torch.stack(lengths).squeeze()
    
    return X_padded, y_remaining, y_ultimate, lengths

# Create datasets
train_dataset = ClaimDataset(train_data)
test_dataset = ClaimDataset(test_data, scaler=train_dataset.scaler)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=False,
    collate_fn=collate_fn
)

# ============================================================================
# PART 4: LSTM MODEL FOR CLAIM RESERVING
# ============================================================================

class ClaimReservingLSTM(nn.Module):
    """
    LSTM for predicting remaining claim payments and ultimate loss
    
    Key features:
    - Handles variable-length sequences
    - Outputs both remaining payments and ultimate loss
    - Uses packed sequences for efficiency
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(ClaimReservingLSTM, self).__init__()
        
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
        x: (batch, max_seq_len, 1) - padded sequences
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

# Initialize model
model = ClaimReservingLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
print("\n" + "="*80)
print("MODEL ARCHITECTURE")
print("="*80)
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# PART 5: TRAINING WITH DUAL OBJECTIVES
# ============================================================================
# Quick cheat sheet:
# model.train(): enables training behaviors (e.g., dropout ON, batch norm updates). Forward passes track gradients by default (unless in torch.no_grad()).
# model.eval(): disables training behaviors (dropout OFF, batch norm uses running stats). Use with torch.no_grad() for inference to save memory/compute.
# model(x): just a forward pass. It doesn’t update weights by itself.
# Weight updates require the three-step loop: zero_grad → forward → loss → backward → optimizer.step().
# --------------------------------------------------------------
# What happens each step:
# Pull a minibatch from train_loader (this loader comes from the training dataset).
# X_batch are padded sequences; lengths_batch are true lengths (so the model can ignore padding).
# Forward pass: model outputs two predictions per claim: remaining and ultimate.
# Compute two MSE losses and add them.
# Backprop (loss.backward()), then update weights (optimizer.step()).
# --------------------------------------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
train_losses = []
test_losses = []

print("\n" + "="*80)
print("TRAINING THE MODEL")
print("="*80)

for epoch in range(num_epochs):
    # Training
    model.train()
    epoch_train_loss = 0
    
    for X_batch, y_remaining_batch, y_ultimate_batch, lengths_batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        pred_remaining, pred_ultimate = model(X_batch, lengths_batch)
        
        # Combined loss: predict both remaining and ultimate
        loss_remaining = criterion(pred_remaining, y_remaining_batch)
        loss_ultimate = criterion(pred_ultimate, y_ultimate_batch)
        loss = loss_remaining + loss_ultimate
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
    
    # Evaluation
    model.eval()
    epoch_test_loss = 0
    
    with torch.no_grad():
        for X_batch, y_remaining_batch, y_ultimate_batch, lengths_batch in test_loader:
            pred_remaining, pred_ultimate = model(X_batch, lengths_batch)
            loss_remaining = criterion(pred_remaining, y_remaining_batch)
            loss_ultimate = criterion(pred_ultimate, y_ultimate_batch)
            loss = loss_remaining + loss_ultimate
            epoch_test_loss += loss.item()
    
    train_losses.append(epoch_train_loss / len(train_loader))
    test_losses.append(epoch_test_loss / len(test_loader))
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}")

# ============================================================================
# PART 6: EVALUATION AND PREDICTIONS
# ============================================================================

model.eval()
all_predictions = []

with torch.no_grad():
    for X_batch, y_remaining_batch, y_ultimate_batch, lengths_batch in test_loader:
        pred_remaining, pred_ultimate = model(X_batch, lengths_batch)
        
        for i in range(len(X_batch)):
            all_predictions.append({
                'actual_remaining': y_remaining_batch[i].item(),
                'pred_remaining': pred_remaining[i].item(),
                'actual_ultimate': y_ultimate_batch[i].item(),
                'pred_ultimate': pred_ultimate[i].item()
            })

df_predictions = pd.DataFrame(all_predictions)

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_remaining = mean_absolute_error(df_predictions['actual_remaining'], df_predictions['pred_remaining'])
mae_ultimate = mean_absolute_error(df_predictions['actual_ultimate'], df_predictions['pred_ultimate'])
r2_remaining = r2_score(df_predictions['actual_remaining'], df_predictions['pred_remaining'])
r2_ultimate = r2_score(df_predictions['actual_ultimate'], df_predictions['pred_ultimate'])

print("\n" + "="*80)
print("MODEL PERFORMANCE")
print("="*80)
print(f"Remaining Payments Prediction:")
print(f"  MAE: ${mae_remaining:,.2f}")
print(f"  R²:  {r2_remaining:.4f}")
print(f"\nUltimate Loss Prediction:")
print(f"  MAE: ${mae_ultimate:,.2f}")
print(f"  R²:  {r2_ultimate:.4f}")

# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training loss
axes[0, 0].plot(train_losses, label='Train', linewidth=2)
axes[0, 0].plot(test_losses, label='Test', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training History', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Remaining payments prediction
axes[0, 1].scatter(df_predictions['actual_remaining'], 
                   df_predictions['pred_remaining'], 
                   alpha=0.5, s=20)
max_val = max(df_predictions['actual_remaining'].max(), df_predictions['pred_remaining'].max())
axes[0, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
axes[0, 1].set_xlabel('Actual Remaining ($)')
axes[0, 1].set_ylabel('Predicted Remaining ($)')
axes[0, 1].set_title(f'Remaining Payments (R²={r2_remaining:.3f})', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Ultimate loss prediction
axes[1, 0].scatter(df_predictions['actual_ultimate'], 
                   df_predictions['pred_ultimate'], 
                   alpha=0.5, s=20, color='green')
max_val = max(df_predictions['actual_ultimate'].max(), df_predictions['pred_ultimate'].max())
axes[1, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
axes[1, 0].set_xlabel('Actual Ultimate ($)')
axes[1, 0].set_ylabel('Predicted Ultimate ($)')
axes[1, 0].set_title(f'Ultimate Loss (R²={r2_ultimate:.3f})', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Error distribution
errors_ultimate = df_predictions['actual_ultimate'] - df_predictions['pred_ultimate']
axes[1, 1].hist(errors_ultimate, bins=50, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Prediction Error ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Ultimate Loss Error Distribution', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('claim_reserving_results.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PART 8: PRACTICAL APPLICATION - RESERVE ESTIMATION
# ============================================================================

print("\n" + "="*80)
print("PRACTICAL APPLICATION: PORTFOLIO RESERVE ESTIMATION")
print("="*80)

# Simulate current portfolio (claims at various development stages)
current_claims = claims[:50]  # Use first 50 claims as "current" portfolio

total_paid = 0
total_reserved_actual = 0
total_reserved_predicted = 0

model.eval()
with torch.no_grad():
    for claim in current_claims:
        # Observe claim up to a random development point
        observe_until = np.random.randint(2, min(claim['settlement_period'], 8))
        
        observed_payments = claim['payments'][:observe_until]
        paid_to_date = sum(observed_payments)
        actual_remaining = claim['ultimate'] - paid_to_date
        
        # Prepare input
        X = train_dataset.scaler.transform(np.array(observed_payments).reshape(-1, 1))
        X = torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension
        length = torch.LongTensor([observe_until])
        
        # Predict
        pred_remaining, pred_ultimate = model(X, length)
        
        total_paid += paid_to_date
        total_reserved_actual += actual_remaining
        total_reserved_predicted += pred_remaining.item()

print(f"\nPortfolio Summary (50 claims):")
print(f"  Total Paid to Date:        ${total_paid:,.2f}")
print(f"  Actual Reserve Needed:     ${total_reserved_actual:,.2f}")
print(f"  LSTM Predicted Reserve:    ${total_reserved_predicted:,.2f}")
print(f"  Prediction Error:          ${abs(total_reserved_actual - total_reserved_predicted):,.2f}")
print(f"  Error %:                   {100 * abs(total_reserved_actual - total_reserved_predicted) / total_reserved_actual:.2f}%")

print("\n" + "="*80)
print("KEY ADVANTAGES FOR ACTUARIES")
print("="*80)
print("""
1. CLAIM-LEVEL PREDICTIONS: Unlike triangle methods, this predicts individual claims
2. VARIABLE DEVELOPMENT: Handles claims that settle at different speeds
3. DYNAMIC UPDATES: Re-predict reserves as new payments are observed
4. FEATURE RICH: Can add claim characteristics (injury type, jurisdiction, adjuster)
5. NO TRIANGLE NEEDED: Works with raw transaction data
6. PARTIAL INFORMATION: Predicts from any development stage

NEXT STEPS FOR PRODUCTION:
- Add claim features (injury severity, jurisdiction, policy type)
- Use attention mechanism for interpretability
- Implement uncertainty quantification (prediction intervals)
- Compare to chain-ladder and Bornhuetter-Ferguson methods
- Handle reopened claims and late reported claims
""")