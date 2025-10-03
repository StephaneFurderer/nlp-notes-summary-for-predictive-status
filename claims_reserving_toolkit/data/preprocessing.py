import numpy as np


def prepare_claim_sequences(claims_data, train_split=0.8):
    """
    Prepare data for LSTM claim forecasting
    
    This is a GENERIC function that expects claims_data to be a list of dicts
    with a standard structure. Data loading is platform-specific and should be
    done before calling this function.
    
    Args:
        claims_data: List of claim dicts, each containing:
            - 'claim_id': unique identifier
            - 'payments': list/array of payment amounts per period
            - 'reserves': list/array of reserve amounts per period (optional)
            - 'ultimate': float, total payments (known for closed claims)
            - 'settlement_period': int, last period with activity
        train_split: float, fraction of sequences to use for training
    
    Returns:
        tuple: (train_data, test_data) where each is a list of sequence dicts
    
    Example:
        >>> claims = [
        ...     {
        ...         'claim_id': 123,
        ...         'ultimate': 5000,
        ...         'payments': [0, 1000, 2000, 1500, 500],
        ...         'reserves': [8000, 5000, 3000, 1000, 0],
        ...         'settlement_period': 4
        ...     },
        ...     # ... more claims
        ... ]
        >>> train_data, test_data = prepare_claim_sequences(claims, train_split=0.8)
    """
    
    sequences = []
    
    for claim in claims_data:
        payments = np.array(claim['payments'])
        ultimate = claim['ultimate']
        settlement = claim['settlement_period']
        
        # Check if reserves are provided (for multi-feature models)
        has_reserves = 'reserves' in claim and claim['reserves'] is not None
        if has_reserves:
            reserves = np.array(claim['reserves'])
        
        # Create multiple training samples per claim
        # Observe first k periods, predict remainder
        # This is like evaluating the claim at different development points
        
        for observe_until in range(2, min(settlement + 1, len(payments))):
            # Input: payments (and optionally reserves) from period 1 to observe_until
            observed_payments = payments[:observe_until].copy()
            
            if has_reserves:
                observed_reserves = reserves[:observe_until].copy()
                # Stack into 2D array: [observe_until, 2]
                # Column 0: payment, Column 1: reserve
                X = np.column_stack([observed_payments, observed_reserves])
            else:
                # Single feature: just payments
                X = observed_payments.reshape(-1, 1)
            
            # Target: remaining payments + ultimate
            remaining_periods = payments[observe_until:].sum()
            
            sequences.append({
                'claim_id': claim['claim_id'],
                'observed_periods': observe_until,
                'input_sequence': X,  # Shape: [observe_until, n_features]
                'target_ultimate': ultimate,
                'target_remaining': remaining_periods,
                'paid_to_date': observed_payments.sum()
            })
    
    # Split into train/test
    np.random.shuffle(sequences)
    split_idx = int(len(sequences) * train_split)
    
    train_data = sequences[:split_idx]
    test_data = sequences[split_idx:]
    
    return train_data, test_data

