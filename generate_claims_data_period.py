"""
Generate insurance claims data with per-period payment probabilities
P(N_t = 0) = 70%, P(N_t = 1) = 10%, etc. for each period t
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

def generate_claims_data_period(n_claims=1000, n_periods=60, seed=42):
    """
    Generate insurance claims data with per-period payment probabilities:
    - P(N_t = 0) = 70% for each period t
    - P(N_t = 1) = 10% for each period t
    - P(N_t = 2) = 8% for each period t
    - etc.
    """
    print(f"Generating {n_claims} claims over {n_periods} periods...")
    np.random.seed(seed)
    
    # Define per-period frequency probabilities
    # P(N_t = 0) = 70%, P(N_t = 1) = 10%, P(N_t = 2) = 8%, P(N_t = 3) = 6%, P(N_t = 4) = 4%, P(N_t = 5+) = 2%
    period_frequency_probs = [0.70, 0.10, 0.08, 0.06, 0.04, 0.02]  # Sum = 1.0
    
    # Severity distribution: Log-normal with high skewness (heavy tail)
    # Parameters for log-normal: mean=8, std=1.5 (creates highly skewed distribution)
    severity_mu = 8.0
    severity_sigma = 1.5
    
    claims_data = []
    
    for claim_id in range(n_claims):
        if claim_id % 100 == 0:
            print(f"Processing claim {claim_id}...")
        
        # Initialize payment vector
        payments = np.zeros(n_periods)
        
        # For each period, determine if there's a payment
        for period in range(n_periods):
            # Generate number of payments for this period
            n_payments_this_period = np.random.choice(
                range(len(period_frequency_probs)), 
                p=period_frequency_probs
            )
            
            if n_payments_this_period > 0:
                # Generate payment amount(s) for this period
                payment_amounts = np.random.lognormal(
                    mean=severity_mu, 
                    sigma=severity_sigma, 
                    size=n_payments_this_period
                )
                
                # Sum all payments in this period
                payments[period] = np.sum(payment_amounts)
        
        # Calculate cumulative payments
        cumulative_payments = np.cumsum(payments)
        
        # Calculate statistics
        total_payments = cumulative_payments[-1]
        n_payments = np.sum(payments > 0)
        first_payment_period = np.argmax(payments > 0) if np.any(payments > 0) else n_periods
        last_payment_period = np.max(np.where(payments > 0)[0]) if np.any(payments > 0) else 0
        
        claims_data.append({
            'claim_id': claim_id,
            'periods': list(range(n_periods)),
            'payments': payments.tolist(),
            'cumulative_payments': cumulative_payments.tolist(),
            'total_payments': total_payments,
            'n_payments': n_payments,
            'first_payment_period': first_payment_period,
            'last_payment_period': last_payment_period
        })
    
    return pd.DataFrame(claims_data)

def main():
    print("🏥 Insurance Claims Data Generator (Per-Period Probabilities)")
    print("=" * 60)
    
    # Generate data
    claims_df = generate_claims_data_period(n_claims=1000, n_periods=60, seed=42)
    
    print(f"\n✅ Generated {len(claims_df)} claims")
    print(f"📊 Claims with payments: {(claims_df['total_payments'] > 0).sum()}")
    print(f"💰 Average total payment: ${claims_df['total_payments'].mean():,.0f}")
    print(f"📈 Max total payment: ${claims_df['total_payments'].max():,.0f}")
    
    # Save to CSV
    csv_filename = f"claims_data_period_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    claims_df.to_csv(csv_filename, index=False)
    print(f"\n💾 Saved to: {csv_filename}")
    
    # Also save a simplified version for quick loading
    # Flatten the cumulative payments into separate columns
    cumulative_cols = [f'cumulative_period_{i}' for i in range(60)]
    payments_cols = [f'payment_period_{i}' for i in range(60)]
    
    # Create flattened dataframe
    flattened_data = []
    for _, row in claims_df.iterrows():
        flat_row = {
            'claim_id': row['claim_id'],
            'total_payments': row['total_payments'],
            'n_payments': row['n_payments'],
            'first_payment_period': row['first_payment_period'],
            'last_payment_period': row['last_payment_period']
        }
        
        # Add cumulative payments
        for i, cum_payment in enumerate(row['cumulative_payments']):
            flat_row[f'cumulative_period_{i}'] = cum_payment
            
        # Add individual payments
        for i, payment in enumerate(row['payments']):
            flat_row[f'payment_period_{i}'] = payment
            
        flattened_data.append(flat_row)
    
    flattened_df = pd.DataFrame(flattened_data)
    
    # Save flattened version
    flattened_filename = f"claims_data_period_flattened_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    flattened_df.to_csv(flattened_filename, index=False)
    print(f"💾 Saved flattened version to: {flattened_filename}")
    
    # Save metadata
    metadata = {
        'n_claims': 1000,
        'n_periods': 60,
        'period_frequency_probs': [0.70, 0.10, 0.08, 0.06, 0.04, 0.02],
        'severity_mu': 8.0,
        'severity_sigma': 1.5,
        'generated_at': datetime.now().isoformat(),
        'seed': 42,
        'description': 'Per-period payment probabilities: P(N_t=0)=70%, P(N_t=1)=10%, etc.'
    }
    
    metadata_filename = f"claims_metadata_period_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata to: {metadata_filename}")
    
    # Display sample statistics
    print(f"\n📈 Sample Statistics:")
    print(f"• Claims with no payments: {(claims_df['n_payments'] == 0).sum()} ({(claims_df['n_payments'] == 0).mean()*100:.1f}%)")
    print(f"• Claims with 1-10 payments: {((claims_df['n_payments'] >= 1) & (claims_df['n_payments'] <= 10)).sum()} ({((claims_df['n_payments'] >= 1) & (claims_df['n_payments'] <= 10)).mean()*100:.1f}%)")
    print(f"• Claims with 11-30 payments: {((claims_df['n_payments'] >= 11) & (claims_df['n_payments'] <= 30)).sum()} ({((claims_df['n_payments'] >= 11) & (claims_df['n_payments'] <= 30)).mean()*100:.1f}%)")
    print(f"• Claims with 31+ payments: {(claims_df['n_payments'] >= 31).sum()} ({(claims_df['n_payments'] >= 31).mean()*100:.1f}%)")
    print(f"• Average payments per claim: {claims_df['n_payments'].mean():.2f}")
    
    paid_claims = claims_df[claims_df['total_payments'] > 0]
    print(f"• Mean payment amount: ${paid_claims['total_payments'].mean():,.0f}")
    print(f"• Median payment amount: ${paid_claims['total_payments'].median():,.0f}")
    print(f"• 95th percentile payment: ${paid_claims['total_payments'].quantile(0.95):,.0f}")
    
    # Per-period statistics
    print(f"\n📊 Per-Period Statistics:")
    all_payments = []
    for _, row in claims_df.iterrows():
        all_payments.extend(row['payments'])
    
    all_payments = np.array(all_payments)
    periods_with_payments = np.sum(all_payments > 0)
    total_periods = len(all_payments)
    
    print(f"• Total periods across all claims: {total_periods:,}")
    print(f"• Periods with payments: {periods_with_payments:,} ({periods_with_payments/total_periods*100:.1f}%)")
    print(f"• Periods without payments: {total_periods - periods_with_payments:,} ({(total_periods - periods_with_payments)/total_periods*100:.1f}%)")
    print(f"• Average payment per period (when > 0): ${all_payments[all_payments > 0].mean():,.0f}")
    
    print(f"\n✅ Data generation complete!")
    print(f"📁 Files created:")
    print(f"   • {csv_filename} (full data with lists)")
    print(f"   • {flattened_filename} (flattened for easy loading)")
    print(f"   • {metadata_filename} (generation parameters)")

if __name__ == "__main__":
    main()
