"""
Standardized Claims Visualization App - DEMO VERSION

This demo app shows the transformation from raw transaction data to standardized 30-day periods
following the micro-level reserving framework. Uses synthetic data for demonstration.
"""

import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np

# Import our template and standardization functions
from helpers.functions.app_2_template import ClaimsAnalysisTemplate
from helpers.functions.standardized_claims_transformer import StandardizedClaimsTransformer
from helpers.functions.standardized_claims_schema import StandardizationConfig

# Create synthetic data
@st.cache_data
def create_synthetic_data():
    """Create synthetic claims data for demonstration"""
    np.random.seed(42)
    
    # Create sample claims
    claims_data = []
    claim_numbers = ['CLM001', 'CLM002', 'CLM003', 'CLM004', 'CLM005']
    claim_causes = ['AUTO', 'PROPERTY', 'LIABILITY', 'WORKERS_COMP', 'GENERAL']
    
    for i, claim_num in enumerate(claim_numbers):
        claim_cause = claim_causes[i % len(claim_causes)]
        base_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
        
        # Create 3-8 transactions per claim
        num_transactions = np.random.randint(3, 9)
        
        for txn_num in range(num_transactions):
            # Transaction date - spread over time (ensure at least 30+ days span)
            txn_date = base_date + timedelta(days=txn_num * np.random.randint(20, 45))
            
            # Random amounts
            paid = np.random.exponential(1000) if np.random.random() > 0.6 else 0
            expense = np.random.exponential(200) if np.random.random() > 0.7 else 0
            recovery = np.random.exponential(500) if np.random.random() > 0.9 else 0
            reserve = np.random.exponential(2000) if txn_num == 0 else 0
            
            claims_data.append({
                'clmNum': claim_num,
                'clmCause': claim_cause,
                'booknum': f'BOOK{i+1:03d}',
                'cidpol': f'POL{i+1:06d}',
                'datetxn': txn_date,
                'clmStatus': 'OPEN' if txn_num < num_transactions - 1 else np.random.choice(['PAID', 'DENIED', 'CLOSED']),
                'dateCompleted': None if txn_num < num_transactions - 1 else txn_date + timedelta(days=30),
                'dateReopened': None,
                'paid': paid,
                'expense': expense,
                'recovery': recovery,
                'reserve': reserve,
                'paymentType': np.random.choice(['Benefit', 'Expense', 'Recovery', 'Reserve']),
                'processCategory': 'CLAIM',
                'amt': paid + expense + recovery + reserve,
                'policy_has_open_claims': True,
                'policy_has_reopen_claims': False
            })
    
    df = pd.DataFrame(claims_data)
    
    # Sort by claim and date
    df = df.sort_values(['clmNum', 'datetxn'])
    
    # Calculate cumulative amounts
    for claim_num in df['clmNum'].unique():
        claim_mask = df['clmNum'] == claim_num
        claim_data = df[claim_mask].copy()
        
        df.loc[claim_mask, 'paid_cumsum'] = claim_data['paid'].cumsum()
        df.loc[claim_mask, 'expense_cumsum'] = claim_data['expense'].cumsum()
        df.loc[claim_mask, 'recovery_cumsum'] = claim_data['recovery'].cumsum()
        df.loc[claim_mask, 'reserve_cumsum'] = claim_data['reserve'].cumsum()
        df.loc[claim_mask, 'incurred_cumsum'] = (claim_data['paid'] + claim_data['expense'] + 
                                                claim_data['recovery'] + claim_data['reserve']).cumsum()
    
    return df

# Initialize the template
template = ClaimsAnalysisTemplate("Standardized Claims Analysis - DEMO", is_demo=True)

# Set up page configuration
template.setup_page_config("Standardized Claims Analysis - Demo", "📊")

# Load synthetic data
with st.spinner("Creating synthetic data..."):
    df_raw_txn = create_synthetic_data()

# Create final claim status data (equivalent to get_final_data from claims_utils.py)
def get_final_data(df):
    """Helper function to get final transaction data for each claim"""
    return df.groupby('clmNum').apply(lambda x: x.loc[x['datetxn'].idxmax()]).reset_index(drop=True)

df_raw_final = get_final_data(df_raw_txn)

st.success(f"✅ Created synthetic data: {len(df_raw_txn):,} transactions for {df_raw_txn['clmNum'].nunique():,} claims")

# Run the app using the template
template.run_app(df_raw_txn, show_all_transactions=True, df_raw_final=df_raw_final)
