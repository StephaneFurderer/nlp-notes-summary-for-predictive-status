"""
Standardized Claims Visualization App

This app shows the transformation from raw transaction data to standardized 30-day periods
following the micro-level reserving framework.
"""

import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np

# Import our template and standardization functions
from helpers.functions.app_2_template import ClaimsAnalysisTemplate
from helpers.functions.claims_utils import _filter_by_
from helpers.functions.standardized_claims_transformer import StandardizedClaimsTransformer
from helpers.functions.standardized_claims_schema import StandardizationConfig
from helpers.functions.load_cache_data import DataLoader

# Initialize the template
template = ClaimsAnalysisTemplate("Standardized Claims Analysis", is_demo=False)

# Set up page configuration
template.setup_page_config("Standardized Claims Analysis", "📊")

# Load data using new organized structure
@st.cache_data
def cached_load_data_from_organized(extraction_date):
    """Load data from organized structure by extraction date"""
    data_loader = DataLoader()
    
    # Load claims data
    df_raw_txn = data_loader.load_claims_data(extraction_date=extraction_date)
    
    if df_raw_txn is None:
        return None, None, None, None, None, None, None, None
    
    # For now, we'll use the same dataframe for all outputs
    # In a real scenario, you might want to process this differently
    df_raw_final = df_raw_txn.groupby('clmNum').last().reset_index()
    
    # Filter by status
    closed_txn = df_raw_txn[df_raw_txn['clmStatus'].isin(['CLOSED', 'DENIED', 'PAID'])]
    open_txn = df_raw_txn[df_raw_txn['clmStatus'] == 'OPEN']
    paid_txn = df_raw_txn[df_raw_txn['clmStatus'] == 'PAID']
    
    closed_final = df_raw_final[df_raw_final['clmStatus'].isin(['CLOSED', 'DENIED', 'PAID'])]
    open_final = df_raw_final[df_raw_final['clmStatus'] == 'OPEN']
    paid_final = df_raw_final[df_raw_final['clmStatus'] == 'PAID']
    
    return df_raw_txn, closed_txn, open_txn, paid_txn, df_raw_final, closed_final, paid_final, open_final

# Custom claim filter function for real data
def filter_claims_real_data(df_raw_txn, claim_filter):
    """Custom filter function for real data using _filter_by_"""
    return _filter_by_(df_raw_txn, 'clmNum', claim_filter)

# Check for available data versions
data_loader = DataLoader()
available_versions = data_loader.get_available_data_versions()

if not available_versions:
    st.error("❌ No organized data found!")
    st.info("""
    **To use this app, you need to organize your data first:**
    
    1. Create a folder structure like `_data/2024-01-15/`
    2. Place your `clm_with_amt.csv` file in that folder
    3. Optionally add `notes_summary.csv` and `policy_info.csv`
    
    **Example structure:**
    ```
    _data/
    └── 2024-01-15/
        ├── clm_with_amt.csv
        ├── notes_summary.csv (optional)
        └── policy_info.csv (optional)
    ```
    """)
    st.stop()

# Let user select extraction date
st.subheader("📅 Select Data Version")
extraction_dates = [v['extraction_date'] for v in available_versions]
selected_extraction_date = st.selectbox(
    "Choose the data extraction date:",
    options=extraction_dates,
    index=0,
    help="Select which version of your data to analyze"
)

# Show info about selected data
selected_version = next(v for v in available_versions if v['extraction_date'] == selected_extraction_date)
st.info(f"📊 **Selected Data:** {selected_extraction_date} - Files: {', '.join(selected_version['file_types'])}")

# Load the raw data
with st.spinner(f"Loading data from {selected_extraction_date}..."):
    df_raw_txn, closed_txn, open_txn, paid_txn, df_raw_final, closed_final, paid_final, open_final = cached_load_data_from_organized(selected_extraction_date)

if df_raw_txn is None:
    st.error(f"❌ Could not load data from {selected_extraction_date}")
    st.stop()

st.success(f"✅ Loaded {len(df_raw_txn):,} transactions for {df_raw_txn['clmNum'].nunique():,} claims")

# Run the app using the template
template.run_app(df_raw_txn, claim_filter_func=filter_claims_real_data, show_all_transactions=True, df_raw_final=df_raw_final)
