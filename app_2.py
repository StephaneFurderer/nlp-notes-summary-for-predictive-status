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
from helpers.functions.claims_utils import load_data, _filter_by_
from helpers.functions.standardized_claims_transformer import StandardizedClaimsTransformer
from helpers.functions.standardized_claims_schema import StandardizationConfig

# Initialize the template
template = ClaimsAnalysisTemplate("Standardized Claims Analysis", is_demo=False)

# Set up page configuration
template.setup_page_config("Standardized Claims Analysis", "📊")

# Load data
@st.cache_data
def cached_load_data(report_date=None):
    """Cached wrapper for load_data function"""
    return load_data(report_date)

# Custom claim filter function for real data
def filter_claims_real_data(df_raw_txn, claim_filter):
    """Custom filter function for real data using _filter_by_"""
    return _filter_by_(df_raw_txn, 'clmNum', claim_filter)

# Load the raw data
with st.spinner("Loading data..."):
    df_raw_txn, closed_txn, open_txn, paid_txn, df_raw_final, closed_final, paid_final, open_final = cached_load_data(None)

st.success(f"✅ Loaded {len(df_raw_txn):,} transactions for {df_raw_txn['clmNum'].nunique():,} claims")

# Run the app using the template
template.run_app(df_raw_txn, claim_filter_func=filter_claims_real_data, show_all_transactions=True, df_raw_final=df_raw_final)
