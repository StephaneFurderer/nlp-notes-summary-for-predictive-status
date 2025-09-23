import pandas as pd
import streamlit as st
from helpers.functions.claims_utils import load_data
from helpers.functions.load_cache_data import DataLoader, CacheManager as PeriodCacheManager
from helpers.functions.standardized_claims_transformer import StandardizedClaimsTransformer
from helpers.functions.standardized_claims_schema import StandardizationConfig

st.set_page_config(page_title="Claims Analysis", layout="wide")
st.title("Claims Analysis")

# Initialize modules
data_loader = DataLoader()
period_cache_manager = PeriodCacheManager()
transformer = StandardizedClaimsTransformer()

# Sidebar controls
st.sidebar.markdown("## Data Selection")
extraction_date = st.sidebar.selectbox(
    "Extraction Date",
    ["2025-09-21"],  # Add more dates as available
    help="Select the data extraction date to analyze"
)

# Load data using the proper modules
data = load_data(extraction_date=extraction_date)
df_raw_txn, closed_txn, open_txn, paid_txn, df_raw_final, closed_final, paid_final, open_final = data

# Check for cached periodized data
periods_df = period_cache_manager.load_cache(df_raw_txn, extraction_date)

if periods_df is None:
    st.sidebar.info("Computing periodized data...")
    # Create standardization config
    config = StandardizationConfig(period_length_days=30, max_periods=60)
    transformer.config = config
    
    # Transform to periods
    dataset = transformer.transform_claims_data(df_raw_txn)
    periods_df = dataset.to_dataframes()['dynamic_periods']
    
    # Save to cache
    period_cache_manager.save_cache(periods_df, df_raw_txn, extraction_date)
    st.sidebar.success("Periodized data cached")
else:
    st.sidebar.success("Loaded periodized data from cache")

# Data tables
with st.expander("All Transactions", expanded=False):
    st.subheader("All Transactions Data")
    st.dataframe(df_raw_txn, use_container_width=True)

with st.expander("All Final Claims", expanded=False):
    st.subheader("Final Status for All Claims")
    st.dataframe(df_raw_final, use_container_width=True)

with st.expander("All Periods", expanded=False):
    st.subheader("Standardized Periods Data")
    st.dataframe(periods_df, use_container_width=True)


