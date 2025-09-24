import pandas as pd
import streamlit as st
from helpers.functions.claims_utils import read_transformed_claims_data_from_parquet

from helpers.functions.standardized_claims_transformer import StandardizedClaimsTransformer
from helpers.functions.standardized_claims_schema import StandardizationConfig

st.set_page_config(page_title="Claims Analysis", layout="wide")
st.title("Claims Analysis")

# Initialize modules
# data_loader = DataLoader()
# period_cache_manager = PeriodCacheManager()
# transformer = StandardizedClaimsTransformer()


# Get available dates
available_versions = get_available_data_versions()
available_dates = [v['extraction_date'] for v in available_versions]

# Sidebar controls
st.sidebar.markdown("## Data Selection")
extraction_date = st.sidebar.selectbox(
    "Extraction Date",
    available_dates,
    help="Select the data extraction date to analyze"
)

df_raw_txn, closed_txn, open_txn, paid_txn, df_raw_final, closed_final, paid_final, open_final  = read_transformed_claims_data_from_parquet(extraction_date)
# Load data using the proper modules
# import claim data pipeline
# raw_claim_data = load_claims_data(extraction_date=extraction_date)
# transformed_claim_data = transform_claims_raw_data(raw_claim_data)
# df_raw_txn, closed_txn, open_txn, paid_txn, df_raw_final, closed_final, paid_final, open_final = transformed_claim_data

# # Check for cached periodized data
# periods_df = period_cache_manager.load_cache(df_raw_txn, extraction_date)

# if periods_df is None:
#     st.sidebar.info("Computing periodized data...")
#     # Create standardization config
#     config = StandardizationConfig(period_length_days=30, max_periods=60)
#     transformer.config = config
    
#     # Transform to periods
#     dataset = transformer.transform_claims_data(df_raw_txn)
#     periods_df = dataset.to_dataframes()['dynamic_periods']
    
#     # Save to cache
#     period_cache_manager.save_cache(periods_df, df_raw_txn, extraction_date)
#     st.sidebar.success("Periodized data cached")
# else:
#     st.sidebar.success("Loaded periodized data from cache")

# Data tables
with st.expander("All Transactions", expanded=False):
    st.subheader("All Transactions Data")
    st.dataframe(df_raw_txn, use_container_width=True)

with st.expander("All Final Claims", expanded=False):
    st.subheader("Final Status for All Claims")
    st.dataframe(df_raw_final, use_container_width=True)

# with st.expander("All Periods", expanded=False):
#     st.subheader("Standardized Periods Data")
#     st.dataframe(periods_df, use_container_width=True)


