import pandas as pd
import streamlit as st
from helpers.functions.claims_utils import load_data

st.set_page_config(page_title="Claims Analysis", layout="wide")
st.title("Claims Analysis")

# Sidebar controls
st.sidebar.markdown("## Data Selection")
extraction_date = st.sidebar.selectbox(
    "Extraction Date",
    ["2025-09-21"],  # Add more dates as available
    help="Select the data extraction date to analyze"
)

# Load data
data = load_data(extraction_date=extraction_date)
df_raw_txn, closed_txn, open_txn, paid_txn, df_raw_final, closed_final, paid_final, open_final = data

# Data tables
with st.expander("All Transactions", expanded=False):
    st.subheader("All Transactions Data")
    st.dataframe(df_raw_txn, use_container_width=True)

with st.expander("All Final Claims", expanded=False):
    st.subheader("Final Status for All Claims")
    st.dataframe(df_raw_final, use_container_width=True)


