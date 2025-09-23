import streamlit as st
import pandas as pd
from helpers.functions.claims_utils import load_data

st.set_page_config(page_title="Claims Data Viewer", layout="wide")
st.title("Claims Data Viewer")

# Load data
extraction_date = "2025-09-21"
data = load_data(extraction_date=extraction_date)
df_raw_txn, closed_txn, open_txn, paid_txn, df_raw_final, closed_final, paid_final, open_final = data

# Summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Transactions", len(df_raw_txn))
with col2:
    st.metric("Total Claims", df_raw_txn['clmNum'].nunique())
with col3:
    st.metric("Open Claims", len(open_final))
with col4:
    st.metric("Closed Claims", len(closed_final))

# All Transactions Table
with st.expander("All Transactions", expanded=False):
    st.subheader("All Transactions Data")
    st.dataframe(df_raw_txn, use_container_width=True)

# All Final Table  
with st.expander("All Final Claims", expanded=False):
    st.subheader("Final Status for All Claims")
    st.dataframe(df_raw_final, use_container_width=True)

# All Periods Table
with st.expander("All Periods", expanded=False):
    st.subheader("All Periods Data")
    # This would show periodized data if available
    st.write("Period data would be displayed here")
