"""
Human interface to train RNN model using shared sidebar and filtered data.
"""

import streamlit as st
import pandas as pd
from helpers.UI.common import get_shared_state

st.set_page_config(page_title="RNN Training Lab", layout="wide")
st.title("RNN Training Lab")

_, filtered = get_shared_state()
df_txn = filtered["data"]["df_raw_txn"]
df_periods = filtered["data"]["df_raw_txn_to_periods"]
filters = filtered["filters"]

st.caption(f"Extraction: {filters['extraction_date']} | Cause: {filters['cause']} | Status: {filters['status']}")

st.subheader("Preview")
st.dataframe(df_txn.head(50), use_container_width=True)
st.dataframe(df_periods.head(50), use_container_width=True)
