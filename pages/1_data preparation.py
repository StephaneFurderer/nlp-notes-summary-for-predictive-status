import pandas as pd
import streamlit as st
from helpers.UI.common import get_shared_state
from helpers.UI.sidebar import test_if_claim_number_is_valid
from helpers.UI.plot_utils import plot_single_claim_lifetime

st.set_page_config(page_title="Claims Analysis", layout="wide")
st.title("Claims Analysis")

raw, filtered = get_shared_state()
df_raw_txn = raw["df_raw_txn"]
df_raw_final = raw["df_raw_final"]
df_raw_txn_to_periods = raw["df_raw_txn_to_periods"]

df_raw_txn_filtered = filtered["data"]["df_raw_txn"]
df_raw_final_filtered = filtered["data"]["df_raw_final"]
df_raw_txn_to_periods_filtered = filtered["data"]["df_raw_txn_to_periods"]
claim_number = filtered["filters"]["claim_number"]


# All data
with st.expander("All data", expanded=False):
    st.subheader("Transactions Data")
    st.dataframe(df_raw_txn, use_container_width=True)

    st.subheader("Final Status")
    st.dataframe(df_raw_final, use_container_width=True)

# filtered data
with st.expander("Filtered data", expanded=False):
    st.subheader("All Transactions Data")
    st.dataframe(df_raw_txn_filtered, use_container_width=True)

    st.subheader("Final Status for All Claims")
    st.dataframe(df_raw_final_filtered, use_container_width=True)

    st.subheader("Periods Data")
    st.dataframe(df_raw_txn_to_periods_filtered, use_container_width=True)

if test_if_claim_number_is_valid(claim_number):
    with st.expander("Lifetime Development", expanded=False):
        # lifetime per transaction date and period date
        st.subheader("Lifetime Development per Transaction Date")
        fig = plot_single_claim_lifetime(df_raw_txn_filtered, x_axis='datetxn', selected_claim=claim_number,y_axis=['reserve_cumsum', 'paid_cumsum', 'incurred_cumsum', 'expense_cumsum'])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Lifetime Development per Period")
        fig = plot_single_claim_lifetime(df_raw_txn_to_periods_filtered, x_axis='period', selected_claim=claim_number,y_axis=['reserve_cumsum', 'paid_cumsum', 'incurred_cumsum', 'expense_cumsum'])
        st.plotly_chart(fig, use_container_width=True)


