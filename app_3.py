import pandas as pd
import streamlit as st
from helpers.UI.sidebar import initialize_sidebar, advanced_sidebar, test_if_claim_number_is_valid
from helpers.functions.toPeriods_utils import create_period_column_fast
from helpers.functions.claims_utils import read_transformed_claims_data_from_parquet, read_periods_data

from helpers.UI.plot_utils import plot_single_claim_lifetime

from helpers.functions.standardized_claims_transformer import StandardizedClaimsTransformer
from helpers.functions.standardized_claims_schema import StandardizationConfig

st.set_page_config(page_title="Claims Analysis", layout="wide")
st.title("Claims Analysis")



# Sidebar controls
extraction_date = initialize_sidebar()
df_raw_txn, closed_txn, open_txn, paid_txn, df_raw_final, closed_final, paid_final, open_final  = read_transformed_claims_data_from_parquet(extraction_date)
df_raw_txn_to_periods, closed_txn_to_periods, open_txn_to_periods, paid_txn_to_periods = read_periods_data(extraction_date)
df_raw_txn_filtered, df_raw_final_filtered, df_raw_txn_to_periods_filtered, cause, status, claim_number = advanced_sidebar([df_raw_txn, df_raw_final, df_raw_txn_to_periods])
#df_periods_filtered = create_period_column_fast(df_raw_txn_filtered)


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


