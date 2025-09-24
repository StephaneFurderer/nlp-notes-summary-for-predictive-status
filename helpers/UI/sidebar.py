import streamlit as st
import pandas as pd
from helpers.functions.load_cache_data import get_available_data_versions


available_versions = get_available_data_versions()
available_dates = [v['extraction_date'] for v in available_versions]

def _filter_by_(df,col,value):
    return df[df[col].str.contains(value.strip(), case=False, na=False)]


def initialize_sidebar():
    with st.sidebar:
        # date selection
        st.markdown("## Data Selection")
        extraction_date = st.selectbox(
        "Extraction Date",
        available_dates,
        help="Select the data extraction date to analyze"
        )
    return extraction_date

def advanced_sidebar(df_raw_txn):
    # claim number selection
    # cause selection
    st.markdown("## Cause Selection")
    cause = st.selectbox("Cause", df_raw_txn['clmCause'].dropna().unique(), help="Select the claim cause to analyze")

    # status selection
    st.markdown("## Status Selection")
    status = st.selectbox("Status", df_raw_txn['clmStatus'].dropna().unique(), help="Select the claim status to analyze")


    st.markdown("## Claim Selection")
    df_transaction_filtered = df_raw_txn.copy()
    # filter the data
    if cause == 'ALL':
        cause = None
    else:
        cause = _filter_by_(df_transaction_filtered,'clmCause',cause)
    if status == 'ALL':
        status = None
    else:
        status = _filter_by_(df_transaction_filtered,'clmStatus',status)
    
    claim_number = st.text_input("Claim Number", df_transaction_filtered['clmNum'].dropna().unique(), help="Enter a claim number to filter data, leave blank to show all")

    return df_transaction_filtered, cause, status, claim_number