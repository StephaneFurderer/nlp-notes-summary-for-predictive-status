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


def claim_number_filter(df_transaction_filtered, claim_number):
    if claim_number is not None and claim_number.strip() != "":
        df_transaction_filtered = df_transaction_filtered[df_transaction_filtered['clmNum'].astype(str).str.contains(claim_number.strip(), case=False, na=False)]
    return df_transaction_filtered

def advanced_sidebar(df_raw_txn,df_raw_final):
    # claim number selection
    # cause selection
    with st.sidebar:
        st.markdown("## Cause Selection")
        causes = ['ALL'] + list(df_raw_txn['clmCause'].dropna().unique())
        cause = st.selectbox("Cause", causes, help="Select the claim cause to analyze")

        # status selection
        st.markdown("## Status Selection")
        statuses = ['ALL'] + list(df_raw_txn['clmStatus'].dropna().unique())
        status = st.selectbox("Status", statuses, help="Select the claim status to analyze")

        st.markdown("## Claim Selection")
        df_transaction_filtered = df_raw_txn.copy()
        df_raw_final_filtered = df_raw_final.copy()
        # filter the data
        if cause != 'ALL':
            df_transaction_filtered = _filter_by_(df_transaction_filtered, 'clmCause', cause)
            df_raw_final_filtered = _filter_by_(df_raw_final_filtered, 'clmCause', cause)
        if status != 'ALL':
            df_transaction_filtered = _filter_by_(df_transaction_filtered, 'clmStatus', status)
            df_raw_final_filtered = _filter_by_(df_raw_final_filtered, 'clmStatus', status)

        claim_number = st.text_input(
            "Claim Number",
            value="",
            help="Enter a claim number to filter data, leave blank to show all"
        )
        df_raw_final_filtered = claim_number_filter(df_raw_final_filtered, claim_number)
        df_transaction_filtered = claim_number_filter(df_transaction_filtered, claim_number)

        
    return df_transaction_filtered, df_raw_final_filtered, cause, status, claim_number
