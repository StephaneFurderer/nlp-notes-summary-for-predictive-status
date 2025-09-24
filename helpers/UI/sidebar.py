import streamlit as st
import pandas as pd
from helpers.functions.load_cache_data import get_available_data_versions
from typing import List

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

def test_if_claim_number_is_valid(claim_number):
    if claim_number is not None and claim_number.strip() != "":
        return True
    return False

def claim_number_filter(df_transaction_filtered, claim_number):
    if claim_number is not None and claim_number.strip() != "":
        df_transaction_filtered = df_transaction_filtered[df_transaction_filtered['clmNum'].astype(str).str.contains(claim_number.strip(), case=False, na=False)]
    return df_transaction_filtered

def advanced_sidebar(dataframes:List[pd.DataFrame]):
    """
    Accepts a list of dataframes and applies the same filters (cause, status, claim number) to each.
    Returns the list of filtered dataframes along with the selected cause, status, and claim_number.
    """
    if not isinstance(dataframes, list) or len(dataframes) == 0:
        raise ValueError("Input must be a non-empty list of dataframes.")

    # Use the first dataframe to get unique causes and statuses for the sidebar
    df_for_options = dataframes[0]

    with st.sidebar:
        st.markdown("## Cause Selection")
        causes = ['ALL'] + list(df_for_options['clmCause'].dropna().unique())
        cause = st.selectbox("Cause", causes, help="Select the claim cause to analyze")

        st.markdown("## Status Selection")
        statuses = ['ALL'] + list(df_for_options['clmStatus'].dropna().unique())
        status = st.selectbox("Status", statuses, help="Select the claim status to analyze")

        st.markdown("## Claim Selection")
        claim_number = st.text_input(
            "Claim Number",
            value="",
            help="Enter a claim number to filter data, leave blank to show all"
        )

        filtered_dataframes = []
        for df in dataframes:
            df_filtered = df.copy()
            if cause != 'ALL':
                df_filtered = _filter_by_(df_filtered, 'clmCause', cause)
            if status != 'ALL':
                df_filtered = _filter_by_(df_filtered, 'clmStatus', status)
            df_filtered = claim_number_filter(df_filtered, claim_number)
            filtered_dataframes.append(df_filtered)

    return filtered_dataframes, cause, status, claim_number
