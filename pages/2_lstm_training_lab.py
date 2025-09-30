"""
Human interface to train RNN model using shared sidebar and filtered data.
Step-by-step testing lab for LSTM sequence preparation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import traceback
from helpers.UI.common import get_shared_state
from helpers.functions.rnn_data_prep import prepare_lstm_sequences, censor_large_payments, scale_features

st.set_page_config(page_title="LSTM Training Lab", layout="wide")
st.title("LSTM Training Lab")
st.markdown("Step-by-step LSTM model preparation and training")



from helpers.functions.CONST import BASE_DATA_DIR
import os
extraction_date = "2025-09-21"
df_periods = pd.read_parquet(os.path.join(BASE_DATA_DIR, extraction_date,"closed_txn_to_periods.parquet"))

st.sidebar.header("Filtering Options")
evaluation_date = st.sidebar.text_input("Evaluation Date", value="2025-09-30", help="Evaluation date for reserving (YYYY-MM-DD format)")

def filter_data_for_lstm_training(df_periods,evaluation_date:str='2025-09-30',clmCause:str='ABB_SLIP_&_FALL',clmStatus:str=['CLOSED','PAID','DENIED']):
    """ Filter closed claims for LSTM training """
    # remove blank spaces from clmStatus and clmCause
    df_periods['clmStatus'] = df_periods['clmStatus'].str.replace(' ', '')
    df_periods['clmCause'] = df_periods['clmCause'].str.replace(' ', '')
    # filtering the data to only include claims that are closed and part of slip and fall claims
    df_periods = df_periods[df_periods['clmCause'].isin([clmCause])]
    df_periods = df_periods[df_periods['clmStatus'].isin([clmStatus])]
    df_periods = df_periods[df_periods['period_end_date']<=pd.to_datetime(evaluation_date)]
    return df_periods

# Step 1: Data Preview
st.header("Step 1: Your Data")
st.write(f"**Shape:** {df_periods.shape}")
st.write(f"**unique claims:** {df_periods['clmNum'].nunique()}")

df_periods = filter_data_for_lstm_training(df_periods,evaluation_date = evaluation_date)

with st.expander("Data Preview", expanded=False):
    st.dataframe(df_periods.head(20))
