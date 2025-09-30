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
# This would be called with your closed claims dataframe
df_periods['clmStatus'] = df_periods['clmStatus'].str.replace(' ', '')

# filtering the data to only include claims that are closed and part of slip and fall claims
df_periods = df_periods[df_periods['clmCause'].isin(['ABB_SLIP_&_FALL'])]
df_periods = df_periods[df_periods['clmStatus'].isin(['CLOSED','PAID','DENIED'])]

# Step 1: Data Preview
st.header("Step 1: Your Data")
st.write(f"**Shape:** {df_periods.shape}")
st.write("**Columns:**", list(df_periods.columns))

with st.expander("Data Preview", expanded=False):
    st.dataframe(df_periods.head(20))
