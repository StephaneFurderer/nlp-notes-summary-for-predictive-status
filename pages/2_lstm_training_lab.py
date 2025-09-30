"""
Human interface to train RNN model using shared sidebar and filtered data.
Step-by-step testing lab for LSTM sequence preparation.
"""

import streamlit as st
from typing import List
import pandas as pd
import numpy as np
import sys
import traceback
from helpers.UI.common import get_shared_state
from helpers.functions.rnn_data_prep import prepare_lstm_sequences, censor_large_payments, scale_features
from plotly.subplots import make_subplots
import plotly.graph_objects as go


st.set_page_config(page_title="LSTM Training Lab", layout="wide")
st.title("LSTM Training Lab")
st.markdown("Step-by-step LSTM model preparation and training")



from helpers.functions.CONST import BASE_DATA_DIR
import os
extraction_date = "2025-09-21"
df_periods = pd.read_parquet(os.path.join(BASE_DATA_DIR, extraction_date,"closed_txn_to_periods.parquet"))

st.sidebar.header("Filtering Options")
evaluation_date = st.sidebar.text_input("Evaluation Date", value="2025-09-30", help="Evaluation date for reserving (YYYY-MM-DD format)")

def filter_data_for_lstm_training(df_periods,evaluation_date:str='2025-09-30',clmCause:str='ABB_SLIP_&_FALL',clmStatus:List[str]=['CLOSED','PAID','DENIED']):
    """ Filter closed claims for LSTM training """
    # remove blank spaces from clmStatus and clmCause
    df_periods['clmStatus'] = df_periods['clmStatus'].str.replace(' ', '')
    df_periods['clmCause'] = df_periods['clmCause'].str.replace(' ', '')
    # filtering the data to only include claims that are closed and part of slip and fall claims
    df_periods = df_periods[df_periods['clmCause'].isin([clmCause])]
    df_periods = df_periods[df_periods['clmStatus'].isin(clmStatus)]
    df_periods = df_periods[df_periods['period_end_date']<=pd.to_datetime(evaluation_date)]
    return df_periods

# Step 1: Data Preview
st.header("Step 1: Your Data")
st.write(f"**Shape:** {df_periods.shape}")
st.write(f"**unique claims:** {df_periods['clmNum'].nunique()}")

df_periods = filter_data_for_lstm_training(df_periods,evaluation_date = evaluation_date)

with st.expander("Data Preview", expanded=False):
    st.dataframe(df_periods.head(20))


def split_data_for_lstm(df_periods, split_ratio: List[float] = [0.6, 0.2, 0.2]):
    """Split the data into train, val, test based on period_start_date and given ratios."""
    df = df_periods.sort_values(['clmNum', 'period_start_date']).copy()
    n = len(df)
    train_end = int(split_ratio[0] * n)
    val_end = train_end + int(split_ratio[1] * n)
    df['dataset_split'] = np.select(
        [df.index < train_end, df.index < val_end],
        ['train', 'val'],
        default='test'
    )
    return df

##### Step 2: Data Preparation for LSTM training
# sort the claims by period_start_date and split into train, val, test with  60/20/20 ratio
df_periods = split_data_for_lstm(df_periods)

df_periods['Y'] = df_periods['paid']

# scale the data based on the train data
df_train = df_periods[df_periods['dataset_split']=='train']
df_train_mean = df_train['Y'].mean()
df_train_std = df_train['Y'].std()
df_periods['Y_star'] = (df_periods['Y'] - df_train_mean) / df_train_std
df_periods['Y_star_cumsum'] = df_periods['Y_star'].cumsum()

with st.expander("Training Data Statistics"):
    st.write(f"**mean of Y:** {df_train_mean}")
    st.write(f"**std of Y:** {df_train_std}")
    st.write(f"**mean of Y_star:** {df_periods['Y_star'].mean()}")
    st.write(f"**std of Y_star:** {df_periods['Y_star'].std()}")


# prepare the sequences from the df_periods

def prepare_X_y(df_periods,set_name:str=['train','val','test']):
    """Prepare the X and y for the LSTM model for a given set name"""
    X, y = [], []
    unique_clmNums = df_periods[df_periods['dataset_split']==set_name]['clmNum'].unique()
    
    for clm_id in unique_clmNums: # iterate over the unique claim numbers
        # Get all periods for this claim, sorted by period
        claim_data = df_periods[df_periods['clmNum']==clm_id].sort_values('period')
        
        if len(claim_data) > 1:  # Only include claims with at least 2 periods
            # X: cumulative payments (input sequence) - shape: [sequence_length]
            X.append(claim_data['Y_star_cumsum'].values)
            
            # y: individual payments (target sequence) - shape: [sequence_length] 
            y.append(claim_data['Y_star'].values)
    
    # Convert to numpy arrays
    X = np.array(X, dtype=object)  # Use object dtype for variable length sequences
    y = np.array(y, dtype=object)
    
    return X, y    

X_train, y_train = prepare_X_y(df_periods,'train')
X_val, y_val = prepare_X_y(df_periods,'val')
X_test, y_test = prepare_X_y(df_periods,'test')


# Show sample sequences
st.markdown("**Sample Input Sequences (Cumulative Payments):**")
fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Sequence 1", "Sequence 2", "Sequence 3", "Sequence 4")
)

for i in range(min(4, len(X_train))):
    row = i // 2 + 1
    col = i % 2 + 1
    
    sample_seq = X_train[i]  # This is already a 1D array
    
    fig2.add_trace(go.Scatter(
        x=list(range(len(sample_seq))),
        y=sample_seq,
        mode='lines+markers',
        name=f'Claim {i+1}',
        showlegend=False
    ), row=row, col=col)

fig2.update_layout(
    title="Sample Input Sequences (Cumulative Payments)",
    height=500
)
st.plotly_chart(fig2, use_container_width=True)

# Show corresponding target sequences
st.markdown("**Sample Target Sequences (Individual Payments):**")
fig3 = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Target 1", "Target 2", "Target 3", "Target 4")
)

for i in range(min(4, len(y_train))):
    row = i // 2 + 1
    col = i % 2 + 1
    
    sample_target = y_train[i]  # This is already a 1D array
    
    fig3.add_trace(go.Scatter(
        x=list(range(len(sample_target))),
        y=sample_target,
        mode='lines+markers',
        name=f'Target {i+1}',
        showlegend=False
    ), row=row, col=col)

fig3.update_layout(
    title="Sample Target Sequences (Individual Payments)",
    height=500
)
st.plotly_chart(fig3, use_container_width=True)

# Show sequence statistics
st.markdown("**Sequence Statistics:**")
st.write(f"Number of training sequences: {len(X_train)}")
st.write(f"Sequence lengths: {[len(seq) for seq in X_train[:10]]}")  # Show first 10 lengths
st.write(f"Average sequence length: {np.mean([len(seq) for seq in X_train]):.2f}")


 # for the LSTM model, we need to reshape the data to [batch_size, sequence_length, 1]


