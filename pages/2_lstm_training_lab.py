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
    X, y, statuses = [], [], []
    unique_clmNums = df_periods[df_periods['dataset_split']==set_name]['clmNum'].unique()
    
    for clm_id in unique_clmNums: # iterate over the unique claim numbers
        # Get all periods for this claim, sorted by period
        claim_data = df_periods[df_periods['clmNum']==clm_id].sort_values('period')
        
        if len(claim_data) > 1:  # Only include claims with at least 2 periods
            # X: cumulative payments (input sequence) - shape: [sequence_length]
            X.append(claim_data['Y_star_cumsum'].values)
            
            # y: individual payments (target sequence) - shape: [sequence_length] 
            y.append(claim_data['Y_star'].values)
            
            # status: claim status (for color coding)
            statuses.append(claim_data['clmStatus'].iloc[0])  # Take the first status (should be consistent)
    
    # Convert to numpy arrays
    X = np.array(X, dtype=object)  # Use object dtype for variable length sequences
    y = np.array(y, dtype=object)
    statuses = np.array(statuses)
    
    return X, y, statuses    

X_train, y_train, status_train = prepare_X_y(df_periods,'train')
X_val, y_val, status_val = prepare_X_y(df_periods,'val')
X_test, y_test, status_test = prepare_X_y(df_periods,'test')

# Show sequence statistics
st.markdown("**Sequence Statistics:**")
st.write(f"Number of training sequences: {len(X_train)}")
st.write(f"Sequence lengths (first 10): {[len(seq) for seq in X_train[:10]]}")
st.write(f"Average sequence length: {np.mean([len(seq) for seq in X_train]):.2f}")

# Plot first 100 claims' cumulative payments on the same graph with color coding
st.markdown("**First 100 Claims - Cumulative Payments (Color-coded by Status):**")

# Create color mapping for claim statuses
unique_statuses = np.unique(status_train)
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
status_colors = {status: colors[i % len(colors)] for i, status in enumerate(unique_statuses)}

fig = go.Figure()

# Plot up to 100 sequences on the same graph with color coding
for i in range(min(100, len(X_train))):
    sample_seq = X_train[i]  # This is already a 1D array
    claim_status = status_train[i]
    color = status_colors[claim_status]
    
    fig.add_trace(go.Scatter(
        x=list(range(len(sample_seq))),
        y=sample_seq,
        mode='lines',
        name=f'Claim {i+1} ({claim_status})',
        showlegend=False,
        opacity=0.6,
        line=dict(width=1, color=color)
    ))

# Add legend for status colors
legend_traces = []
for status, color in status_colors.items():
    legend_traces.append(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        name=status,
        line=dict(color=color, width=3),
        showlegend=True
    ))

fig.add_traces(legend_traces)

fig.update_layout(
    title="First 100 Claims - Cumulative Payment Patterns (Color-coded by Status)",
    xaxis_title="Period",
    yaxis_title="Cumulative Payments (Scaled)",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# Show status distribution
st.markdown("**Claim Status Distribution:**")
status_counts = pd.Series(status_train).value_counts()
st.bar_chart(status_counts)

 # for the LSTM model, we need to reshape the data to [batch_size, sequence_length, 1]


