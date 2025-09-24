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

st.set_page_config(page_title="RNN Training Lab", layout="wide")
st.title("🧠 RNN Training Lab")
st.markdown("Step-by-step LSTM model preparation and training")



from helpers.functions.CONST import BASE_DATA_DIR
import os
extraction_date = "2025-09-21"
df_periods = pd.read_parquet(os.path.join(BASE_DATA_DIR, extraction_date,"closed_txn_to_periods.parquet"))
# This would be called with your closed claims dataframe


# Step 1: Data Preview
st.header("Step 1: Your Data")
st.write(f"**Shape:** {df_periods.shape}")
st.write("**Columns:**", list(df_periods.columns))

with st.expander("Data Preview", expanded=False):
    st.dataframe(df_periods.head(20))

# Step 2: Configure LSTM sequence preparation
st.header("Step 2: Configure LSTM Sequences")


claim_id_col = 'clmNum'
period_col = 'period'
payment_col = 'paid'
expense_col = 'expense'
status_col = 'clmStatus'


with st.sidebar:
    st.subheader("Parameters")
    max_periods = st.number_input("Max Periods", value=60, min_value=6, max_value=60)
    min_periods = st.number_input("Min Periods", value=3, min_value=1, max_value=10)

    # Test unique claims
    unique_claims = df_periods[claim_id_col].nunique()
    st.info(f"Found {unique_claims:,} unique claims in your data")

# Preview selected data
with st.expander("Selected Columns Preview", expanded=False):
    st.subheader("Selected Columns Preview")
    preview_cols = [claim_id_col, status_col, period_col, payment_col,expense_col]
    st.dataframe(df_periods[preview_cols].head(10))

# Step 3: Test sequence preparation
st.header("Step 3: Test Sequence Preparation")

if st.button("🔬 Prepare LSTM Sequences", type="primary"):
    try:
        with st.spinner("Preparing sequences..."):
            # Create a simple split for testing (you can modify this)
            # Create train/val/test split (80/10/10)
            n_claims = df_periods[claim_id_col].nunique()
            claim_ids = df_periods[claim_id_col].unique()

            train_end = int(0.8 * n_claims)
            val_end = int(0.9 * n_claims)

            train_ids = claim_ids[:train_end]
            val_ids = claim_ids[train_end:val_end]
            test_ids = claim_ids[val_end:]

            df_periods['dataset_split'] = np.where(df_periods[claim_id_col].isin(train_ids), 'train', np.where(df_periods[claim_id_col].isin(val_ids), 'val', 'test'))
            
            df_train = df_periods[df_periods["dataset_split"] == "train"]
            df_val = df_periods[df_periods["dataset_split"] == "val"]
            df_test = df_periods[df_periods["dataset_split"] == "test"]

            # Call the function
            sequences_data = prepare_lstm_sequences(
                df_train,
                claim_id_col=claim_id_col,
                period_col=period_col,
                payment_col=payment_col,
                split_col='dataset_split',
                max_periods=max_periods,
                min_periods=min_periods
            )

            # Store in session state
            st.session_state['sequences_data'] = sequences_data

            st.success("✅ Sequences prepared successfully!")

            # Show results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Claims Processed", len(sequences_data['metadata']))
            with col2:
                st.metric("Sequence Length", sequences_data['max_periods'])
            with col3:
                st.metric("Feature Dimensions", sequences_data['sequences'].shape[-1])

    except Exception as e:
        st.error(f"❌ Error preparing sequences: {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())

# Step 4: Show results (if sequences are ready)
if 'sequences_data' in st.session_state:
    st.header("Step 4: Results")

    sequences_data = st.session_state['sequences_data']

    # Data shapes
    st.subheader("📊 Data Shapes")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.code(f"Sequences: {sequences_data['sequences'].shape}")
    with col2:
        st.code(f"Static: {sequences_data['static_features'].shape}")
    with col3:
        st.code(f"Targets: {sequences_data['targets'].shape}")

    # Metadata preview
    st.subheader("📋 Claim Metadata")
    st.dataframe(sequences_data['metadata'].head())

    # Payment analysis
    st.subheader("💰 Payment Analysis")
    targets = sequences_data['targets']
    payment_indicators = targets[:, :, 0].flatten()
    payment_amounts = targets[:, :, 1].flatten()
    non_zero_payments = payment_amounts[payment_amounts != 0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Observations", len(payment_indicators))
        st.metric("Non-zero Payments", len(non_zero_payments))
    with col2:
        zero_ratio = 1 - np.mean(payment_indicators)
        st.metric("Zero Payment Ratio", f"{zero_ratio:.1%}")
        if len(non_zero_payments) > 0:
            st.metric("Mean Non-zero Payment", f"${np.mean(non_zero_payments):,.2f}")
    with col3:
        if len(non_zero_payments) > 0:
            st.metric("Max Payment", f"${np.max(non_zero_payments):,.2f}")
            st.metric("95th Percentile", f"${np.percentile(non_zero_payments, 95):,.2f}")

# Next steps
if 'sequences_data' in st.session_state:
    st.header("✅ Ready for Next Steps")
    st.markdown("""
    Your data has been successfully converted to LSTM format! Next steps:
    1. **Payment Censoring**: Handle extremely large payments
    2. **Feature Scaling**: Normalize features for training
    3. **Model Training**: Train the LSTM network
    4. **Evaluation**: Compare with traditional methods
    """)
else:
    st.header("🎯 Next Steps")
    st.markdown("""
    1. Review the column mappings above
    2. Click **"Prepare LSTM Sequences"** to test the data conversion
    3. Check the results and fix any issues
    """)

# Debug info
st.sidebar.header("🔍 Debug Info")
st.sidebar.write(f"Claims: {unique_claims:,}")
st.sidebar.write(f"Columns: {len(df_periods.columns)}")
st.sidebar.write(f"Rows: {len(df_periods):,}")

if 'sequences_data' in st.session_state:
    st.sidebar.success("✅ Sequences ready")
else:
    st.sidebar.info("ℹ️ Ready to test")
