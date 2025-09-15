from helpers.functions.data_utils import load_data
import pandas as pd
import streamlit as st

st.title("NLP Notes Summary for Predictive Status")
report_date = st.sidebar.date_input("Report Date", value=None, help="Select the date of the report that would consider that any claims not closed by that date is still open and any claims closed and not reopened is closed")

if report_date is not None:
    report_date = pd.Timestamp(report_date)

df_raw_txn,closed_txn,open_txn,paid_txn,df_raw_final,closed_final,paid_final,open_final = load_data(report_date)