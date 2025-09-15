from helpers.functions.claims_utils import load_data, _filter_by_, calculate_claim_features
from helpers.functions.plot_utils import plot_single_claim_lifetime
import pandas as pd
import streamlit as st
from helpers.functions.notes_utils import NotesReviewerAgent

@st.cache_data
def cached_load_data(report_date=None):
    """Cached wrapper for load_data function"""
    return load_data(report_date)

@st.cache_data
def cached_calculate_claim_features(df_hash):
    """Cached wrapper for calculate_claim_features function"""
    # Note: We pass a hash instead of the dataframe directly for caching
    # The actual dataframe will be loaded from the main cached_load_data
    df_raw_txn, _, _, _, _, _, _, _ = cached_load_data(None)
    return calculate_claim_features(df_raw_txn)

@st.cache_data
def cached_load_notes():
    """Cached wrapper for notes loading"""
    agent = NotesReviewerAgent()
    return agent.import_notes(), agent

st.title("NLP Notes Summary for Predictive Status")
report_date = st.sidebar.date_input("Report Date", value=None, help="Select the date of the report that would consider that any claims not closed by that date is still open and any claims closed and not reopened is closed")

if report_date is not None:
    report_date = pd.Timestamp(report_date)

# get timeseries data (cached)
df_raw_txn,closed_txn,open_txn,paid_txn,df_raw_final,closed_final,paid_final,open_final = cached_load_data(report_date)

# calculate claim features (cached)
all_claim_metrics_df,summary_all_stats,display_all_claims_metrics_df = cached_calculate_claim_features(str(report_date))

# get claim id
st.sidebar.markdown("------")
st.sidebar.markdown("🔍 Portfolio Filters")
selected_cause = st.sidebar.selectbox("Select Claim Cause", ['ALL', *df_raw_txn['clmCause'].dropna().unique()])

st.sidebar.markdown("🔍 Claim Search")
claim_filter = st.sidebar.selectbox("Filter by Claim Number (optional)", [*df_raw_txn[df_raw_txn['clmCause']==selected_cause]['clmNum'].unique()], help="Enter a claim number to filter data, leave blank to show all")

# Panel to visualize the lifetime of a claim at the center of the screen
st.markdown("--")
st.markdown("## 📈 Claim Lifetime")
if claim_filter.strip() and len(claim_filter) > 0:
    df_raw_txn_filtered = _filter_by_(df_raw_txn,'clmNum',claim_filter) #df_raw_txn[df_raw_txn['clmNum'].str.contains(claim_filter.strip(), case=False, na=False)]
    df_raw_final_filtered = _filter_by_(df_raw_final,'clmNum',claim_filter) #df_raw_final[df_raw_final['clmNum'].str.contains(claim_filter.strip(), case=False, na=False)]
    all_claim_metrics_df_filtered = _filter_by_(all_claim_metrics_df,'clmNum',claim_filter) #all_claim_metrics_df[all_claim_metrics_df['clmNum'].str.contains(claim_filter.strip(), case=False, na=False)]
    plot_single_claim_lifetime(df_raw_txn,claim_filter)
    # Display dataframes
    st.write("### Transactions")
    st.write(f"Shape: {df_raw_txn_filtered.shape[0]:,} rows x {df_raw_txn_filtered.shape[1]} columns")
    st.dataframe(df_raw_txn_filtered,use_container_width=True)

    st.write("### Final Status")
    st.write(f"Shape: {df_raw_final_filtered.shape[0]:,} rows x {df_raw_final_filtered.shape[1]} columns")
    st.dataframe(df_raw_final_filtered[["clmCause","booknum","cidpol","clmNum","clmStatus","dateCompleted","dateReopened","paid_cumsum","expense_cumsum","incurred_cumsum"]],use_container_width=True)

    st.write("### Features ")
    st.write(f"Shape: {all_claim_metrics_df_filtered.shape[0]:,} rows x {all_claim_metrics_df_filtered.shape[1]} columns")
    st.dataframe(all_claim_metrics_df_filtered, use_container_width=True)

# get notes (cached)
notes_df, agent = cached_load_notes()

claim_notes = agent.get_notes_by_claim(claim_filter)
st.metric("Total Notes", len(claim_notes))

# Display notes timeline
if claim_filter:
    st.markdown("---")
    st.subheader(f"Notes Timeline for {claim_filter}")
    
    # Get timeline data
    timeline_df = agent.get_notes_timeline(claim_filter)
    
    if len(timeline_df) > 0:
        # Create timeline view
        for idx, row in timeline_df.iterrows():
            # Create a card-like container for each note
            with st.container():
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    # Date and time info
                    st.markdown(f"**{row['whenadded'].strftime('%Y-%m-%d')}**")
                    st.markdown(f"*{row['whenadded'].strftime('%H:%M')}*")
                    if row['days_since_first'] > 0:
                        st.caption(f"+{row['days_since_first']} days")
                
                with col2:
                    # Note content
                    st.markdown(f"**{row['note']}**")
                    
                    # Note metadata
                    col2a, col2b, col2c = st.columns(3)
                    with col2a:
                        st.caption(f"Length: {row['note_length']} chars")
                    with col2b:
                        st.caption(f"Words: {row['note_word_count']}")
                    with col2c:
                        st.caption(f"Note #{idx + 1}")
                
                # Separator line
                st.markdown("---")
                
                # Add some spacing for better readability
                st.markdown("&nbsp;")
    
    else:
        st.info("No notes found for this claim.")