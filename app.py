from helpers.functions.data_utils import load_data
import pandas as pd
import streamlit as st
from helpers.functions.notes_utils import NotesReviewerAgent

st.title("NLP Notes Summary for Predictive Status")
report_date = st.sidebar.date_input("Report Date", value=None, help="Select the date of the report that would consider that any claims not closed by that date is still open and any claims closed and not reopened is closed")

if report_date is not None:
    report_date = pd.Timestamp(report_date)

# get timeseries data
df_raw_txn,closed_txn,open_txn,paid_txn,df_raw_final,closed_final,paid_final,open_final = load_data(report_date)

# get claim id
st.sidebar.markdown("------")
st.sidebar.markdown("🔍 Portfolio Filters")
selected_cause = st.sidebar.selectbox("Select Claim Cause", ['ALL', *df_raw_txn['clmCause'].dropna().unique()])

st.sidebar.markdown("🔍 Claim Search")
claim_filter = st.sidebar.text_input("Filter by Claim Number (optional)", [*df_raw_txn[df_raw_txn['clmCause']==selected_cause]['clmNum'].unique()], help="Enter a claim number to filter data, leave blank to show all")


# get notes
agent = NotesReviewerAgent()
notes_df = agent.import_notes()

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