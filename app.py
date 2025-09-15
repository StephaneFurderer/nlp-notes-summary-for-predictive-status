from helpers.functions.claims_utils import load_data, _filter_by_, calculate_claim_features
from helpers.functions.plot_utils import plot_single_claim_lifetime
from helpers.functions.nlp_utils import ClaimNotesNLPAnalyzer, create_nlp_feature_summary, analyze_claims_with_caching
from helpers.functions.ml_models import ClaimStatusBaselineModel, create_model_performance_summary, plot_feature_importance, plot_confusion_matrix
import pandas as pd
import streamlit as st
import os
from helpers.functions.notes_utils import NotesReviewerAgent

# Set page config
st.set_page_config(
    page_title="Claim Notes Analysis",
    page_icon="📊",
    layout="wide"
)

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

def get_nlp_analysis_with_progress(report_date, claims_df):
    """Get NLP analysis with progress tracking"""
    notes_df, _ = cached_load_notes()

    # Check if cache exists first
    from helpers.functions.nlp_utils import _generate_nlp_data_hash, _get_nlp_cache_path
    data_hash = _generate_nlp_data_hash(notes_df, report_date)
    cache_path = _get_nlp_cache_path(data_hash)

    if os.path.exists(cache_path):
        # Load from cache instantly
        return pd.read_parquet(cache_path)

    # If no cache, show progress during calculation
    st.info("🔄 Calculating NLP features for eligible claims...")

    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(percent, current, total):
        progress_bar.progress(percent)
        status_text.text(f"Processing claim {current} of {total} ({percent:.1%})")

    # Calculate with progress tracking
    result = analyze_claims_with_caching(
        notes_df,
        claims_df=claims_df,
        report_date=report_date,
        progress_callback=update_progress
    )

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    st.success(f"✅ NLP analysis completed for {len(result)} eligible claims!")

    return result

@st.cache_data
def cached_nlp_analysis_simple(report_date_str):
    """Simple cached wrapper for when cache exists"""
    notes_df, _ = cached_load_notes()
    report_date = None if report_date_str == "None" else pd.to_datetime(report_date_str)

    # Check cache first
    from helpers.functions.nlp_utils import _generate_nlp_data_hash, _get_nlp_cache_path
    data_hash = _generate_nlp_data_hash(notes_df, report_date)
    cache_path = _get_nlp_cache_path(data_hash)

    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    else:
        # This will trigger the progress version
        return None

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

st.sidebar.markdown("🔍 NLP Analysis Filters")
# Show info about NLP eligible claims
final_status_filter = df_raw_final['clmStatus'].isin(['PAID', 'DENIED', 'CLOSED'])
initial_review_reopened = (df_raw_final['clmStatus'] == 'INITIAL_REVIEW') & (df_raw_final['dateReopened'].notna())
eligible_claims_df = df_raw_final[final_status_filter | initial_review_reopened]

st.sidebar.info(f"""
**NLP Analysis Criteria:**
- PAID/DENIED/CLOSED: All claims
- INITIAL_REVIEW: Only if reopened

**Eligible Claims:** {len(eligible_claims_df):,}
- PAID: {(eligible_claims_df['clmStatus'] == 'PAID').sum():,}
- DENIED: {(eligible_claims_df['clmStatus'] == 'DENIED').sum():,}
- CLOSED: {(eligible_claims_df['clmStatus'] == 'CLOSED').sum():,}
- INITIAL_REVIEW (reopened): {((eligible_claims_df['clmStatus'] == 'INITIAL_REVIEW') & (eligible_claims_df['dateReopened'].notna())).sum():,}
""")

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

# get NLP features (cached with report date awareness and progress tracking)
# First try the cached version
nlp_features_df = cached_nlp_analysis_simple(str(report_date))

# If cache miss, use progress version
if nlp_features_df is None:
    nlp_features_df = get_nlp_analysis_with_progress(report_date, df_raw_final)

claim_notes = agent.get_notes_by_claim(claim_filter)
st.metric("Total Notes", len(claim_notes))

# Display NLP Analysis section
st.markdown("---")
st.markdown("## 🤖 NLP Analysis")

# Create tabs for different NLP views
nlp_tab1, nlp_tab2, nlp_tab3, nlp_tab4 = st.tabs(["📊 Feature Summary", "🔍 Claim Analysis", "📈 Insights", "🤖 ML Baseline"])

with nlp_tab1:
    st.subheader("NLP Feature Summary")
    if len(nlp_features_df) > 0:
        summary_stats = create_nlp_feature_summary(nlp_features_df)
        st.dataframe(summary_stats, use_container_width=True)

        # Show keyword category totals
        st.subheader("Keyword Category Totals")
        keyword_cols = [col for col in nlp_features_df.columns if col.endswith('_count')]
        keyword_totals = nlp_features_df[keyword_cols].sum().sort_values(ascending=False)
        st.bar_chart(keyword_totals)

with nlp_tab2:
    st.subheader("Individual Claim NLP Features")
    if claim_filter and len(nlp_features_df) > 0:
        # Get NLP features for the selected claim
        claim_nlp_features = nlp_features_df[nlp_features_df['clmNum'] == claim_filter]

        if len(claim_nlp_features) > 0:
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                sentiment_score = claim_nlp_features['sentiment_score'].iloc[0]
                st.metric("Sentiment Score", f"{sentiment_score:.2f}")
            with col2:
                total_notes = claim_nlp_features['total_notes'].iloc[0]
                st.metric("Total Notes", int(total_notes))
            with col3:
                comm_freq = claim_nlp_features['communication_frequency'].iloc[0]
                st.metric("Comm. Frequency", f"{comm_freq:.3f}")
            with col4:
                last_sentiment = claim_nlp_features['last_note_sentiment'].iloc[0]
                st.metric("Last Note Sentiment", f"{last_sentiment:.2f}")

            # Show detailed features
            st.subheader("Detailed NLP Features")
            feature_display = claim_nlp_features.T
            feature_display.columns = ['Value']
            st.dataframe(feature_display, use_container_width=True)
        else:
            st.info(f"No NLP features found for claim {claim_filter}")
    else:
        st.info("Select a claim to see detailed NLP analysis")

with nlp_tab3:
    st.subheader("NLP Insights & Patterns")
    if len(nlp_features_df) > 0:
        # Merge with claim status data for insights
        merged_df = nlp_features_df.merge(
            df_raw_final[['clmNum', 'clmStatus', 'clmCause']],
            on='clmNum',
            how='left'
        )

        if len(merged_df) > 0:
            # Sentiment by status
            st.subheader("Sentiment by Claim Status")
            sentiment_by_status = merged_df.groupby('clmStatus')['sentiment_score'].mean().sort_values(ascending=False)
            st.bar_chart(sentiment_by_status)

            # Communication frequency by status
            st.subheader("Communication Frequency by Claim Status")
            comm_by_status = merged_df.groupby('clmStatus')['communication_frequency'].mean().sort_values(ascending=False)
            st.bar_chart(comm_by_status)

            # Top predictive keywords
            st.subheader("Keyword Usage by Claim Status")
            status_filter = st.selectbox("Select Status for Keyword Analysis", merged_df['clmStatus'].unique())

            if status_filter:
                status_data = merged_df[merged_df['clmStatus'] == status_filter]
                keyword_cols = [col for col in status_data.columns if col.endswith('_count')]
                avg_keywords = status_data[keyword_cols].mean().sort_values(ascending=False)
                st.bar_chart(avg_keywords.head(10))

with nlp_tab4:
    st.subheader("🤖 Baseline ML Model")

    if len(nlp_features_df) > 0:
        # Model training section
        st.subheader("Model Training")

        col1, col2, col3 = st.columns(3)
        with col1:
            train_model = st.button("🚀 Train Baseline Model", type="primary")
        with col2:
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        with col3:
            save_model = st.checkbox("Save Model", value=True)

        # Initialize session state for model
        if 'baseline_model' not in st.session_state:
            st.session_state.baseline_model = None
            st.session_state.training_results = None

        if train_model:
            try:
                with st.spinner("Training baseline Random Forest model..."):
                    # Initialize model
                    model = ClaimStatusBaselineModel()

                    # Train model
                    training_results = model.train(
                        nlp_features_df=nlp_features_df,
                        claims_df=df_raw_final,
                        test_size=test_size
                    )

                    # Store in session state
                    st.session_state.baseline_model = model
                    st.session_state.training_results = training_results

                    # Save model if requested
                    if save_model:
                        model_path = './_data/models/baseline_rf_model.joblib'
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        model.save_model(model_path)

                    st.success("✅ Model training completed!")

            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")

        # Display model results if available
        if st.session_state.training_results is not None:
            results = st.session_state.training_results

            # Performance metrics
            st.subheader("📊 Model Performance")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Train Accuracy", f"{results['train_accuracy']:.3f}")
            with col2:
                st.metric("Test Accuracy", f"{results['test_accuracy']:.3f}")
            with col3:
                st.metric("CV Mean", f"{results['cv_mean']:.3f}")
            with col4:
                st.metric("CV Std", f"{results['cv_std']:.3f}")

            # Performance summary table
            st.subheader("📋 Detailed Performance")
            performance_summary = create_model_performance_summary(results)
            st.dataframe(performance_summary, use_container_width=True)

            # Feature importance
            st.subheader("🎯 Feature Importance")
            top_n = st.slider("Number of top features to display", 5, 20, 10)

            col1, col2 = st.columns([1, 1])
            with col1:
                feature_importance = results['feature_importance'].head(top_n)
                st.dataframe(feature_importance, use_container_width=True)

            with col2:
                fig = plot_feature_importance(results['feature_importance'], top_n)
                st.pyplot(fig)

            # Confusion matrix
            st.subheader("🔄 Confusion Matrix")
            fig_cm = plot_confusion_matrix(results['confusion_matrix'], results['class_names'])
            st.pyplot(fig_cm)

            # Model predictions on current data
            if st.session_state.baseline_model is not None:
                st.subheader("🎯 Model Predictions")

                try:
                    predictions = st.session_state.baseline_model.predict(nlp_features_df, df_raw_final)

                    # Show prediction summary
                    pred_summary = predictions['predicted_status'].value_counts()
                    st.write("**Prediction Summary:**")
                    for status, count in pred_summary.items():
                        st.write(f"- {status}: {count} claims")

                    # Show top confident predictions
                    st.write("**Most Confident Predictions:**")
                    top_predictions = predictions.nlargest(10, 'prediction_confidence')[
                        ['clmNum', 'predicted_status', 'prediction_confidence']
                    ]
                    st.dataframe(top_predictions, use_container_width=True)

                except Exception as e:
                    st.error(f"Error making predictions: {str(e)}")

        # Load existing model option
        st.subheader("💾 Load Existing Model")
        model_path = './_data/models/baseline_rf_model.joblib'
        if os.path.exists(model_path):
            if st.button("📂 Load Saved Model"):
                try:
                    model = ClaimStatusBaselineModel()
                    model.load_model(model_path)
                    st.session_state.baseline_model = model
                    st.session_state.training_results = model.training_history
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
        else:
            st.info("No saved model found. Train a new model first.")

    else:
        st.info("No NLP features available. Please ensure notes data is loaded and processed.")

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