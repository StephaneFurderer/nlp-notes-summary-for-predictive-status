from helpers.functions.claims_utils import transform_claims_raw_data, _filter_by_, calculate_claim_features
from helpers.functions.plot_utils import plot_single_claim_lifetime
from helpers.functions.nlp_utils import ClaimNotesNLPAnalyzer, create_nlp_feature_summary, analyze_claims_with_caching
from helpers.functions.ml_models import ClaimStatusBaselineModel, ClaimStatusEnhancedModel, ClaimStatusNERModel, create_model_performance_summary, plot_feature_importance, plot_confusion_matrix, plot_prediction_comparison, plot_prediction_confidence_analysis, get_open_claims_for_prediction, plot_enhanced_feature_importance, plot_model_comparison, get_available_models
from helpers.functions.pattern_analysis import ClaimNotesPatternAnalyzer, create_pattern_visualizations
import pandas as pd
import streamlit as st
import os
import matplotlib.pyplot as plt
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
    return transform_claims_raw_data(report_date)

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
claim_filter = st.sidebar.text_input("Filter by Claim Number (optional)", placeholder="Enter claim number...", help="Enter a claim number to filter data, leave blank to show all")

st.sidebar.markdown("🔍 ML Training Data")
# Show info about training eligible claims (no INITIAL_REVIEW)
training_status_filter = df_raw_final['clmStatus'].isin(['PAID', 'DENIED', 'CLOSED'])
training_eligible_df = df_raw_final[training_status_filter]

# Calculate granular classes for display
training_reopened = training_eligible_df['dateReopened'].notna()
paid_count = (training_eligible_df['clmStatus'] == 'PAID').sum()
paid_reopened_count = ((training_eligible_df['clmStatus'] == 'PAID') & training_reopened).sum()
denied_count = (training_eligible_df['clmStatus'] == 'DENIED').sum()
denied_reopened_count = ((training_eligible_df['clmStatus'] == 'DENIED') & training_reopened).sum()
closed_count = (training_eligible_df['clmStatus'] == 'CLOSED').sum()
closed_reopened_count = ((training_eligible_df['clmStatus'] == 'CLOSED') & training_reopened).sum()

st.sidebar.info(f"""
**Training Target Classes:**
- PAID: {paid_count - paid_reopened_count:,} | PAID_REOPENED: {paid_reopened_count:,}
- DENIED: {denied_count - denied_reopened_count:,} | DENIED_REOPENED: {denied_reopened_count:,}
- CLOSED: {closed_count - closed_reopened_count:,} | CLOSED_REOPENED: {closed_reopened_count:,}

**Total Training Data:** {len(training_eligible_df):,} claims
""")

st.sidebar.markdown("🔮 Open Claims for Prediction")
# Show open claims info
open_statuses = ['OPEN', 'ESTABLISHED', 'INITIAL_REVIEW', 'FUTURE_PAY_POTENTIAL']
open_claims_df = df_raw_final[df_raw_final['clmStatus'].isin(open_statuses)]

st.sidebar.info(f"""
**Open Claims to Predict:**
- OPEN: {(open_claims_df['clmStatus'] == 'OPEN').sum():,}
- ESTABLISHED: {(open_claims_df['clmStatus'] == 'ESTABLISHED').sum():,}
- INITIAL_REVIEW: {(open_claims_df['clmStatus'] == 'INITIAL_REVIEW').sum():,}
- FUTURE_PAY_POTENTIAL: {(open_claims_df['clmStatus'] == 'FUTURE_PAY_POTENTIAL').sum():,}

**Total Open Claims:** {len(open_claims_df):,}
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
nlp_tab1, nlp_tab2, nlp_tab3, nlp_tab4, nlp_tab5, nlp_tab6, nlp_tab7 = st.tabs(["📊 Feature Summary", "🔍 Claim Analysis", "📈 Insights", "🤖 ML Baseline", "🔬 Pattern Analysis", "🚀 Enhanced Model", "🎯 NER Model"])

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

            # Prediction comparison visualization
            st.subheader("📊 Prediction Analysis")

            # Get test set predictions for comparison
            if 'test_actual' in results and 'test_predicted' in results:
                test_actual = results['test_actual']
                test_predicted = results['test_predicted']
            else:
                # If not stored, we'll skip this visualization
                test_actual = None
                test_predicted = None

            # Get open claims for prediction
            open_claims_df = get_open_claims_for_prediction(df_raw_final)

            if len(open_claims_df) > 0:
                st.write(f"**Found {len(open_claims_df)} open claims for prediction**")

                # Get NLP features for open claims
                open_nlp_features = nlp_features_df[nlp_features_df['clmNum'].isin(open_claims_df['clmNum'])]

                if len(open_nlp_features) > 0:
                    try:
                        # Make predictions on open claims
                        open_predictions = st.session_state.baseline_model.predict(open_nlp_features, open_claims_df)

                        # Merge with claim information
                        open_results = open_predictions.merge(
                            open_claims_df[['clmNum', 'clmStatus', 'clmCause']],
                            on='clmNum',
                            how='left'
                        )

                        # Create comparison visualization
                        if test_actual is not None and test_predicted is not None:
                            fig_comparison = plot_prediction_comparison(
                                test_actual,
                                test_predicted,
                                open_predictions['predicted_status']
                            )
                        else:
                            # Create simplified visualization with just open predictions
                            open_pred_counts = open_predictions['predicted_status'].value_counts()
                            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                            ax.pie(open_pred_counts.values, labels=open_pred_counts.index, autopct='%1.1f%%')
                            ax.set_title('Open Claims - Predicted Status Distribution')
                            fig_comparison = fig

                        st.pyplot(fig_comparison)

                        # Confidence analysis
                        st.subheader("🎯 Prediction Confidence Analysis")
                        fig_confidence = plot_prediction_confidence_analysis(open_predictions)
                        st.pyplot(fig_confidence)

                        # Display detailed predictions
                        st.subheader("📋 Open Claims Predictions")

                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            avg_confidence = open_predictions['prediction_confidence'].mean()
                            st.metric("Average Confidence", f"{avg_confidence:.2f}")
                        with col2:
                            high_conf_count = (open_predictions['prediction_confidence'] > 0.8).sum()
                            st.metric("High Confidence (>80%)", high_conf_count)
                        with col3:
                            pred_paid = (open_predictions['predicted_status'] == 'PAID').sum()
                            st.metric("Predicted PAID", pred_paid)
                        with col4:
                            pred_denied = (open_predictions['predicted_status'] == 'DENIED').sum()
                            st.metric("Predicted DENIED", pred_denied)

                        # Detailed predictions table (include all 6 target classes)
                        display_cols = ['clmNum', 'clmStatus', 'clmCause', 'predicted_status',
                                       'prediction_confidence', 'prob_PAID', 'prob_PAID_REOPENED',
                                       'prob_DENIED', 'prob_DENIED_REOPENED', 'prob_CLOSED', 'prob_CLOSED_REOPENED']
                        available_cols = [col for col in display_cols if col in open_results.columns]

                        # Sort by confidence descending
                        open_results_sorted = open_results.sort_values('prediction_confidence', ascending=False)

                        st.dataframe(
                            open_results_sorted[available_cols],
                            use_container_width=True
                        )

                        # Download option for predictions
                        csv = open_results_sorted.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Open Claims Predictions",
                            data=csv,
                            file_name=f"open_claims_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"Error predicting open claims: {str(e)}")
                else:
                    st.warning("No NLP features found for open claims")
            else:
                st.info("No open claims found for prediction")

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

                    # Show top confident predictions (include all probability classes)
                    st.write("**Most Confident Predictions:**")
                    prob_cols = ['clmNum', 'predicted_status', 'prediction_confidence']

                    # Add all available probability columns
                    all_prob_cols = ['prob_PAID', 'prob_PAID_REOPENED', 'prob_DENIED',
                                   'prob_DENIED_REOPENED', 'prob_CLOSED', 'prob_CLOSED_REOPENED']
                    available_prob_cols = [col for col in all_prob_cols if col in predictions.columns]

                    display_cols = prob_cols + available_prob_cols
                    top_predictions = predictions.nlargest(10, 'prediction_confidence')[display_cols]
                    st.dataframe(top_predictions, use_container_width=True)

                except Exception as e:
                    st.error(f"Error making predictions: {str(e)}")

        # Load existing model option
        st.subheader("💾 Load Existing Baseline Model")

        # Show available baseline models
        available_models = get_available_models("baseline")

        if len(available_models) > 0 and 'Message' not in available_models.columns:
            st.write("**Available Baseline Models:**")
            display_cols = ['Filename', 'Type', 'Test Accuracy', 'CV Accuracy', 'Features', 'File Size (MB)', 'Modified']
            display_models = available_models[display_cols]
            st.dataframe(display_models, use_container_width=True)

            # Model selection
            model_options = available_models['Filename'].tolist()
            selected_model = st.selectbox("Select Model to Load:", model_options)

            if st.button("📂 Load Selected Baseline Model"):
                try:
                    selected_path = available_models[available_models['Filename'] == selected_model]['Full Path'].iloc[0]
                    model = ClaimStatusBaselineModel()
                    model.load_model(selected_path)
                    st.session_state.baseline_model = model
                    st.session_state.training_results = model.training_history
                    st.success(f"✅ Baseline model '{selected_model}' loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
        else:
            st.info("No saved baseline models found. Train a new model first.")

    else:
        st.info("No NLP features available. Please ensure notes data is loaded and processed.")

with nlp_tab5:
    st.subheader("🔬 Phase 1: Pattern Analysis for Enhanced Keywords")

    if len(notes_df) > 0:
        # Pattern analysis section
        st.markdown("""
        **Objective**: Discover data-driven patterns in notes to enhance our keyword system.

        This analysis will:
        - Identify most frequent words by claim status
        - Find discriminative terms that predict outcomes
        - Discover n-gram patterns and phrases
        - Calculate TF-IDF features for each status
        - Analyze temporal patterns in note language
        """)

        # Analysis controls
        col1, col2 = st.columns(2)
        with col1:
            run_analysis = st.button("🚀 Run Pattern Analysis", type="primary")
        with col2:
            max_features = st.slider("Max TF-IDF Features", 50, 200, 100, 25)

        # Initialize session state for pattern analysis
        if 'pattern_analysis_report' not in st.session_state:
            st.session_state.pattern_analysis_report = None

        if run_analysis:
            try:
                with st.spinner("🔍 Analyzing note patterns across all claim statuses..."):
                    # Initialize pattern analyzer
                    analyzer = ClaimNotesPatternAnalyzer()

                    # Generate comprehensive report
                    report = analyzer.generate_comprehensive_report(notes_df, df_raw_final)

                    # Store in session state
                    st.session_state.pattern_analysis_report = report

                    st.success("✅ Pattern analysis completed!")

            except Exception as e:
                st.error(f"❌ Error during pattern analysis: {str(e)}")

        # Display results if available
        if st.session_state.pattern_analysis_report is not None:
            report = st.session_state.pattern_analysis_report

            # Summary metrics
            st.subheader("📊 Analysis Summary")
            summary = report['summary']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Notes", f"{summary['total_notes']:,}")
            with col2:
                st.metric("Unique Claims", f"{summary['unique_claims']:,}")
            with col3:
                st.metric("Status Groups", len(summary['status_groups']))
            with col4:
                st.metric("Analysis Date", summary['analysis_timestamp'].strftime('%Y-%m-%d'))

            # Status group breakdown
            st.subheader("📋 Claims by Status")
            status_breakdown = pd.DataFrame([
                {'Status': status, 'Claim Count': count}
                for status, count in summary['status_groups'].items()
            ])
            st.dataframe(status_breakdown, use_container_width=True)

            # Discriminative terms analysis
            st.subheader("🎯 Most Discriminative Terms by Status")

            if 'discriminative_terms' in report and report['discriminative_terms']:
                # Create visualization
                try:
                    fig = create_pattern_visualizations(report)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not create visualization: {str(e)}")

                # Show detailed discriminative terms
                st.subheader("📝 Detailed Discriminative Terms")
                selected_status = st.selectbox(
                    "Select status to view discriminative terms:",
                    list(report['discriminative_terms'].keys())
                )

                if selected_status and selected_status in report['discriminative_terms']:
                    terms = report['discriminative_terms'][selected_status]

                    if terms:
                        terms_df = pd.DataFrame([
                            {
                                'Term': term[0],
                                'Enrichment Factor': f"{term[1]['enrichment']:.2f}",
                                'Count in Status': term[1]['target_count'],
                                'Rate in Status': f"{term[1]['target_rate']:.4f}",
                                'Count in Others': term[1]['other_count'],
                                'Rate in Others': f"{term[1]['other_rate']:.4f}"
                            }
                            for term in terms[:20]  # Top 20
                        ])
                        st.dataframe(terms_df, use_container_width=True)
                    else:
                        st.info(f"No discriminative terms found for {selected_status}")

            # TF-IDF Analysis
            st.subheader("📈 TF-IDF Analysis")
            if 'tfidf_analysis' in report and 'tfidf_scores' in report['tfidf_analysis']:
                tfidf_data = report['tfidf_analysis']['tfidf_scores']

                # Show TF-IDF scores by status
                tfidf_status = st.selectbox(
                    "Select status for TF-IDF analysis:",
                    list(tfidf_data.keys())
                )

                if tfidf_status and tfidf_status in tfidf_data:
                    tfidf_terms = tfidf_data[tfidf_status][:15]  # Top 15

                    if tfidf_terms:
                        tfidf_df = pd.DataFrame([
                            {'Term': term[0], 'TF-IDF Score': f"{term[1]:.4f}"}
                            for term in tfidf_terms
                        ])
                        st.dataframe(tfidf_df, use_container_width=True)
                    else:
                        st.info(f"No TF-IDF terms found for {tfidf_status}")

            # N-gram patterns
            st.subheader("🔤 N-gram Patterns")
            if 'ngram_patterns' in report:
                ngram_status = st.selectbox(
                    "Select status for n-gram analysis:",
                    list(report['ngram_patterns'].keys())
                )

                if ngram_status and ngram_status in report['ngram_patterns']:
                    ngram_data = report['ngram_patterns'][ngram_status]

                    for ngram_type in ['2gram', '3gram']:
                        if ngram_type in ngram_data and ngram_data[ngram_type]:
                            st.write(f"**{ngram_type.upper()} Patterns:**")

                            # Sort by frequency and show top patterns
                            sorted_patterns = sorted(
                                ngram_data[ngram_type].items(),
                                key=lambda x: x[1],
                                reverse=True
                            )[:10]

                            patterns_df = pd.DataFrame([
                                {'Pattern': pattern, 'Frequency': freq}
                                for pattern, freq in sorted_patterns
                            ])
                            st.dataframe(patterns_df, use_container_width=True)

            # Download report option
            st.subheader("📥 Export Analysis")
            if st.button("💾 Prepare Download Report"):
                # Create simplified report for download
                download_data = {
                    'summary': summary,
                    'discriminative_terms': {
                        status: [{'term': t[0], 'enrichment': t[1]['enrichment'], 'count': t[1]['target_count']}
                                for t in terms[:10]]
                        for status, terms in report['discriminative_terms'].items()
                    }
                }

                import json
                report_json = json.dumps(download_data, indent=2, default=str)
                st.download_button(
                    label="📥 Download Pattern Analysis Report",
                    data=report_json,
                    file_name=f"pattern_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )

    else:
        st.info("No notes data available. Please ensure notes are loaded.")

with nlp_tab6:
    st.subheader("🚀 Enhanced Model with TF-IDF Features")

    if len(nlp_features_df) > 0 and len(notes_df) > 0:
        # Enhanced model training section
        st.markdown("""
        **Enhanced Model**: Combines baseline NLP features with TF-IDF features discovered through pattern analysis.

        **Key Improvements**:
        - Incorporates discriminative terms from pattern analysis
        - Uses TF-IDF vectorization for better text representation
        - Enhanced feature set for improved prediction accuracy
        """)

        # Model training controls
        st.subheader("Enhanced Model Training")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            train_enhanced = st.button("🚀 Train Enhanced Model", type="primary")
        with col2:
            test_size_enhanced = st.slider("Test Set Size (Enhanced)", 0.1, 0.4, 0.2, 0.05, key="enhanced_test_size")
        with col3:
            max_tfidf_features = st.slider("Max TF-IDF Features", 25, 100, 50, 5)
        with col4:
            save_enhanced = st.checkbox("Save Enhanced Model", value=True)

        # Initialize session state for enhanced model
        if 'enhanced_model' not in st.session_state:
            st.session_state.enhanced_model = None
            st.session_state.enhanced_results = None

        if train_enhanced:
            try:
                with st.spinner("Training Enhanced Random Forest model with TF-IDF features..."):
                    # Initialize enhanced model
                    enhanced_model = ClaimStatusEnhancedModel(max_tfidf_features=max_tfidf_features)

                    # Train enhanced model
                    enhanced_results = enhanced_model.train_enhanced(
                        nlp_features_df=nlp_features_df,
                        claims_df=df_raw_final,
                        notes_df=notes_df,
                        test_size=test_size_enhanced
                    )

                    # Store in session state
                    st.session_state.enhanced_model = enhanced_model
                    st.session_state.enhanced_results = enhanced_results

                    # Save model if requested
                    if save_enhanced:
                        model_path = './_data/models/enhanced_rf_model.joblib'
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        enhanced_model.save_model(model_path)

                    st.success("✅ Enhanced model training completed!")

            except Exception as e:
                st.error(f"❌ Error training enhanced model: {str(e)}")
                st.exception(e)

        # Display enhanced model results
        if st.session_state.enhanced_results is not None:
            results = st.session_state.enhanced_results

            # Performance metrics
            st.subheader("📊 Enhanced Model Performance")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Train Accuracy", f"{results['train_accuracy']:.3f}")
            with col2:
                st.metric("Test Accuracy", f"{results['test_accuracy']:.3f}")
            with col3:
                st.metric("CV Mean", f"{results['cv_mean']:.3f}")
            with col4:
                st.metric("Total Features", results['n_features'])
            with col5:
                st.metric("TF-IDF Features", results['n_tfidf_features'])

            # Performance comparison with baseline (if available)
            if st.session_state.training_results is not None:
                st.subheader("📈 Model Comparison")
                baseline_results = st.session_state.training_results

                # Create comparison visualization
                try:
                    fig_comparison = plot_model_comparison(baseline_results, results)
                    st.pyplot(fig_comparison)
                except Exception as e:
                    st.warning(f"Could not create comparison plot: {str(e)}")

                # Comparison metrics table
                comparison_data = {
                    'Metric': ['Train Accuracy', 'Test Accuracy', 'CV Accuracy', 'Number of Features'],
                    'Baseline': [
                        f"{baseline_results['train_accuracy']:.3f}",
                        f"{baseline_results['test_accuracy']:.3f}",
                        f"{baseline_results['cv_mean']:.3f}",
                        baseline_results['n_features']
                    ],
                    'Enhanced': [
                        f"{results['train_accuracy']:.3f}",
                        f"{results['test_accuracy']:.3f}",
                        f"{results['cv_mean']:.3f}",
                        results['n_features']
                    ],
                    'Improvement': [
                        f"{(results['train_accuracy'] - baseline_results['train_accuracy']):.3f}",
                        f"{(results['test_accuracy'] - baseline_results['test_accuracy']):.3f}",
                        f"{(results['cv_mean'] - baseline_results['cv_mean']):.3f}",
                        f"+{results['n_features'] - baseline_results['n_features']}"
                    ]
                }
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)

            # Enhanced feature importance analysis
            st.subheader("🎯 Enhanced Feature Importance")
            top_n_enhanced = st.slider("Number of top features to display", 5, 30, 15, key="enhanced_top_n")

            try:
                fig_enhanced = plot_enhanced_feature_importance(results, top_n_enhanced)
                st.pyplot(fig_enhanced)
            except Exception as e:
                st.warning(f"Could not create enhanced feature importance plot: {str(e)}")

            # Show detailed feature importance tables
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Top Base Features:**")
                if 'base_importance' in results and len(results['base_importance']) > 0:
                    st.dataframe(results['base_importance'], use_container_width=True)
                else:
                    st.info("No base features in top importance")

            with col2:
                st.write("**Top TF-IDF Features:**")
                if 'tfidf_importance' in results and len(results['tfidf_importance']) > 0:
                    # Clean feature names for display
                    tfidf_display = results['tfidf_importance'].copy()
                    tfidf_display['clean_feature'] = tfidf_display['feature'].str.replace('tfidf_', '')
                    tfidf_display = tfidf_display[['clean_feature', 'importance']].rename(columns={'clean_feature': 'feature'})
                    st.dataframe(tfidf_display, use_container_width=True)
                else:
                    st.info("No TF-IDF features in top importance")

            # Enhanced model predictions
            if st.session_state.enhanced_model is not None:
                st.subheader("🎯 Enhanced Model Predictions")

                # Predictions on open claims
                st.write("**Open Claims Predictions:**")
                open_claims_df = get_open_claims_for_prediction(df_raw_final)

                if len(open_claims_df) > 0:
                    # Get NLP features for open claims
                    open_nlp_features = nlp_features_df[nlp_features_df['clmNum'].isin(open_claims_df['clmNum'])]
                    # Get notes for open claims
                    open_notes = notes_df[notes_df['clmNum'].isin(open_claims_df['clmNum'])]

                    if len(open_nlp_features) > 0 and len(open_notes) > 0:
                        try:
                            # Make enhanced predictions
                            enhanced_predictions = st.session_state.enhanced_model.predict_enhanced(
                                open_nlp_features, open_notes, open_claims_df
                            )

                            # Show prediction summary
                            pred_summary = enhanced_predictions['predicted_status'].value_counts()
                            st.write("**Enhanced Prediction Summary:**")
                            for status, count in pred_summary.items():
                                st.write(f"- {status}: {count} claims")

                            # Show confidence analysis
                            avg_confidence = enhanced_predictions['prediction_confidence'].mean()
                            high_conf_count = len(enhanced_predictions[enhanced_predictions['prediction_confidence'] > 0.8])

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Average Confidence", f"{avg_confidence:.3f}")
                            with col2:
                                st.metric("High Confidence (>80%)", f"{high_conf_count}/{len(enhanced_predictions)}")

                            # Show most confident predictions
                            st.write("**Most Confident Enhanced Predictions:**")
                            prob_cols = ['clmNum', 'predicted_status', 'prediction_confidence']
                            all_prob_cols = ['prob_PAID', 'prob_PAID_REOPENED', 'prob_DENIED',
                                           'prob_DENIED_REOPENED', 'prob_CLOSED', 'prob_CLOSED_REOPENED']
                            available_prob_cols = [col for col in all_prob_cols if col in enhanced_predictions.columns]

                            display_cols = prob_cols + available_prob_cols
                            top_enhanced_predictions = enhanced_predictions.nlargest(10, 'prediction_confidence')[display_cols]
                            st.dataframe(top_enhanced_predictions, use_container_width=True)

                        except Exception as e:
                            st.error(f"Error making enhanced predictions: {str(e)}")
                    else:
                        st.warning("No NLP features or notes found for open claims")
                else:
                    st.info("No open claims found for prediction")

        # Load existing enhanced model option
        st.subheader("💾 Load Existing Enhanced Model")

        # Show available enhanced models
        available_enhanced_models = get_available_models("enhanced")

        if len(available_enhanced_models) > 0 and 'Message' not in available_enhanced_models.columns:
            st.write("**Available Enhanced Models:**")
            display_cols = ['Filename', 'Type', 'Test Accuracy', 'CV Accuracy', 'Features', 'TF-IDF Features', 'File Size (MB)', 'Modified']
            # Only show TF-IDF Features column if it exists
            available_cols = [col for col in display_cols if col in available_enhanced_models.columns]
            display_enhanced = available_enhanced_models[available_cols]
            st.dataframe(display_enhanced, use_container_width=True)

            # Model selection
            enhanced_options = available_enhanced_models['Filename'].tolist()
            selected_enhanced = st.selectbox("Select Enhanced Model to Load:", enhanced_options)

            if st.button("📂 Load Selected Enhanced Model"):
                try:
                    selected_path = available_enhanced_models[available_enhanced_models['Filename'] == selected_enhanced]['Full Path'].iloc[0]
                    enhanced_model = ClaimStatusEnhancedModel()
                    enhanced_model.load_model(selected_path)
                    st.session_state.enhanced_model = enhanced_model
                    st.session_state.enhanced_results = enhanced_model.training_history
                    st.success(f"✅ Enhanced model '{selected_enhanced}' loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading enhanced model: {str(e)}")
        else:
            st.info("No saved enhanced models found. Train a new enhanced model first.")

    else:
        if len(nlp_features_df) == 0:
            st.info("No NLP features available. Please ensure notes data is loaded and processed.")
        if len(notes_df) == 0:
            st.info("No notes data available. Please ensure notes are loaded.")

with nlp_tab7:
    st.subheader("🎯 NER Model with Named Entity Recognition")

    if len(nlp_features_df) > 0 and len(notes_df) > 0:
        # NER model description
        st.markdown("""
        **NER Model**: Builds on the Enhanced Model by adding Named Entity Recognition features.

        **NER Features Include**:
        - **Person entities**: Names mentioned in notes
        - **Organizations**: Companies, institutions referenced
        - **Locations**: Places, cities, states mentioned
        - **Money amounts**: Financial figures detected
        - **Dates/Times**: Temporal references
        - **Facilities**: Buildings, properties mentioned
        - **Entity metrics**: Density, diversity, uniqueness

        **Total Feature Stack**: Base NLP + Financial + TF-IDF + NER
        """)

        # Check spaCy installation
        try:
            import spacy
            spacy_available = True
            # Try to detect available models
            available_models = []
            for model_name in ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']:
                try:
                    spacy.load(model_name)
                    available_models.append(model_name)
                except OSError:
                    continue

            if available_models:
                st.success(f"✅ spaCy available with models: {', '.join(available_models)}")
            else:
                st.warning("⚠️ spaCy installed but no English models found. Install with: `python -m spacy download en_core_web_sm`")
                spacy_available = False
        except ImportError:
            st.error("❌ spaCy not installed. Install with: `pip install spacy`")
            spacy_available = False

        if spacy_available:
            # NER model training controls
            st.subheader("NER Model Training")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                train_ner = st.button("🎯 Train NER Model", type="primary")
            with col2:
                test_size_ner = st.slider("Test Set Size (NER)", 0.1, 0.4, 0.2, 0.05, key="ner_test_size")
            with col3:
                max_tfidf_ner = st.slider("Max TF-IDF Features (NER)", 25, 100, 50, 5, key="ner_tfidf")
            with col4:
                save_ner = st.checkbox("Save NER Model", value=True)

            # Initialize session state for NER model
            if 'ner_model' not in st.session_state:
                st.session_state.ner_model = None
                st.session_state.ner_results = None

            if train_ner:
                try:
                    st.info("🎯 Starting NER model training (this may take several minutes due to NER processing)...")

                    # Create progress containers
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def ner_progress_callback(percent, message):
                        progress_bar.progress(percent)
                        status_text.text(message)

                    # Initialize NER model
                    ner_model = ClaimStatusNERModel(max_tfidf_features=max_tfidf_ner)

                    # Train NER model with progress tracking
                    ner_results = ner_model.train_ner_enhanced(
                        nlp_features_df=nlp_features_df,
                        claims_df=df_raw_final,
                        notes_df=notes_df,
                        test_size=test_size_ner,
                        progress_callback=ner_progress_callback
                    )

                    # Store in session state
                    st.session_state.ner_model = ner_model
                    st.session_state.ner_results = ner_results

                    # Save model if requested
                    if save_ner:
                        status_text.text("💾 Saving NER model...")
                        progress_bar.progress(0.95)
                        model_path = './_data/models/ner_rf_model.joblib'
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        ner_model.save_model(model_path)

                    # Clear progress indicators
                    progress_bar.progress(1.0)
                    status_text.text("✅ NER model training completed!")

                    # Clean up progress display after a moment
                    import time
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()

                    st.success("✅ NER model training completed successfully!")

                except Exception as e:
                    st.error(f"❌ Error training NER model: {str(e)}")
                    st.exception(e)

            # Display NER model results
            if st.session_state.ner_results is not None:
                results = st.session_state.ner_results

                # Performance metrics
                st.subheader("📊 NER Model Performance")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("Train Accuracy", f"{results['train_accuracy']:.3f}")
                with col2:
                    st.metric("Test Accuracy", f"{results['test_accuracy']:.3f}")
                with col3:
                    st.metric("CV Mean", f"{results['cv_mean']:.3f}")
                with col4:
                    st.metric("Total Features", results['n_features'])
                with col5:
                    st.metric("TF-IDF Features", results['n_tfidf_features'])
                with col6:
                    st.metric("NER Features", results['n_ner_features'])

                # Three-way model comparison (if available)
                if st.session_state.training_results is not None and st.session_state.enhanced_results is not None:
                    st.subheader("📈 Three-Way Model Comparison")

                    # Comparison metrics table
                    baseline_results = st.session_state.training_results
                    enhanced_results = st.session_state.enhanced_results

                    comparison_data = {
                        'Metric': ['Train Accuracy', 'Test Accuracy', 'CV Accuracy', 'Total Features', 'Base Features', 'TF-IDF Features', 'NER Features'],
                        'Baseline': [
                            f"{baseline_results['train_accuracy']:.3f}",
                            f"{baseline_results['test_accuracy']:.3f}",
                            f"{baseline_results['cv_mean']:.3f}",
                            baseline_results['n_features'],
                            baseline_results['n_features'],
                            0,
                            0
                        ],
                        'Enhanced': [
                            f"{enhanced_results['train_accuracy']:.3f}",
                            f"{enhanced_results['test_accuracy']:.3f}",
                            f"{enhanced_results['cv_mean']:.3f}",
                            enhanced_results['n_features'],
                            enhanced_results['n_features'] - enhanced_results['n_tfidf_features'],
                            enhanced_results['n_tfidf_features'],
                            0
                        ],
                        'NER': [
                            f"{results['train_accuracy']:.3f}",
                            f"{results['test_accuracy']:.3f}",
                            f"{results['cv_mean']:.3f}",
                            results['n_features'],
                            results['n_base_features'],
                            results['n_tfidf_features'],
                            results['n_ner_features']
                        ],
                        'NER vs Enhanced': [
                            f"{(results['train_accuracy'] - enhanced_results['train_accuracy']):.3f}",
                            f"{(results['test_accuracy'] - enhanced_results['test_accuracy']):.3f}",
                            f"{(results['cv_mean'] - enhanced_results['cv_mean']):.3f}",
                            f"+{results['n_features'] - enhanced_results['n_features']}",
                            "0",
                            "0",
                            f"+{results['n_ner_features']}"
                        ]
                    }
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)

                # NER feature importance analysis
                st.subheader("🎯 NER Feature Importance Breakdown")

                # Show feature importance by category
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**Top Base Features:**")
                    if 'base_importance' in results and len(results['base_importance']) > 0:
                        st.dataframe(results['base_importance'], use_container_width=True)
                    else:
                        st.info("No base features in top importance")

                with col2:
                    st.write("**Top TF-IDF Features:**")
                    if 'tfidf_importance' in results and len(results['tfidf_importance']) > 0:
                        tfidf_display = results['tfidf_importance'].copy()
                        tfidf_display['clean_feature'] = tfidf_display['feature'].str.replace('tfidf_', '')
                        tfidf_display = tfidf_display[['clean_feature', 'importance']].rename(columns={'clean_feature': 'feature'})
                        st.dataframe(tfidf_display, use_container_width=True)
                    else:
                        st.info("No TF-IDF features in top importance")

                with col3:
                    st.write("**Top NER Features:**")
                    if 'ner_importance' in results and len(results['ner_importance']) > 0:
                        st.dataframe(results['ner_importance'], use_container_width=True)
                    else:
                        st.info("No NER features in top importance")

                # NER model predictions
                if st.session_state.ner_model is not None:
                    st.subheader("🎯 NER Model Predictions")

                    # Predictions on open claims
                    st.write("**Open Claims NER Predictions:**")
                    open_claims_df = get_open_claims_for_prediction(df_raw_final)

                    if len(open_claims_df) > 0:
                        # Get features for open claims
                        open_nlp_features = nlp_features_df[nlp_features_df['clmNum'].isin(open_claims_df['clmNum'])]
                        open_notes = notes_df[notes_df['clmNum'].isin(open_claims_df['clmNum'])]

                        if len(open_nlp_features) > 0 and len(open_notes) > 0:
                            try:
                                # Make NER predictions
                                ner_predictions = st.session_state.ner_model.predict_ner_enhanced(
                                    open_nlp_features, open_notes, open_claims_df
                                )

                                # Show prediction summary
                                pred_summary = ner_predictions['predicted_status'].value_counts()
                                st.write("**NER Prediction Summary:**")
                                for status, count in pred_summary.items():
                                    st.write(f"- {status}: {count} claims")

                                # Show confidence analysis
                                avg_confidence = ner_predictions['prediction_confidence'].mean()
                                high_conf_count = len(ner_predictions[ner_predictions['prediction_confidence'] > 0.8])

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Average Confidence", f"{avg_confidence:.3f}")
                                with col2:
                                    st.metric("High Confidence (>80%)", f"{high_conf_count}/{len(ner_predictions)}")

                                # Show most confident predictions
                                st.write("**Most Confident NER Predictions:**")
                                prob_cols = ['clmNum', 'predicted_status', 'prediction_confidence']
                                all_prob_cols = ['prob_PAID', 'prob_PAID_REOPENED', 'prob_DENIED',
                                               'prob_DENIED_REOPENED', 'prob_CLOSED', 'prob_CLOSED_REOPENED']
                                available_prob_cols = [col for col in all_prob_cols if col in ner_predictions.columns]

                                display_cols = prob_cols + available_prob_cols
                                top_ner_predictions = ner_predictions.nlargest(10, 'prediction_confidence')[display_cols]
                                st.dataframe(top_ner_predictions, use_container_width=True)

                            except Exception as e:
                                st.error(f"Error making NER predictions: {str(e)}")
                                st.exception(e)
                        else:
                            st.warning("No NLP features or notes found for open claims")
                    else:
                        st.info("No open claims found for prediction")

            # Load existing NER model option
            st.subheader("💾 Load Existing NER Model")

            # Show available NER models
            available_ner_models = get_available_models("ner")

            if len(available_ner_models) > 0 and 'Message' not in available_ner_models.columns:
                st.write("**Available NER Models:**")
                display_cols = ['Filename', 'Type', 'Test Accuracy', 'CV Accuracy', 'Features', 'TF-IDF Features', 'NER Features', 'File Size (MB)', 'Modified']
                # Only show columns that exist
                available_cols = [col for col in display_cols if col in available_ner_models.columns]
                display_ner = available_ner_models[available_cols]
                st.dataframe(display_ner, use_container_width=True)

                # Model selection
                ner_options = available_ner_models['Filename'].tolist()
                selected_ner = st.selectbox("Select NER Model to Load:", ner_options)

                if st.button("📂 Load Selected NER Model"):
                    try:
                        selected_path = available_ner_models[available_ner_models['Filename'] == selected_ner]['Full Path'].iloc[0]
                        ner_model = ClaimStatusNERModel()
                        ner_model.load_model(selected_path)
                        st.session_state.ner_model = ner_model
                        st.session_state.ner_results = ner_model.training_history
                        st.success(f"✅ NER model '{selected_ner}' loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading NER model: {str(e)}")
            else:
                st.info("No saved NER models found. Train a new NER model first.")

        else:
            st.warning("⚠️ spaCy is required for NER features. Please install spaCy and an English model.")
            st.code("pip install spacy\npython -m spacy download en_core_web_sm")

    else:
        if len(nlp_features_df) == 0:
            st.info("No NLP features available. Please ensure notes data is loaded and processed.")
        if len(notes_df) == 0:
            st.info("No notes data available. Please ensure notes are loaded.")

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