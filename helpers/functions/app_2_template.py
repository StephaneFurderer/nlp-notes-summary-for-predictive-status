"""
Common Template for Standardized Claims Analysis Apps

This template provides the core functionality for both the real data app and demo app,
avoiding code duplication while allowing customization for different data sources.
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Import our standardization functions
from helpers.functions.standardized_claims_transformer import StandardizedClaimsTransformer
from helpers.functions.standardized_claims_schema import StandardizationConfig
from helpers.functions.load_cache_data import CacheManager, DataOrganizer, DataLoader


class ClaimsAnalysisTemplate:
    """
    Template class for standardized claims analysis apps.
    Handles the common UI and logic, while allowing customization for data loading.
    """
    
    def __init__(self, app_title: str, is_demo: bool = False):
        self.app_title = app_title
        self.is_demo = is_demo
        
        # Initialize data management components
        self.cache_manager = CacheManager()
        self.data_organizer = DataOrganizer()
        self.data_loader = DataLoader()
        
        self.transformer = StandardizedClaimsTransformer()
        
    def setup_page_config(self, page_title: str, page_icon: str = "📊"):
        """Set up Streamlit page configuration."""
        st.set_page_config(
            page_title=page_title,
            page_icon=page_icon,
            layout="wide"
        )
        
    def render_header(self):
        """Render the app header."""
        st.title(f"📊 {self.app_title}")
        st.markdown("**Micro-Level Reserving Framework - 30-Day Period Standardization**")
        
        if self.is_demo:
            st.info("🔬 This is a demo version using synthetic data for demonstration purposes")
    
    def render_sidebar_controls(self, df_raw_txn: pd.DataFrame) -> Tuple[str, int, int, int, str, str]:
        """Render sidebar controls and return configuration values."""
        st.sidebar.markdown("## 🔍 Claim Selection")
        claim_filter = st.sidebar.text_input(
            "Enter Claim Number", 
            placeholder="e.g., CLM001" if self.is_demo else "e.g., CLM123456", 
            help="Enter a claim number to analyze"
        )
        
        st.sidebar.markdown("## 🎯 Data Filters")
        
        # Claim Cause filter
        available_causes = sorted(df_raw_txn['clmCause'].dropna().unique()) if 'clmCause' in df_raw_txn.columns else []
        if available_causes:
            clm_cause_filter = st.sidebar.selectbox(
                "Filter by Claim Cause",
                options=["All"] + available_causes,
                help="Filter claims by their cause"
            )
        else:
            clm_cause_filter = "All"
        
        # Claim Status filter
        available_statuses = sorted(df_raw_txn['clmStatus'].dropna().unique()) if 'clmStatus' in df_raw_txn.columns else []
        if available_statuses:
            clm_status_filter = st.sidebar.selectbox(
                "Filter by Claim Status",
                options=["All"] + available_statuses,
                help="Filter claims by their current status"
            )
        else:
            clm_status_filter = "All"
        
        st.sidebar.markdown("## ⚙️ Standardization Settings")
        period_length = st.sidebar.slider("Period Length (days)", 15, 60, 30, 5)
        max_periods = st.sidebar.slider("Max Periods to Track", 10, 120, 60, 5)
        min_periods = st.sidebar.slider("Min Periods Required", 1, 10, 1, 1)
        
        return claim_filter, period_length, max_periods, min_periods, clm_cause_filter, clm_status_filter
    
    def apply_filters(self, df_raw_txn: pd.DataFrame, clm_cause_filter: str, clm_status_filter: str) -> pd.DataFrame:
        """Apply filters to the raw dataframe."""
        filtered_df = df_raw_txn.copy()
        
        # Apply claim cause filter
        if clm_cause_filter != "All" and 'clmCause' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['clmCause'] == clm_cause_filter]
        
        # Apply claim status filter
        if clm_status_filter != "All" and 'clmStatus' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['clmStatus'] == clm_status_filter]
        
        return filtered_df
    
    def create_config(self, period_length: int, max_periods: int, min_periods: int) -> StandardizationConfig:
        """Create standardization configuration."""
        return StandardizationConfig(
            period_length_days=period_length,
            max_periods=max_periods,
            min_periods=min_periods
        )
    
    def render_original_data_section(self, df_raw_txn_filtered: pd.DataFrame, claim_filter: str, 
                                   config: StandardizationConfig) -> Optional[Any]:
        """Render the original transaction data section with expander."""
        with st.expander("📊 View Raw Transaction Data and Standardized Periods", expanded=False):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                self._render_original_transactions(df_raw_txn_filtered, claim_filter)
            
            with col2:
                standardized_claim = self._render_standardized_data(df_raw_txn_filtered, config, claim_filter)
                return standardized_claim
        return None
    
    def _render_original_transactions(self, df_raw_txn_filtered: pd.DataFrame, claim_filter: str):
        """Render original transaction data."""
        st.subheader("📋 Original Transaction Data")
        
        # Apply normalization if available
        df_display_data = df_raw_txn_filtered.copy()
        if hasattr(self, 'transformer') and self.transformer.normalization_computed:
            df_display_data = self.transformer.apply_normalization(df_display_data)
        
        # Display original transactions (including normalized columns if available)
        display_cols = ['datetxn', 'paid', 'paid_normalized', 'expense', 'expense_normalized', 'recovery', 'reserve', 'paid_cumsum', 'expense_cumsum', 'incurred_cumsum']
        available_cols = [col for col in display_cols if col in df_display_data.columns]
        
        # Sort by date
        df_display = df_display_data[available_cols].sort_values('datetxn').copy()
        st.write(f"**Found {len(df_display)} transactions for claim {claim_filter}**")
        
        # Handle data types robustly to avoid JSON serialization issues
        for col in df_display.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df_display[col]):
                    # Handle datetime columns
                    df_display[col] = df_display[col].dt.strftime('%Y-%m-%d')
                elif pd.api.types.is_numeric_dtype(df_display[col]):
                    # Handle numeric columns
                    df_display[col] = df_display[col].astype(str)
                else:
                    # Handle all other types (object, bool, etc.)
                    df_display[col] = df_display[col].astype(str)
            except Exception as e:
                # Fallback: convert to string regardless of type
                df_display[col] = df_display[col].astype(str)
        
        # Double-check: ensure all columns are strings
        for col in df_display.columns:
            df_display[col] = df_display[col].astype(str)
        
        st.dataframe(df_display, use_container_width=True)
        
        # Show final claim status using pre-computed final data
        st.subheader("📊 Final Claim Status")
        if not df_raw_txn_filtered.empty:
            # Use the claim_filter to get the specific claim from pre-computed final data
            # This avoids reconstructing and data type issues
            if hasattr(self, 'df_raw_final') and self.df_raw_final is not None:
                final_claim = self.df_raw_final[self.df_raw_final['clmNum'] == claim_filter]
                if not final_claim.empty:
                    # Display relevant final status columns
                    final_display_cols = ['clmNum', 'clmCause', 'clmStatus', 'datetxn', 'dateCompleted', 'dateReopened', 
                                        'paid_cumsum', 'expense_cumsum', 'incurred_cumsum']
                    available_final_cols = [col for col in final_display_cols if col in final_claim.columns]
                    final_display = final_claim[available_final_cols]
                    st.dataframe(final_display, use_container_width=True)
                else:
                    st.info("No final status data available for this claim")
            else:
                st.info("Final claim data not available")
        
        # Summary statistics
        st.subheader("📊 Original Data Summary")
        
        # Safe date formatting function
        def safe_strftime(date_series, format_str='%Y-%m-%d'):
            """Safely format dates, handling NaT values"""
            if pd.isna(date_series):
                return 'N/A'
            try:
                return date_series.strftime(format_str)
            except (ValueError, TypeError):
                return 'N/A'
        
        summary_stats = {
            'Total Transactions': len(df_raw_txn_filtered),
            'First Transaction': safe_strftime(df_raw_txn_filtered['datetxn'].min()),
            'Last Transaction': safe_strftime(df_raw_txn_filtered['datetxn'].max()),
            'Total Paid': f"${df_raw_txn_filtered['paid'].sum():,.2f}",
            'Total Expense': f"${df_raw_txn_filtered['expense'].sum():,.2f}",
            'Final Incurred': f"${df_raw_txn_filtered['incurred_cumsum'].iloc[-1]:,.2f}",
            'Claim Status': df_raw_txn_filtered['clmStatus'].iloc[-1],
            'Claim Cause': df_raw_txn_filtered['clmCause'].iloc[0] if 'clmCause' in df_raw_txn_filtered.columns else 'N/A'
        }
        
        for key, value in summary_stats.items():
            st.metric(key, value)
    
    def _render_standardized_data(self, df_raw_txn_filtered: pd.DataFrame, 
                                config: StandardizationConfig, claim_filter: str) -> Optional[Any]:
        """Render standardized data section."""
        st.subheader("🔄 Standardized Periods")
        
        # Use the shared transformer instance and update its config
        self.transformer.config = config
        
        try:
            with st.spinner("Standardizing claim data..."):
                dataset = self.transformer.transform_claims_data(df_raw_txn_filtered)
            
            st.write(f"**Dataset created with {len(dataset.claims)} claims**")
            
            if len(dataset.claims) > 0:
                standardized_claim = dataset.claims[0]
                
                st.write(f"**Claim {standardized_claim.static_context.clmNum} standardized successfully**")
                
                # Display standardized periods
                periods_data = []
                for period in standardized_claim.dynamic_periods:
                    periods_data.append({
                        'Period': str(period.period),
                        'Days from Receipt': str(period.days_from_receipt),
                        'Period Start': period.period_start_date.strftime('%Y-%m-%d'),
                        'Period End': period.period_end_date.strftime('%Y-%m-%d'),
                        'Incremental Paid': f"${period.incremental_paid:,.2f}",
                        'Incremental Paid (Norm)': f"{period.incremental_paid_normalized:.4f}",
                        'Incremental Expense': f"${period.incremental_expense:,.2f}",
                        'Incremental Expense (Norm)': f"{period.incremental_expense_normalized:.4f}",
                        'Cumulative Paid': f"${period.cumulative_paid:,.2f}",
                        'Cumulative Paid (Norm)': f"{period.cumulative_paid_normalized:.4f}",
                        'Cumulative Expense': f"${period.cumulative_expense:,.2f}",
                        'Cumulative Expense (Norm)': f"{period.cumulative_expense_normalized:.4f}",
                        'Cumulative Incurred': f"${period.cumulative_incurred:,.2f}",
                        'Has Payment': 'Yes' if period.has_payment else 'No',
                        'Has Expense': 'Yes' if period.has_expense else 'No',
                        'Transactions': str(period.num_transactions)
                    })
                
                periods_df = pd.DataFrame(periods_data)
                
                if len(periods_df) > 0:
                    st.dataframe(periods_df, use_container_width=True)
                else:
                    st.warning("No standardized periods generated")
                
                # Summary of standardized data
                st.subheader("📊 Standardized Data Summary")
                standardized_summary = {
                    'Total Periods': len(standardized_claim.dynamic_periods),
                    'Max Period': standardized_claim.max_period,
                    'Total Paid': f"${standardized_claim.total_paid:,.2f}",
                    'Total Expense': f"${standardized_claim.total_expense:,.2f}",
                    'Final Incurred': f"${standardized_claim.final_incurred:,.2f}",
                    'Final Reserve': f"${standardized_claim.final_reserve:,.2f}",
                    'Periods with Payments': sum(1 for p in standardized_claim.dynamic_periods if p.has_payment),
                    'Periods with Expenses': sum(1 for p in standardized_claim.dynamic_periods if p.has_expense)
                }
                
                for key, value in standardized_summary.items():
                    st.metric(key, value)
                
                # Validation
                st.subheader("✅ Data Validation")
                comparison = transformer.compare_with_original(standardized_claim, df_raw_txn_filtered)
                
                validation_results = {
                    'Total Paid Match': '✅' if comparison['total_paid_match'] else '❌',
                    'Total Expense Match': '✅' if comparison['total_expense_match'] else '❌',
                    'Total Recovery Match': '✅' if comparison['total_recovery_match'] else '❌',
                    'Total Reserve Match': '✅' if comparison['total_reserve_match'] else '❌'
                }
                
                for key, value in validation_results.items():
                    st.metric(key, value)
                
                return standardized_claim
            
            else:
                st.error("❌ No standardized claims generated")
                self._render_debug_info(df_raw_txn_filtered, config)
                return None
                
        except Exception as e:
            st.error(f"❌ Error during standardization: {str(e)}")
            st.exception(e)
            return None
    
    def _render_debug_info(self, df_raw_txn_filtered: pd.DataFrame, config: StandardizationConfig):
        """Render debug information when standardization fails."""
        st.write("**Debug Information:**")
        st.write(f"- Input transactions: {len(df_raw_txn_filtered)}")
        st.write(f"- Claim numbers: {df_raw_txn_filtered['clmNum'].unique()}")
        st.write(f"- Date range: {df_raw_txn_filtered['datetxn'].min()} to {df_raw_txn_filtered['datetxn'].max()}")
        st.write(f"- Min periods required: {config.min_periods}")
        st.write(f"- Max periods allowed: {config.max_periods}")
        st.write(f"- Period length: {config.period_length_days} days")
    
    def render_visualizations(self, df_raw_txn_filtered: pd.DataFrame, claim_filter: str, 
                            period_length: int, config: StandardizationConfig):
        """Render the interactive visualizations."""
        with st.expander("📈 Interactive Visualization - Individual Claim Analysis", expanded=False):
            if len(df_raw_txn_filtered) > 0:
                # Always generate standardized data for the charts
                # Use the shared transformer instance and update its config
                self.transformer.config = config
                standardized_claim = None
                
                try:
                    with st.spinner("Generating standardized data for visualization..."):
                        dataset = self.transformer.transform_claims_data(df_raw_txn_filtered)
                    if len(dataset.claims) > 0:
                        standardized_claim = dataset.claims[0]
                except Exception as e:
                    st.warning(f"Could not generate standardized data for visualization: {str(e)}")
                
                df_raw_txn_filtered_sorted = df_raw_txn_filtered.sort_values('datetxn')
                
                # Create subplots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        'Cumulative Amounts Over Time',
                        'Incremental Payments by Transaction',
                        f'Standardized Periods ({period_length}-day)',
                        'Cumulative Amounts by Period'
                    ),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # 1. Transaction timeline - Cumulative amounts
                fig.add_trace(
                    go.Scatter(
                        x=df_raw_txn_filtered_sorted['datetxn'],
                        y=df_raw_txn_filtered_sorted['paid_cumsum'],
                        mode='lines+markers',
                        name='Cumulative Paid',
                        line=dict(color='blue', width=3),
                        marker=dict(size=6)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df_raw_txn_filtered_sorted['datetxn'],
                        y=df_raw_txn_filtered_sorted['expense_cumsum'],
                        mode='lines+markers',
                        name='Cumulative Expense',
                        line=dict(color='red', width=3),
                        marker=dict(size=6)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df_raw_txn_filtered_sorted['datetxn'],
                        y=df_raw_txn_filtered_sorted['incurred_cumsum'],
                        mode='lines+markers',
                        name='Cumulative Incurred',
                        line=dict(color='green', width=3),
                        marker=dict(size=6)
                    ),
                    row=1, col=1
                )
                
                # 2. Incremental payments
                transaction_numbers = list(range(len(df_raw_txn_filtered_sorted)))
                
                fig.add_trace(
                    go.Bar(
                        x=transaction_numbers,
                        y=df_raw_txn_filtered_sorted['paid'],
                        name='Paid',
                        marker_color='blue',
                        opacity=0.7
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Bar(
                        x=transaction_numbers,
                        y=df_raw_txn_filtered_sorted['expense'],
                        name='Expense',
                        marker_color='red',
                        opacity=0.7
                    ),
                    row=1, col=2
                )
                
                # 3. Standardized periods (if available)
                if standardized_claim and len(standardized_claim.dynamic_periods) > 0:
                    periods = [p.period for p in standardized_claim.dynamic_periods]
                    incremental_paid = [p.incremental_paid for p in standardized_claim.dynamic_periods]
                    incremental_expense = [p.incremental_expense for p in standardized_claim.dynamic_periods]
                    
                    fig.add_trace(
                        go.Bar(
                            x=periods,
                            y=incremental_paid,
                            name='Incremental Paid (Periods)',
                            marker_color='blue',
                            opacity=0.7,
                            offsetgroup=1
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=periods,
                            y=incremental_expense,
                            name='Incremental Expense (Periods)',
                            marker_color='red',
                            opacity=0.7,
                            offsetgroup=2
                        ),
                        row=2, col=1
                    )
                    
                    # 4. Cumulative standardized
                    cumulative_paid = [p.cumulative_paid for p in standardized_claim.dynamic_periods]
                    cumulative_expense = [p.cumulative_expense for p in standardized_claim.dynamic_periods]
                    cumulative_incurred = [p.cumulative_incurred for p in standardized_claim.dynamic_periods]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=periods,
                            y=cumulative_paid,
                            mode='lines+markers',
                            name='Cumulative Paid (Periods)',
                            line=dict(color='blue', width=3),
                            marker=dict(size=6)
                        ),
                        row=2, col=2
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=periods,
                            y=cumulative_expense,
                            mode='lines+markers',
                            name='Cumulative Expense (Periods)',
                            line=dict(color='red', width=3),
                            marker=dict(size=6)
                        ),
                        row=2, col=2
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=periods,
                            y=cumulative_incurred,
                            mode='lines+markers',
                            name='Cumulative Incurred (Periods)',
                            line=dict(color='green', width=3),
                            marker=dict(size=6)
                        ),
                        row=2, col=2
                    )
                else:
                    # Show placeholder text if no standardized data
                    fig.add_annotation(
                        text="No standardized data available",
                        xref="x3", yref="y3",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=16, color="gray"),
                        row=2, col=1
                    )
                    fig.add_annotation(
                        text="No standardized data available",
                        xref="x4", yref="y4",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=16, color="gray"),
                        row=2, col=2
                    )
                
                # Update layout
                fig.update_layout(
                    height=800,
                    title_text=f"Claim Analysis: {claim_filter}",
                    title_x=0.5,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                # Update x and y axis labels
                fig.update_xaxes(title_text="Date", row=1, col=1)
                fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
                
                fig.update_xaxes(title_text="Transaction Number", row=1, col=2)
                fig.update_yaxes(title_text="Amount ($)", row=1, col=2)
                
                fig.update_xaxes(title_text="Period", row=2, col=1)
                fig.update_yaxes(title_text="Amount ($)", row=2, col=1)
                
                fig.update_xaxes(title_text="Period", row=2, col=2)
                fig.update_yaxes(title_text="Amount ($)", row=2, col=2)
                
                # Format y-axis as currency
                fig.update_layout(yaxis_tickformat='$,.0f')
                fig.update_layout(yaxis2_tickformat='$,.0f')
                fig.update_layout(yaxis3_tickformat='$,.0f')
                fig.update_layout(yaxis4_tickformat='$,.0f')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add claim-level normalization visualization
                if standardized_claim and hasattr(self.transformer, 'normalization_computed') and self.transformer.normalization_computed:
                    st.subheader("🔍 Claim-Level Normalization Analysis")
                    
                    # Show normalized vs original values for this specific claim
                    periods = [p.period for p in standardized_claim.dynamic_periods]
                    
                    # Get original and normalized incremental values
                    original_paid = [p.incremental_paid for p in standardized_claim.dynamic_periods]
                    normalized_paid = [p.incremental_paid_normalized for p in standardized_claim.dynamic_periods]
                    original_expense = [p.incremental_expense for p in standardized_claim.dynamic_periods]
                    normalized_expense = [p.incremental_expense_normalized for p in standardized_claim.dynamic_periods]
                    
                    # Create normalization comparison chart
                    fig_norm = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=(
                            f'Paid Amounts - Original vs Normalized (Claim {claim_filter})',
                            f'Expense Amounts - Original vs Normalized (Claim {claim_filter})'
                        )
                    )
                    
                    # Paid amounts comparison
                    fig_norm.add_trace(
                        go.Bar(
                            x=periods,
                            y=original_paid,
                            name='Original Paid',
                            marker_color='lightblue',
                            opacity=0.7,
                            text=[f'${x:,.0f}' for x in original_paid],
                            textposition='outside'
                        ),
                        row=1, col=1
                    )
                    
                    fig_norm.add_trace(
                        go.Bar(
                            x=periods,
                            y=normalized_paid,
                            name='Normalized Paid',
                            marker_color='darkblue',
                            opacity=0.7,
                            text=[f'{x:.2f}' for x in normalized_paid],
                            textposition='outside',
                            yaxis='y2'
                        ),
                        row=1, col=1
                    )
                    
                    # Expense amounts comparison
                    fig_norm.add_trace(
                        go.Bar(
                            x=periods,
                            y=original_expense,
                            name='Original Expense',
                            marker_color='lightcoral',
                            opacity=0.7,
                            text=[f'${x:,.0f}' for x in original_expense],
                            textposition='outside'
                        ),
                        row=1, col=2
                    )
                    
                    fig_norm.add_trace(
                        go.Bar(
                            x=periods,
                            y=normalized_expense,
                            name='Normalized Expense',
                            marker_color='darkred',
                            opacity=0.7,
                            text=[f'{x:.2f}' for x in normalized_expense],
                            textposition='outside'
                        ),
                        row=1, col=2
                    )
                    
                    # Add secondary y-axis for normalized values
                    fig_norm.update_layout(
                        yaxis=dict(title="Original Amount ($)", side="left"),
                        yaxis2=dict(title="Normalized Value", side="right", overlaying="y"),
                        yaxis3=dict(title="Original Amount ($)", side="left"),
                        yaxis4=dict(title="Normalized Value", side="right", overlaying="y3"),
                        height=400,
                        title_text=f"Normalization Analysis for Claim {claim_filter}",
                        title_x=0.5
                    )
                    
                    fig_norm.update_xaxes(title_text="Period", row=1, col=1)
                    fig_norm.update_xaxes(title_text="Period", row=1, col=2)
                    
                    st.plotly_chart(fig_norm, use_container_width=True)
                    
                    # Show normalization parameters used
                    st.subheader("📊 Normalization Parameters Applied")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Paid Normalization:**")
                        paid_params = self.transformer.normalization_params['paid']
                        if paid_params['mean'] is not None:
                            st.write(f"μ = ${paid_params['mean']:,.2f}")
                            st.write(f"σ = ${paid_params['std']:,.2f}")
                        
                        # Show transformation for this claim's paid values
                        if any(original_paid):
                            st.markdown("**This Claim's Paid Transformations:**")
                            for i, (orig, norm) in enumerate(zip(original_paid, normalized_paid)):
                                if orig > 0:  # Only show non-zero values
                                    st.write(f"Period {i}: ${orig:,.0f} → {norm:.2f}")
                    
                    with col2:
                        st.markdown("**Expense Normalization:**")
                        expense_params = self.transformer.normalization_params['expense']
                        if expense_params['mean'] is not None:
                            st.write(f"μ = ${expense_params['mean']:,.2f}")
                            st.write(f"σ = ${expense_params['std']:,.2f}")
                        
                        # Show transformation for this claim's expense values
                        if any(original_expense):
                            st.markdown("**This Claim's Expense Transformations:**")
                            for i, (orig, norm) in enumerate(zip(original_expense, normalized_expense)):
                                if orig > 0:  # Only show non-zero values
                                    st.write(f"Period {i}: ${orig:,.0f} → {norm:.2f}")
                
                else:
                    st.info("💡 Normalization analysis will appear here once normalization parameters are computed. Check the 'Normalization Validation' section below.")
    
    def render_download_section(self, df_raw_txn_filtered: pd.DataFrame, claim_filter: str, 
                              standardized_claim: Optional[Any] = None):
        """Render download options."""
        st.subheader("📥 Download Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download original data
            csv_original = df_raw_txn_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Original Transactions",
                data=csv_original,
                file_name=f"original_transactions_{claim_filter}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download standardized data
            if standardized_claim and len(standardized_claim.dynamic_periods) > 0:
                periods_data = []
                for period in standardized_claim.dynamic_periods:
                    periods_data.append({
                        'Period': period.period,
                        'Days from Receipt': period.days_from_receipt,
                        'Period Start': period.period_start_date.strftime('%Y-%m-%d'),
                        'Period End': period.period_end_date.strftime('%Y-%m-%d'),
                        'Incremental Paid': f"${period.incremental_paid:,.2f}",
                        'Incremental Expense': f"${period.incremental_expense:,.2f}",
                        'Cumulative Paid': f"${period.cumulative_paid:,.2f}",
                        'Cumulative Expense': f"${period.cumulative_expense:,.2f}",
                        'Cumulative Incurred': f"${period.cumulative_incurred:,.2f}",
                        'Has Payment': 'Yes' if period.has_payment else 'No',
                        'Has Expense': 'Yes' if period.has_expense else 'No',
                        'Transactions': period.num_transactions
                    })
                
                periods_df = pd.DataFrame(periods_data)
                csv_standardized = periods_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Standardized Periods",
                    data=csv_standardized,
                    file_name=f"standardized_periods_{claim_filter}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    
    def render_no_claim_selected(self, df_raw_txn_filtered: pd.DataFrame):
        """Render content when no claim is selected."""
        st.info("👆 Please enter a claim number in the sidebar to begin analysis")
        
        # Show some sample claims (from filtered data)
        st.subheader("📋 Sample Claims Available")
        sample_claims = df_raw_txn_filtered['clmNum'].unique()[:20]
        if len(sample_claims) > 0:
            st.write("Here are some sample claim numbers you can try:")
            
            # Create columns for better display
            cols = st.columns(4)
            for i, claim in enumerate(sample_claims):
                with cols[i % 4]:
                    if st.button(f"Select {claim}", key=f"claim_{claim}"):
                        st.session_state.claim_filter = claim
                        st.rerun()
        else:
            st.warning("No claims available with the current filters.")
        
        # Show dataset overview
        self._render_dataset_overview(df_raw_txn_filtered)
    
    def _render_dataset_overview(self, df_raw_txn: pd.DataFrame):
        """Render dataset overview."""
        st.subheader("📊 Dataset Overview")
        overview_stats = {
            'Total Claims': df_raw_txn['clmNum'].nunique(),
            'Total Transactions': len(df_raw_txn),
            'Date Range': f"{df_raw_txn['datetxn'].min().strftime('%Y-%m-%d')} to {df_raw_txn['datetxn'].max().strftime('%Y-%m-%d')}",
            'Total Paid': f"${df_raw_txn['paid'].sum():,.2f}",
            'Total Expense': f"${df_raw_txn['expense'].sum():,.2f}",
            'Claims with Payments': (df_raw_txn['paid'] > 0).groupby(df_raw_txn['clmNum']).any().sum(),
            'Claims with Expenses': (df_raw_txn['expense'] > 0).groupby(df_raw_txn['clmNum']).any().sum()
        }
        
        col1, col2, col3 = st.columns(3)
        for i, (key, value) in enumerate(overview_stats.items()):
            with [col1, col2, col3][i % 3]:
                st.metric(key, value)
    
    def render_normalization_visualization(self, df_raw_txn: pd.DataFrame, df_raw_final: Optional[pd.DataFrame] = None):
        """Render normalization visualization for validation."""
        
        with st.expander("📊 Normalization Validation - Global Distribution", expanded=False):
            st.subheader("🔍 Normalization Parameters Validation")
            
            if not hasattr(self.transformer, 'normalization_computed') or not self.transformer.normalization_computed:
                st.warning("⚠️ Normalization parameters not computed yet. Computing now...")
                
                if df_raw_final is not None:
                    self.transformer.calculate_normalization_parameters(df_raw_txn, df_raw_final)
                else:
                    st.error("❌ Cannot compute normalization parameters: df_raw_final not provided")
                    return
            
            # Display normalization parameters
            st.subheader("📈 Normalization Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Paid Amounts:**")
                paid_params = self.transformer.normalization_params['paid']
                if paid_params['mean'] is not None:
                    st.metric("Mean (μ)", f"${paid_params['mean']:,.2f}")
                    st.metric("Std Dev (σ)", f"${paid_params['std']:,.2f}")
                else:
                    st.warning("No paid normalization parameters available")
            
            with col2:
                st.markdown("**Expense Amounts:**")
                expense_params = self.transformer.normalization_params['expense']
                if expense_params['mean'] is not None:
                    st.metric("Mean (μ)", f"${expense_params['mean']:,.2f}")
                    st.metric("Std Dev (σ)", f"${expense_params['std']:,.2f}")
                else:
                    st.warning("No expense normalization parameters available")
            
            # Create and display the visualization
            st.subheader("📊 Distribution Comparison: Before vs After Normalization")
            
            try:
                fig = self.transformer.create_normalization_visualization(df_raw_txn, df_raw_final)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to create normalization visualization")
            except Exception as e:
                st.error(f"Error creating normalization visualization: {str(e)}")
            
            # Show summary statistics
            st.subheader("📋 Normalization Summary")
            
            # Filter to completed claims only (same logic as normalization)
            completed_statuses = ['PAID', 'DENIED', 'CLOSED']
            if df_raw_final is not None:
                completed_claims = df_raw_final[df_raw_final['clmStatus'].isin(completed_statuses)]['clmNum'].unique()
                df_completed_txn = df_raw_txn[df_raw_txn['clmNum'].isin(completed_claims)].copy()
                
                # Apply normalization
                df_normalized = self.transformer.apply_normalization(df_completed_txn)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Completed Claims Used", len(completed_claims))
                    st.metric("Total Transactions", len(df_completed_txn))
                
                with col2:
                    paid_positive = df_completed_txn[df_completed_txn['paid'] > 0]['paid']
                    expense_positive = df_completed_txn[df_completed_txn['expense'] > 0]['expense']
                    st.metric("Positive Paid Transactions", len(paid_positive))
                    st.metric("Positive Expense Transactions", len(expense_positive))
                
                with col3:
                    if len(paid_positive) > 0 and len(expense_positive) > 0:
                        paid_normalized = df_normalized[df_normalized['paid'] > 0]['paid_normalized']
                        expense_normalized = df_normalized[df_normalized['expense'] > 0]['expense_normalized']
                        
                        st.metric("Paid Norm Mean", f"{paid_normalized.mean():.4f}")
                        st.metric("Expense Norm Mean", f"{expense_normalized.mean():.4f}")
            else:
                st.warning("Cannot show summary: df_raw_final not provided")
    
    def render_footer(self):
        """Render the footer."""
        st.markdown("---")
        st.markdown("**Micro-Level Reserving Framework** - Standardizing claims data into discrete 30-day periods for LSTM-based reserving models.")
    
    def run_app(self, df_raw_txn: pd.DataFrame, 
                claim_filter_func=None, 
                show_all_transactions: bool = False,
                df_raw_final: Optional[pd.DataFrame] = None):
        """
        Main app runner method.
        
        Args:
            df_raw_txn: The raw transaction dataframe
            claim_filter_func: Optional function to filter claims (for real data apps)
            show_all_transactions: Whether to show all transactions dataframe
            df_raw_final: Pre-computed final claim status data
        """
        # Store the final data for use in display methods
        self.df_raw_final = df_raw_final
        
        # Calculate normalization parameters early if we have final data
        if df_raw_final is not None and not self.transformer.normalization_computed:
            try:
                with st.spinner("Calculating normalization parameters..."):
                    self.transformer.calculate_normalization_parameters(df_raw_txn, df_raw_final)
            except Exception as e:
                st.warning(f"Could not calculate normalization parameters: {str(e)}")
        
        # Render header
        self.render_header()
        
        # Render sidebar controls
        claim_filter, period_length, max_periods, min_periods, clm_cause_filter, clm_status_filter = self.render_sidebar_controls(df_raw_txn)
        
        # Apply filters to the dataframe
        df_raw_txn_filtered = self.apply_filters(df_raw_txn, clm_cause_filter, clm_status_filter)
        
        # Show filter summary
        if clm_cause_filter != "All" or clm_status_filter != "All":
            st.info(f"🔍 **Filters Applied:** {clm_cause_filter} | {clm_status_filter} | Showing {len(df_raw_txn_filtered):,} transactions from {df_raw_txn_filtered['clmNum'].nunique():,} claims")
        
        # Show all transactions if requested (for demo app) - now with filters applied
        if show_all_transactions:
            with st.expander("📊 All Claims Data - Transactions & Final Status", expanded=False):
                st.subheader("📋 All Transactions Data")
                # Handle data types robustly to avoid JSON serialization issues
                df_raw_txn_filtered_display = df_raw_txn_filtered.copy()
                for col in df_raw_txn_filtered_display.columns:
                    try:
                        if pd.api.types.is_datetime64_any_dtype(df_raw_txn_filtered_display[col]):
                            # Handle datetime columns
                            df_raw_txn_filtered_display[col] = df_raw_txn_filtered_display[col].dt.strftime('%Y-%m-%d')
                        elif pd.api.types.is_numeric_dtype(df_raw_txn_filtered_display[col]):
                            # Handle numeric columns
                            df_raw_txn_filtered_display[col] = df_raw_txn_filtered_display[col].astype(str)
                        else:
                            # Handle all other types (object, bool, etc.)
                            df_raw_txn_filtered_display[col] = df_raw_txn_filtered_display[col].astype(str)
                    except Exception as e:
                        # Fallback: convert to string regardless of type
                        df_raw_txn_filtered_display[col] = df_raw_txn_filtered_display[col].astype(str)
                
                # Double-check: ensure all columns are strings
                for col in df_raw_txn_filtered_display.columns:
                    df_raw_txn_filtered_display[col] = df_raw_txn_filtered_display[col].astype(str)
                
                st.dataframe(df_raw_txn_filtered_display, use_container_width=True)

                st.subheader("📊 All Periods (All Claims)")
                
                # Cache management with selection UI
                available_caches = self.cache_manager.list_available_caches()
                
                if available_caches:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        # Cache selection dropdown with extraction dates
                        cache_options = {}
                        for cache in available_caches:
                            cache_options[cache['extraction_date']] = f"📅 {cache['extraction_date']} - {cache['description']} ({cache['cache_size_mb']} MB)"
                        
                        selected_extraction_date = st.selectbox(
                            "📁 Select Data Version:",
                            options=list(cache_options.keys()),
                            format_func=lambda x: cache_options[x],
                            index=0,  # Select newest by default
                            help="Choose which version of processed data to use"
                        )
                    
                    with col2:
                        force_recompute = st.button("🔄 Force Recompute", help="Recompute period data even if cache exists")
                    
                    with col3:
                        show_cache_details = st.button("📋 Cache Details", help="Show detailed cache information")
                    
                    # Show cache details if requested
                    if show_cache_details:
                        # Find selected cache info
                        selected_cache_info = next((c for c in available_caches if c['extraction_date'] == selected_extraction_date), None)
                        
                        if selected_cache_info:
                            with st.expander("📋 Cache Details", expanded=True):
                                details = {
                                    "Extraction Date": selected_cache_info['extraction_date'],
                                    "Description": selected_cache_info['description'],
                                    "Created": selected_cache_info['created_at'],
                                    "Transactions": f"{selected_cache_info['num_transactions']:,}",
                                    "Claims": f"{selected_cache_info['num_claims']:,}",
                                    "Cache Size": f"{selected_cache_info['cache_size_mb']} MB",
                                    "Input Files": selected_cache_info['input_files']
                                }
                                
                                st.json(details)
                else:
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        force_recompute = st.button("🔄 Force Recompute", help="Recompute period data even if cache exists")
                
                with st.spinner("Processing all claims with vectorized approach..."):
                    # Determine extraction date and input files based on selection
                    extraction_date = None
                    input_files = []
                    
                    if available_caches and selected_extraction_date:
                        extraction_date = selected_extraction_date
                        # Get organized input files for this extraction date
                        input_files = self.data_organizer.get_organized_files(extraction_date)
                    
                    # Only process if we have an extraction date and input files
                    if extraction_date and input_files:
                        periods_all_df = self.transformer.transform_claims_data_vectorized(
                            df_raw_txn_filtered, 
                            force_recompute=force_recompute,
                            input_files=input_files,
                            extraction_date=extraction_date
                        )
                    else:
                        st.error("❌ No organized data found. Please organize your data by extraction date first.")
                        periods_all_df = pd.DataFrame()
                
                if not periods_all_df.empty:
                    st.dataframe(periods_all_df, use_container_width=True)
                    st.info(f"📈 **Dataset Summary:** {len(periods_all_df):,} periods from {periods_all_df['clmNum'].nunique():,} claims")
                    
                    # Show current cache info
                    if available_caches:
                        st.success(f"💾 **Cache Status:** {len(available_caches)} cache file(s) available")
                else:
                    st.warning("No period data available")
                
                # Show final claim status for all claims using pre-computed final data
                if not df_raw_txn_filtered.empty and hasattr(self, 'df_raw_final') and self.df_raw_final is not None:
                    st.subheader("📊 Final Claim Status (All Claims)")
                    # Filter the pre-computed final data by the same filters applied to transactions
                    final_claims_all = self.df_raw_final[
                        (self.df_raw_final['clmCause'].isin(df_raw_txn_filtered['clmCause'].unique())) &
                        (self.df_raw_final['clmStatus'].isin(df_raw_txn_filtered['clmStatus'].unique()))
                    ]
                    
                    # Display relevant final status columns
                    final_display_cols = ['clmNum', 'clmCause', 'clmStatus', 'datetxn', 'dateCompleted', 'dateReopened', 
                                        'paid_cumsum', 'expense_cumsum', 'incurred_cumsum']
                    available_final_cols = [col for col in final_display_cols if col in final_claims_all.columns]
                    
                    final_display_all = final_claims_all[available_final_cols]
                    st.dataframe(final_display_all, use_container_width=True)
                    
                    # Summary of all final claims
                    st.subheader("📊 Final Claims Summary")
                    final_summary = {
                        'Total Claims': len(final_claims_all),
                        'Open Claims': len(final_claims_all[final_claims_all['clmStatus'] == 'OPEN']) if 'clmStatus' in final_claims_all.columns else 0,
                        'Paid Claims': len(final_claims_all[final_claims_all['clmStatus'] == 'PAID']) if 'clmStatus' in final_claims_all.columns else 0,
                        'Closed Claims': len(final_claims_all[final_claims_all['clmStatus'] == 'CLOSED']) if 'clmStatus' in final_claims_all.columns else 0,
                        'Total Final Paid': f"${final_claims_all['paid_cumsum'].sum():,.2f}" if 'paid_cumsum' in final_claims_all.columns else "$0.00",
                        'Total Final Expense': f"${final_claims_all['expense_cumsum'].sum():,.2f}" if 'expense_cumsum' in final_claims_all.columns else "$0.00",
                        'Total Final Incurred': f"${final_claims_all['incurred_cumsum'].sum():,.2f}" if 'incurred_cumsum' in final_claims_all.columns else "$0.00"
                    }
                    
                    col1, col2, col3, col4 = st.columns(4)
                    for i, (key, value) in enumerate(final_summary.items()):
                        with [col1, col2, col3, col4][i % 4]:
                            st.metric(key, value)
        
        # Create configuration
        config = self.create_config(period_length, max_periods, min_periods)
        
        # Main content logic
        if claim_filter and len(claim_filter.strip()) > 0:
            # Filter data for the selected claim from the already filtered dataframe
            if claim_filter_func:
                claim_filtered_data = claim_filter_func(df_raw_txn_filtered, claim_filter)
            else:
                claim_filtered_data = df_raw_txn_filtered[df_raw_txn_filtered['clmNum'].str.contains(claim_filter.strip(), case=False, na=False)]
            
            if len(claim_filtered_data) == 0:
                st.error(f"❌ No data found for claim: {claim_filter}")
                st.info("Available claims (after filters):")
                available_claims = df_raw_txn_filtered['clmNum'].unique()
                for claim in available_claims:
                    st.write(f"- {claim}")
            else:
                st.success(f"✅ Found {len(claim_filtered_data)} transactions for claim: {claim_filter}")
                
                # Render original data section
                standardized_claim = self.render_original_data_section(claim_filtered_data, claim_filter, config)
                
                # Render visualizations (always show both transaction and standardized plots)
                self.render_visualizations(claim_filtered_data, claim_filter, period_length, config)
                
                # Render download section
                self.render_download_section(claim_filtered_data, claim_filter, standardized_claim)
        else:
            self.render_no_claim_selected(df_raw_txn_filtered)
        
        # Render global normalization visualization
        self.render_normalization_visualization(df_raw_txn, df_raw_final)
        
        # Render footer
        self.render_footer()
