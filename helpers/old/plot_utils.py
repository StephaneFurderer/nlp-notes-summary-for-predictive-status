import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def plot_single_claim_lifetime(df_txn, selected_claim):
    if selected_claim:
        # Get transaction data for the selected claim
        claim_transactions = df_txn[df_txn['clmNum'] == selected_claim].sort_values('datetxn')
        selected_cause = claim_transactions['clmCause'].unique()[0]
        
        if len(claim_transactions) > 0:
            # Create development pattern visualization
            fig_development = go.Figure()
            
            # Add traces for each metric
            metrics_to_plot = ['reserve_cumsum', 'paid_cumsum', 'incurred_cumsum', 'expense_cumsum']
            colors = ['red', 'green', 'blue', 'orange']
            
            for i, metric in enumerate(metrics_to_plot):
                if metric in claim_transactions.columns:
                    fig_development.add_trace(go.Scatter(
                        x=claim_transactions['datetxn'],
                        y=claim_transactions[metric],
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(color=colors[i], width=3),
                        marker=dict(size=6, color=colors[i])
                    ))
            
            # Update layout
            fig_development.update_layout(
                title=f"Claim {selected_claim} Development Pattern - {selected_cause}",
                xaxis_title="Transaction Date",
                yaxis_title="Cumulative Amount ($)",
                showlegend=True,
                height=500,
                hovermode='x unified'
            )
            
            # Format x-axis to show dates nicely
            fig_development.update_xaxes(
                tickformat='%Y-%m-%d',
                tickangle=45
            )
            
            st.plotly_chart(fig_development, use_container_width=True)
            
            # Add claim summary information
            st.write("**Claim Summary Information:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Status", claim_transactions['clmStatus'].iloc[-1])
                reopened_date = claim_transactions['dateReopened'].iloc[-1]
                if pd.notna(reopened_date) and reopened_date is not None:
                    st.metric("Reopened", reopened_date.strftime('%Y-%m-%d'))
                else:
                    st.metric("Reopened", "N/A")
            
            with col2:
                st.metric("Total Transactions", len(claim_transactions))
                if 'reserve_cumsum' in claim_transactions.columns and len(claim_transactions) > 0:
                    initial_reserve = claim_transactions['reserve_cumsum'].iloc[0]
                    if pd.notna(initial_reserve):
                        st.metric("Initial Set Estimate ", f"${initial_reserve:,.2f}")
                    else:
                        st.metric("Initial Set Estimate ", "N/A")
                else:
                    st.metric("Initial Set Estimate ", "N/A")
            
            with col3:
                first_date = claim_transactions['datetxn'].min()
                if pd.notna(first_date):
                    st.metric("First Date", first_date.strftime('%Y-%m-%d'))
                else:
                    st.metric("First Date", "N/A")
                if 'paid_cumsum' in claim_transactions.columns and len(claim_transactions) > 0:
                    total_paid = claim_transactions['paid_cumsum'].iloc[-1]
                    if pd.notna(total_paid):
                        st.metric("Total Paid", f"${total_paid:,.2f}")
                    else:
                        st.metric("Total Paid", "N/A")
                else:
                    st.metric("Total Paid", "N/A")
            
            with col4:
                last_date = claim_transactions['datetxn'].max()
                if pd.notna(last_date):
                    st.metric("Last Date", last_date.strftime('%Y-%m-%d'))
                else:
                    st.metric("Last Date", "N/A")
                if 'expense_cumsum' in claim_transactions.columns and len(claim_transactions) > 0:
                    total_expense = claim_transactions['expense_cumsum'].iloc[-1]
                    if pd.notna(total_expense):
                        st.metric("Total Expense", f"${total_expense:,.2f}")
                    else:
                        st.metric("Total Expense", "N/A")
                else:
                    st.metric("Total Expense", "N/A")
            
            # Show transaction details table
            st.write("**Transaction Details:**")
            display_cols = ['datetxn', 'reserve_cumsum', 'paid_cumsum', 'incurred_cumsum', 'expense_cumsum', 'clmStatus']
            available_display_cols = [col for col in display_cols if col in claim_transactions.columns]
            
            transaction_summary = claim_transactions[available_display_cols].copy()
            transaction_summary.columns = ['Transaction Date', 'Reserve Cumsum', 'Paid Cumsum', 'Incurred Cumsum', 'Expense Cumsum', 'Status']
            
            # Format currency columns
            currency_cols = ['Reserve Cumsum', 'Paid Cumsum', 'Incurred Cumsum', 'Expense Cumsum']
            for col in currency_cols:
                if col in transaction_summary.columns:
                    transaction_summary[col] = transaction_summary[col].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(transaction_summary, use_container_width=True)
            
            # Add change analysis
            st.write("**Change Analysis (Non-Zero Changes):**")
            change_cols = ['reserve_change', 'paid_change', 'incurred_change', 'expense_change']
            available_change_cols = [col for col in change_cols if col in claim_transactions.columns]
            
            if available_change_cols:
                change_data = []
                for col in available_change_cols:
                    non_zero_changes = claim_transactions[claim_transactions[col] != 0]
                    if len(non_zero_changes) > 0:
                        change_data.append({
                            'Metric': col.replace('_', ' ').title(),
                            'Non-Zero Changes': len(non_zero_changes),
                            'Average Change': f"${non_zero_changes[col].mean():,.2f}",
                            'Total Change': f"${non_zero_changes[col].sum():,.2f}",
                            'Max Increase': f"${non_zero_changes[col].max():,.2f}",
                            'Max Decrease': f"${non_zero_changes[col].min():,.2f}"
                        })
                
                if change_data:
                    change_df = pd.DataFrame(change_data)
                    st.dataframe(change_df, use_container_width=True)
                
                else:
                    st.info("No non-zero changes found for this claim")
            else:
                st.info("Change columns not available in the data")
        else:
            st.warning(f"No transaction data found for claim {selected_claim}")