import pandas as pd
import numpy as np
import os



def calculate_claim_features(df_open_txn):
    """ Calculate the comprehensive metrics for claims with the transaction data"""
    if len(df_open_txn) > 0:
        # Calculate comprehensive metrics for open claims (same as closed/paid claims)
        open_claims_metrics = []
        
        # Get the full transaction data for open claims of the selected cause
        open_claims_full_data = df_open_txn.sort_values(by=['clmNum','datetxn']).copy()
        
        for claim_num in open_claims_full_data['clmNum'].unique():
            # Get full transaction history for this claim
            claim_data = open_claims_full_data[open_claims_full_data['clmNum'] == claim_num].copy().sort_values(by=['clmNum','datetxn'])
            
            if len(claim_data) > 0:
                # Basic metrics
                num_transactions = len(claim_data)
                days_since_first_txn = (claim_data['datetxn'].max() - claim_data['datetxn'].min()).days
                
                # Time to first payment/expense
                first_paid_idx = claim_data['paid_cumsum'].gt(0).idxmax() if claim_data['paid_cumsum'].gt(0).any() else -1
                first_expense_idx = claim_data['expense_cumsum'].gt(0).idxmax() if claim_data['expense_cumsum'].gt(0).any() else -1
                
                if first_paid_idx != -1:
                    time_to_first_paid = (claim_data.loc[first_paid_idx, 'datetxn'] - claim_data['datetxn'].min()).days
                else:
                    time_to_first_paid = days_since_first_txn
                    
                if first_expense_idx != -1:
                    time_to_first_expense = (claim_data.loc[first_expense_idx, 'datetxn'] - claim_data['datetxn'].min()).days
                else:
                    time_to_first_expense = days_since_first_txn
                
                # Transaction counts
                num_expense_txns = (claim_data['expense'] > 0).sum()
                num_paid_txns = (claim_data['paid'] > 0).sum()
                
                # Average amounts
                avg_expense_amount = claim_data['expense_cumsum'].iloc[-1] / max(num_expense_txns, 1)
                avg_paid_amount = claim_data['paid_cumsum'].iloc[-1] / max(num_paid_txns, 1)
                
                # Change metrics (excluding zero changes)
                reserve_changes = claim_data[claim_data['reserve_change'] != 0]['reserve_change']
                incurred_changes = claim_data[claim_data['incurred_change'] != 0]['incurred_change']
                
                avg_reserve_change = reserve_changes.mean() if len(reserve_changes) > 0 else 0
                avg_incurred_change = incurred_changes.mean() if len(incurred_changes) > 0 else 0
                
                # Current amounts (from final transaction)
                current_reserve = claim_data['reserve_cumsum'].iloc[-1]
                current_incurred = claim_data['incurred_cumsum'].iloc[-1]
                current_paid = claim_data['paid_cumsum'].iloc[-1]
                current_expense = claim_data['expense_cumsum'].iloc[-1]
                
                # Ratios
                paid_to_incurred_ratio = current_paid / max(current_incurred, 1)
                expense_to_incurred_ratio = current_expense / max(current_incurred, 1)
                
                # Development stage (max 5 years = 1825 days)
                development_stage = min(days_since_first_txn / 1825, 1.0)
                
                open_claims_metrics.append({
                    'clmNum': claim_num,
                    'clmStatus': claim_data['clmStatus'].iloc[-1],  # Use final status
                    'clmCause': claim_data['clmCause'].iloc[0],  # First occurrence per claim
                    'num_transactions': num_transactions,
                    'days_since_first_txn': days_since_first_txn,
                    'time_to_first_paid': time_to_first_paid,
                    'time_to_first_expense': time_to_first_expense,
                    'num_expense_txns': num_expense_txns,
                    'num_paid_txns': num_paid_txns,
                    'avg_expense_amount': avg_expense_amount,
                    'avg_paid_amount': avg_paid_amount,
                    'avg_reserve_change': avg_reserve_change,
                    'avg_incurred_change': avg_incurred_change,
                    'current_reserve': current_reserve,
                    'current_incurred': current_incurred,
                    'current_paid': current_paid,
                    'current_expense': current_expense,
                    'current_cashflow': current_paid + current_expense,  # Add current cashflow
                    'paid_to_incurred_ratio': paid_to_incurred_ratio,
                    'expense_to_incurred_ratio': expense_to_incurred_ratio,
                    'development_stage': development_stage,  # Add development stage
                    'dateReopened': claim_data['dateReopened'].iloc[-1] if 'dateReopened' in claim_data.columns else None,
                    'isReopened': 1 if pd.notna(claim_data['dateReopened'].iloc[-1]) else 0
                })
        
        # Create comprehensive open claims summary
        open_claim_metrics_df = pd.DataFrame(open_claims_metrics)
        
        # Display summary statistics
        #st.write("**📊 Open Claims Summary Statistics**")
        summary_cols = ['num_transactions', 'days_since_first_txn', 'num_expense_txns', 'num_paid_txns', 
                        'current_reserve', 'current_incurred', 'current_paid', 'current_expense']
        
        summary_open_stats = open_claim_metrics_df.groupby(['clmCause','clmStatus'])[summary_cols].describe()
        #st.dataframe(summary_stats, use_container_width=True)
        
        # Display detailed metrics for each claim
        #st.write("**📋 Detailed Open Claims Metrics**")
        display_cols = ['clmCause','clmNum','num_transactions', 'days_since_first_txn', 'num_expense_txns', 
                        'num_paid_txns', 'current_reserve', 'current_incurred', 'current_paid', 'current_expense', 'current_cashflow']
        display_claim_metrics_df = open_claim_metrics_df[display_cols].copy()
        display_claim_metrics_df.columns = ['Claim Clause','Claim Number','Transactions', 'Days Since First Txn', 'Expense Txns', 
                                'Paid Txns', 'Current Reserve', 'Current Incurred', 'Current Paid', 'Current Expense', 'Current Cashflow']
        #st.dataframe(display_df, use_container_width=True)
        return open_claim_metrics_df,summary_open_stats,display_claim_metrics_df
    else:
        # Return empty DataFrames when there are no open claims
        empty_df = pd.DataFrame()
        empty_summary = pd.DataFrame()
        empty_display = pd.DataFrame()
        return empty_df, empty_summary, empty_display   

def _filter_by_(df,col,value):
    return df[df[col].str.contains(value.strip(), case=False, na=False)]
# Identify the main amounts: reserve, paid, expense
def calculate_claim_amounts(df):
    """ Calculate the main amounts: reserve, paid, expense """
    # Create the basic CASE logic for each payment type and category
    conditions_benefit = (df['paymentType'] == 'Benefit') & (df['processCategory'].str.contains('CLAIM', case=False, na=False))
    conditions_expense = (df['paymentType'] == 'Expense') & (df['processCategory'].str.contains('CLAIM', case=False, na=False))
    conditions_recovery = (df['paymentType'] == 'Recovery') & (df['processCategory'].str.contains('CLAIM', case=False, na=False))
    conditions_reserve = (df['paymentType'].isin(['Reserve','Benefit','Expense','Recovery'])) & (df['processCategory'].str.contains('CLAIM', case=False, na=False))
    
    # Calculate individual components
    df['paid'] = np.where(conditions_benefit, df['amt'], 0)
    df['expense'] = np.where(conditions_expense, df['amt'], 0) 
    df['recovery'] = np.where(conditions_recovery, df['amt'], 0)
    df['reserve'] = np.where(conditions_reserve, -df['amt'], 0)
 
    df['incurred'] = df['paid'] + df['expense'] + df['recovery'] + df['reserve']
    return df
    
# Group by your key dimensions and aggregate
def aggregate_by_booking_policy_claim(df,transaction_view:bool = False):

    # Apply the calculations
    df = calculate_claim_amounts(df)

    if transaction_view:
        group = ['booknum', 'cidpol', 'clmNum','clmStatus','dateCompleted','dateReopened','datetxn']

        grouped = df.groupby(group,dropna=False).agg({
            'paid': 'sum',
            'expense': 'sum', 
            'recovery': 'sum',
            'reserve': 'sum',
            'incurred': 'sum'
        }).reset_index()

        # remove rows with no transaction dates
        grouped = grouped[~grouped['datetxn'].isnull()]
        grouped = grouped.sort_values(['booknum', 'cidpol', 'clmNum','datetxn'])
        
        # Calculate cumulative sums over time for each claim
        claim_group = grouped.groupby(['booknum', 'cidpol', 'clmNum'])
        
        grouped['paid_cumsum'] = claim_group['paid'].cumsum()
        grouped['expense_cumsum'] = claim_group['expense'].cumsum()
        grouped['recovery_cumsum'] = claim_group['recovery'].cumsum()
        grouped['reserve_cumsum'] = claim_group['reserve'].cumsum()
        grouped['incurred_cumsum'] = claim_group['incurred'].cumsum()
        
        # Add period-over-period changes
        grouped['incurred_change'] = claim_group['incurred_cumsum'].shift(1)
        grouped['incurred_change'] = grouped['incurred_cumsum'] - grouped['incurred_change'].fillna(0)

        grouped['reserve_change'] = claim_group['reserve_cumsum'].shift(1)
        grouped['reserve_change'] = grouped['reserve_cumsum'] - grouped['reserve_change'].fillna(0)


        grouped['expense_change'] = claim_group['expense_cumsum'].shift(1)
        grouped['expense_change'] = grouped['expense_cumsum'] - grouped['expense_change'].fillna(0)

        grouped['paid_change'] = claim_group['paid_cumsum'].shift(1)
        grouped['paid_change'] = grouped['paid_cumsum'] - grouped['paid_change'].fillna(0)

        
    else:

        group = ['booknum', 'cidpol', 'clmNum','clmStatus','dateCompleted','dateReopened']

        grouped = df.groupby(group,dropna=False).agg({
            'paid': 'sum',
            'expense': 'sum', 
            'recovery': 'sum',
            'reserve': 'sum',
            'incurred': 'sum'
        }).reset_index()
        
    # Update claim status: if a CLOSED claim has paid amounts, change status to PAID
    if transaction_view:
        # For transaction view, we need to check the final status of each claim
        grouped['final_paid_amount'] = grouped.groupby(['booknum', 'cidpol', 'clmNum'])['paid_cumsum'].transform('max')
        grouped['final_status'] = grouped.groupby(['booknum', 'cidpol', 'clmNum'])['clmStatus'].transform('last')
        
        # Update status: CLOSED claims with payments become PAID
        grouped.loc[(grouped['final_status'] == 'CLOSED') & (grouped['final_paid_amount'] > 0), 'clmStatus'] = 'PAID'
        
        # Clean up temporary columns
        grouped = grouped.drop(['final_paid_amount', 'final_status'], axis=1)
    else:
        # For non-transaction view, check if there are any paid amounts
        grouped['has_payments'] = grouped['paid'] > 0
        
        # Update status: CLOSED claims with payments become PAID
        grouped.loc[(grouped['clmStatus'] == 'CLOSED') & (grouped['has_payments']), 'clmStatus'] = 'PAID'
        
        # Clean up temporary column
        grouped = grouped.drop('has_payments', axis=1)
    
    grouped['policy_has_first_open_claims'] = grouped.groupby(['booknum', 'cidpol'])['dateCompleted'].transform( lambda x:x.isnull().any())
    grouped['policy_has_reopen_claims'] = grouped.groupby(['booknum', 'cidpol'])['dateReopened'].transform( lambda x:x.isnull().any())
    grouped['policy_has_open_claims'] = ~(grouped['clmStatus'].isin(['PAID', 'CLOSED', 'DENIED', 'PARTIALLY_PAID','PARTIALLY_DENIED']))

    # ['PAID', 'CLOSED', 'DENIED', 'INITIAL_REVIEW', 'PARTIALLY_PAID','PARTIALLY_DENIED', 'FUTURE_PAY_POTENTIAL', 'ESTABLISHED', 'OPEN']
    
    return grouped

def import_data(reload:bool = False):
    parquet_file = '.\_data\clm_with_amt.parquet'
    if reload or (not os.path.exists(parquet_file)):
        print(f"create parquet file: {parquet_file}")
        df = pd.read_csv('.\_data\clm_with_amt.csv')
        df['booknum'] = np.where(df['booknum'].isnull(),"NO_BOOKING_NUM",df['booknum'])
        df['dateCompleted'] = pd.to_datetime(df['dateCompleted'],errors='coerce')
        df['dateReopened'] = pd.to_datetime(df['dateReopened'],errors='coerce')
        df['datetxn'] = pd.to_datetime(df['datetxn'],errors='coerce')
        df_with_open_flag = aggregate_by_booking_policy_claim(df,transaction_view=True).sort_values('incurred',ascending=False)
        df_with_open_flag = df_with_open_flag.join(df[['clmNum','clmCause']].drop_duplicates().set_index('clmNum'),how='left',on=['clmNum'])
        os.makedirs('./_data',exist_ok=True)
        df_with_open_flag.to_parquet(parquet_file)
    else:
        print(f"load parquet file: {parquet_file}")
        df_with_open_flag = pd.read_parquet(parquet_file)
    return df_with_open_flag


def load_data(report_date = None):
    """
    Load the data and return the dataframes:
    df_raw_txn: raw transaction data containing all transactions
    df_raw_final: final transaction data
    closed_txn: closed claims containing all transactions
    paid_txn: paid claims containing all transactions
    paid_final: final paid claims
    open_final: open claims

    The final dataframes (_final) are used to get the most updated data for each claim.
    The transaction data (_txn) is used to describe the lifetime of a claim per transaction date.

    If the report date is provided (as a date or datetime object), it will be used to filter the data as follows:
    - any claims not closed by the report date is still open -> we will erase the current clmStatus to reflect that
    - any claims closed and not reopened by the report date is closed -> we will erase the current clmStatus to reflect that
    - any claims closed and reopened by the report date is reopened -> we will erase the current clmStatus to reflect that
    - any claims with a first transaction date before the report date will be deleted
    """
    
    def get_final_data(df, group_col='clmNum', date_col='datetxn'):
        """Helper function to get final transaction data for each claim"""
        return df.groupby(group_col).apply(lambda x: x.loc[x[date_col].idxmax()]).reset_index(drop=True)
    
    def sort_dataframe(df, sort_cols=['clmNum', 'datetxn'], ascending=True):
        """Helper function to sort dataframe by specified columns"""
        return df.sort_values(by=sort_cols, ascending=ascending).copy()
    
    def filter_data(df, report_date):
        """Helper function to filter the data based on the report date"""
        if report_date is not None:
            # Create a copy to avoid modifying the original dataframe
            df_filtered = df.copy()
            
            # report_date should be a pandas Timestamp (converted from date/datetime in load_data)
            # Filter transactions: keep only transactions that occurred before or on the report date
            df_filtered = df_filtered[df_filtered['datetxn'] <= report_date]
            
            # Get the latest transaction for each claim (closest to but not after report date)
            latest_claims = df_filtered.groupby('clmNum').agg({
                'datetxn': 'max',
                'dateCompleted': 'last',
                'dateReopened': 'last',  # Get the last value (most recent)
                'clmStatus': 'last',      # Get the last status
                'policy_has_open_claims': 'last',  # For dummy data
                'paid_cumsum': 'sum'
            }).reset_index()
            
            # Determine the correct status for each claim based on report date
            def determine_claim_status(row):
                if pd.notna(row['dateReopened']):
                    if row['dateReopened'] <= report_date:
                        return row['clmStatus'] 
                    elif row['dateReopened'] > report_date and row['dateCompleted']<= report_date:
                        row['dateReopened'] = pd.na()
                        if row['paid_cumsum'] > 0:
                            return 'PAID'
                        else:
                            return 'CLOSED'
                    elif row['dateReopened'] > report_date and row['dateCompleted'] > report_date:
                        row['dateReopened'] = pd.na()
                        row['dateCompleted'] = pd.na()
                        return 'OPEN'
                    else:
                        return row['clmStatus'] 
                else:
                    if row['dateCompleted'] > report_date:
                        return 'OPEN'
                    else:
                        return row['clmStatus'] 
            
            latest_claims['new_status'] = latest_claims.apply(determine_claim_status, axis=1)
            
            # Create a mapping from claim number to new status
            status_mapping = dict(zip(latest_claims['clmNum'], latest_claims['new_status']))
            
            # Apply the status mapping to all transactions
            df_filtered['clmStatus'] = df_filtered['clmNum'].map(status_mapping)
            
            # For dummy data, also update the policy_has_open_claims flag
            if 'policy_has_open_claims' in df_filtered.columns:
                df_filtered['policy_has_open_claims'] = df_filtered['clmNum'].map(
                    dict(zip(latest_claims['clmNum'], 
                            latest_claims['new_status'] == 'OPEN'))
                )
            
            return df_filtered
        
        return df

    df_raw_txn = import_data()
    # For real data, use clmStatus
    closed_txn = df_raw_txn[df_raw_txn['policy_has_open_claims'] == False]
    paid_txn = closed_txn[closed_txn['clmStatus'].isin(['PAID'])]

    # Get open claims
    open_txn = df_raw_txn[df_raw_txn['policy_has_open_claims'] == True]
    
    # Apply report date filtering if provided
    if report_date is not None:
        df_raw_txn = filter_data(df_raw_txn, report_date)
        closed_txn = filter_data(closed_txn, report_date)
        open_txn = filter_data(open_txn, report_date)
        paid_txn = filter_data(paid_txn, report_date)
    
    # Sort all transaction data
    df_raw_txn = sort_dataframe(df_raw_txn)
    closed_txn = sort_dataframe(closed_txn)
    open_txn = sort_dataframe(open_txn)
    paid_txn = sort_dataframe(paid_txn)
    
    # Get final data for each claim type
    df_raw_final = get_final_data(df_raw_txn)
    closed_final = get_final_data(closed_txn)
    paid_final = get_final_data(paid_txn)
    open_final = get_final_data(open_txn)
    
    return (df_raw_txn, closed_txn, open_txn, paid_txn, 
            df_raw_final, closed_final, paid_final, open_final)