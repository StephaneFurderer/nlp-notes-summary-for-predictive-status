import pandas as pd
import numpy as np
import os

import json
import logging
from typing import Optional
from .claims_data_schema import clean_and_convert_dataframe
from .CONST import BASE_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#----------------------------------------------------------
# Deprecated: claim features for predictive models
#----------------------------------------------------------
def calculate_claim_features_vectorized(df_txn):
    """Vectorized calculation of claim features using pandas groupby operations"""
    if len(df_txn) == 0:
        empty_df = pd.DataFrame()
        empty_summary = pd.DataFrame()
        empty_display = pd.DataFrame()
        return empty_df, empty_summary, empty_display

    # Ensure data is sorted
    df_sorted = df_txn.sort_values(['clmNum', 'datetxn']).copy()

    # Group by claim for vectorized operations
    claim_groups = df_sorted.groupby('clmNum')

    # Basic metrics using vectorized operations
    basic_metrics = claim_groups.agg({
        'datetxn': ['count', 'min', 'max'],
        'clmStatus': 'last',
        'clmCause': 'first',
        'dateReopened': 'last',
        'paid_cumsum': 'last',
        'expense_cumsum': 'last',
        'recovery_cumsum': 'last',
        'reserve_cumsum': 'last',
        'incurred_cumsum': 'last'
    }).reset_index()

    # Flatten column names
    basic_metrics.columns = ['clmNum', 'num_transactions', 'first_txn_date', 'last_txn_date',
                            'clmStatus', 'clmCause', 'dateReopened', 'current_paid',
                            'current_expense', 'current_recovery', 'current_reserve', 'current_incurred']

    # Calculate days since first transaction
    basic_metrics['days_since_first_txn'] = (basic_metrics['last_txn_date'] - basic_metrics['first_txn_date']).dt.days

    # Calculate time to first payment/expense vectorized
    def calc_time_to_first(group):
        first_paid = group[group['paid_cumsum'] > 0]['datetxn'].min()
        first_expense = group[group['expense_cumsum'] > 0]['datetxn'].min()
        min_date = group['datetxn'].min()
        max_days = (group['datetxn'].max() - min_date).days

        return pd.Series({
            'time_to_first_paid': (first_paid - min_date).days if pd.notna(first_paid) else max_days,
            'time_to_first_expense': (first_expense - min_date).days if pd.notna(first_expense) else max_days
        })

    time_metrics = claim_groups.apply(calc_time_to_first).reset_index()

    # Transaction counts vectorized
    def calc_txn_counts(group):
        return pd.Series({
            'num_expense_txns': (group['expense'] > 0).sum(),
            'num_paid_txns': (group['paid'] > 0).sum()
        })

    txn_counts = claim_groups.apply(calc_txn_counts).reset_index()

    # Change metrics vectorized
    def calc_change_metrics(group):
        reserve_changes = group[group['reserve_change'] != 0]['reserve_change']
        incurred_changes = group[group['incurred_change'] != 0]['incurred_change']

        return pd.Series({
            'avg_reserve_change': reserve_changes.mean() if len(reserve_changes) > 0 else 0,
            'avg_incurred_change': incurred_changes.mean() if len(incurred_changes) > 0 else 0
        })

    change_metrics = claim_groups.apply(calc_change_metrics).reset_index()

    # Merge all metrics
    features_df = basic_metrics.merge(time_metrics, on='clmNum')
    features_df = features_df.merge(txn_counts, on='clmNum')
    features_df = features_df.merge(change_metrics, on='clmNum')

    # Calculate derived metrics
    features_df['avg_expense_amount'] = features_df['current_expense'] / np.maximum(features_df['num_expense_txns'], 1)
    features_df['avg_paid_amount'] = features_df['current_paid'] / np.maximum(features_df['num_paid_txns'], 1)
    features_df['current_cashflow'] = features_df['current_paid'] + features_df['current_expense']
    features_df['paid_to_incurred_ratio'] = features_df['current_paid'] / np.maximum(features_df['current_incurred'], 1)
    features_df['expense_to_incurred_ratio'] = features_df['current_expense'] / np.maximum(features_df['current_incurred'], 1)
    features_df['development_stage'] = np.minimum(features_df['days_since_first_txn'] / 1825, 1.0)
    features_df['isReopened'] = features_df['dateReopened'].notna().astype(int)

    # Drop temporary date columns
    features_df = features_df.drop(['first_txn_date', 'last_txn_date'], axis=1)

    # Create summary statistics
    summary_cols = ['num_transactions', 'days_since_first_txn', 'num_expense_txns', 'num_paid_txns',
                    'current_reserve', 'current_incurred', 'current_paid', 'current_expense']

    summary_stats = features_df.groupby(['clmCause', 'clmStatus'])[summary_cols].describe()

    # Create display dataframe
    display_cols = ['clmCause', 'clmNum', 'num_transactions', 'days_since_first_txn', 'num_expense_txns',
                    'num_paid_txns', 'current_reserve', 'current_incurred', 'current_paid', 'current_expense', 'current_cashflow']
    display_df = features_df[display_cols].copy()
    display_df.columns = ['Claim Clause', 'Claim Number', 'Transactions', 'Days Since First Txn', 'Expense Txns',
                         'Paid Txns', 'Current Reserve', 'Current Incurred', 'Current Paid', 'Current Expense', 'Current Cashflow']

    return features_df, summary_stats, display_df

def calculate_claim_features(df_open_txn, use_cache=True, force_recalculate=False):
    """Calculate comprehensive metrics for claims with caching and vectorization"""
    if len(df_open_txn) == 0:
        empty_df = pd.DataFrame()
        empty_summary = pd.DataFrame()
        empty_display = pd.DataFrame()
        return empty_df, empty_summary, empty_display

    # Generate data hash for caching
    data_hash = _generate_data_hash(df_open_txn)
    cache_path = _get_cache_path(data_hash)

    # Try to load from cache first
    if use_cache and not force_recalculate and os.path.exists(cache_path):
        print(f"Loading cached claim features from: {cache_path}")
        cached_results = pd.read_parquet(cache_path)

        # Recreate summary and display dataframes
        summary_cols = ['num_transactions', 'days_since_first_txn', 'num_expense_txns', 'num_paid_txns',
                       'current_reserve', 'current_incurred', 'current_paid', 'current_expense']
        summary_stats = cached_results.groupby(['clmCause', 'clmStatus'])[summary_cols].describe()

        display_cols = ['clmCause', 'clmNum', 'num_transactions', 'days_since_first_txn', 'num_expense_txns',
                       'num_paid_txns', 'current_reserve', 'current_incurred', 'current_paid', 'current_expense', 'current_cashflow']
        display_df = cached_results[display_cols].copy()
        display_df.columns = ['Claim Clause', 'Claim Number', 'Transactions', 'Days Since First Txn', 'Expense Txns',
                             'Paid Txns', 'Current Reserve', 'Current Incurred', 'Current Paid', 'Current Expense', 'Current Cashflow']

        return cached_results, summary_stats, display_df

    # Calculate features using vectorized approach
    print("Calculating claim features using vectorized operations...")
    features_df, summary_stats, display_df = calculate_claim_features_vectorized(df_open_txn)

    # Save to cache
    if use_cache and len(features_df) > 0:
        features_df.to_parquet(cache_path)
        print(f"Cached claim features to: {cache_path}")

    return features_df, summary_stats, display_df   








# ----------------------------------------------------------
# Useful functions: claims transformation from raw to transaction and final data model
# ----------------------------------------------------------
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

# def import_data(extraction_date=None):
#     """
#     Load and process claims data using DataLoader

#     Args:
#         extraction_date: Date string in YYYY-MM-DD format (e.g., '2025-09-21')

#     Returns:
#         Processed DataFrame with claim features
#     """
#     from .load_cache_data import DataLoader

#     # Initialize DataLoader
#     data_loader = DataLoader()

#     # If extraction_date is provided, use structured data loading
#     if extraction_date:
#         df = data_loader.load_claims_data(extraction_date=extraction_date)
#         if df is None:
#             raise FileNotFoundError(f"No claims data found for extraction date {extraction_date}")
#     else:
#         # Fallback: try to find the most recent extraction date
#         available_versions = data_loader.get_available_data_versions()
#         if not available_versions:
#             raise FileNotFoundError("No organized claims data found. Please specify extraction_date.")

#         # Use the most recent extraction date
#         latest_date = available_versions[0]['extraction_date']
#         print(f"Using latest available extraction date: {latest_date}")
#         df = data_loader.load_claims_data(extraction_date=latest_date)
#         if df is None:
#             raise FileNotFoundError(f"Failed to load claims data for {latest_date}")

#     # Process the data
#     df['booknum'] = np.where(df['booknum'].isnull(), "NO_BOOKING_NUM", df['booknum'])
#     df['dateCompleted'] = pd.to_datetime(df['dateCompleted'], errors='coerce')
#     df['dateReopened'] = pd.to_datetime(df['dateReopened'], errors='coerce')
#     df['datetxn'] = pd.to_datetime(df['datetxn'], errors='coerce')

#     # Aggregate and process
#     df_with_open_flag = aggregate_by_booking_policy_claim(df, transaction_view=True).sort_values('incurred', ascending=False)
#     df_with_open_flag = df_with_open_flag.join(df[['clmNum','clmCause']].drop_duplicates().set_index('clmNum'), how='left', on=['clmNum'])

#     return df_with_open_flag


def _aggregate_by_booking_policy_claim(df):
    df_with_open_flag = aggregate_by_booking_policy_claim(df, transaction_view=True).sort_values('incurred', ascending=False)
    df_with_open_flag = df_with_open_flag.join(df[['clmNum','clmCause']].drop_duplicates().set_index('clmNum'), how='left', on=['clmNum'])
    return df_with_open_flag

def transform_claims_raw_data(df_raw_txn,report_date=None):
    """
    Load the data and return the dataframes:
    df_raw_txn: raw transaction data containing all transactions
    df_raw_final: final transaction data
    closed_txn: closed claims containing all transactions
    paid_txn: paid claims containing all transactions
    paid_final: final paid claims
    open_final: open claims

    Args:
        report_date: Date or datetime object for temporal filtering
        extraction_date: Date string in YYYY-MM-DD format (e.g., '2025-09-21') for data source

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
        """Helper function to filter the data based on the report date and change the clmStatus"""
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


    # aggregate the data
    df_raw_txn = _aggregate_by_booking_policy_claim(df_raw_txn)

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


#----------------------------------------------------------
# Main endpoints : load and transform claims data
#----------------------------------------------------------

LIST_DATAFRAMES = ['df_raw_txn', 'closed_txn', 'open_txn', 'paid_txn', 'df_raw_final', 'closed_final', 'paid_final', 'open_final']

def load_claims_data(extraction_date: Optional[str] = None, claims_file: Optional[str] = None) -> Optional[pd.DataFrame]:
    if extraction_date:
        claims_file = os.path.join(BASE_DATA_DIR, extraction_date, "clm_with_amt.csv")
        if not os.path.exists(claims_file):
            logger.info(f"No claims file found for extraction date {extraction_date}")
            return None
    if claims_file and os.path.exists(claims_file):
        parquet_file = claims_file.replace('.csv', '.parquet')
        if os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
        else:
            df = clean_and_convert_dataframe(pd.read_csv(claims_file))
            df.to_parquet(parquet_file)
        logger.info(f"Loaded claims data: {len(df):,} transactions from {df['clmNum'].nunique():,} claims")
        return df
    logger.info(f"Claims file not found: {claims_file}")
    return None

def save_transformed_claims_data(extraction_date, dataframes):
    os.makedirs(f"{BASE_DATA_DIR}/{extraction_date}", exist_ok=True)
    for name, df in zip(LIST_DATAFRAMES, dataframes):
        df.to_parquet(f"{BASE_DATA_DIR}/{extraction_date}/{name}.parquet")
    with open(f"{BASE_DATA_DIR}/{extraction_date}/parquet_file_paths.json", "w") as f:
        json.dump({name: f"{BASE_DATA_DIR}/{extraction_date}/{name}.parquet" for name in LIST_DATAFRAMES}, f)

def load_transformed_claims_data(extraction_date=None):
    raw_claim_data = load_claims_data(extraction_date=extraction_date)
    dataframes = transform_claims_raw_data(raw_claim_data)
    save_transformed_claims_data(extraction_date, dataframes)
    return dataframes

def read_transformed_claims_data_from_parquet(extraction_date=None):
    dfs = []
    for name in LIST_DATAFRAMES:
        path = f"{BASE_DATA_DIR}/{extraction_date}/{name}.parquet"
        dfs.append(pd.read_parquet(path) if os.path.exists(path) else None)
    #if one is none, run the load_transformed_claims_data function
    if any(df is None for df in dfs):
        dfs = load_transformed_claims_data(extraction_date)
    return tuple(dfs)

if __name__ == "__main__":
    """ 
    Example usage:
    python -m helpers.functions.claims_utils
    to read the transformed claims data from the parquet files if they exist, otherwise it will load the raw claims data and transform it
    """
    extraction_date = "2025-09-21"
    dataframes = read_transformed_claims_data_from_parquet(extraction_date)
    print(dataframes)