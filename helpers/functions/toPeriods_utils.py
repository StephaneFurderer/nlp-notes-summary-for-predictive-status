"""
Transform claims transactional data to periods data
"""
import pandas as pd
from datetime import timedelta
from typing import List, Dict
import numpy as np

from .standardized_claims_schema import (
    StandardizedClaim, StaticClaimContext, DynamicClaimPeriod, 
    StandardizedClaimsDataset, StandardizationConfig, validate_standardized_claim
)


PERIOD_LENGTH_DAYS = 30
MAX_PERIODS = 60



def _create_periods_vectorized(claim_group: pd.DataFrame, date_received: pd.Timestamp, claim_num: str) -> List[Dict]:
    """
    Create periods for a single claim using vectorized operations
    """
    config = StandardizationConfig()
    periods = []
    
    # Calculate the maximum period needed
    max_date = claim_group[config.transaction_date_col].max()
    max_days = (max_date - date_received).days
    max_period_needed = min(max_days // PERIOD_LENGTH_DAYS + 1, MAX_PERIODS)
    
    # Initialize cumulative amounts
    cumulative_paid = 0.0
    cumulative_expense = 0.0
    cumulative_recovery = 0.0
    cumulative_reserve = 0.0
    cumulative_incurred = 0.0
    cumulative_paid_normalized = 0.0
    cumulative_expense_normalized = 0.0
    
    # Create periods
    for period in range(max_period_needed):
        period_start_days = period * PERIOD_LENGTH_DAYS
        period_end_days = (period + 1) * PERIOD_LENGTH_DAYS
        
        period_start = date_received + timedelta(days=period_start_days)
        period_end = date_received + timedelta(days=period_end_days)
        
        # Vectorized period calculation
        period_mask = (
            (claim_group[config.transaction_date_col] >= period_start) &
            (claim_group[config.transaction_date_col] < period_end)
        )
        
        period_transactions = claim_group[period_mask]
        
        if len(period_transactions) == 0:
            # No transactions in this period, but still create the period with zeros
            incremental_paid = 0.0
            incremental_expense = 0.0
            incremental_recovery = 0.0
            incremental_reserve = 0.0
            incremental_paid_normalized = 0.0
            incremental_expense_normalized = 0.0
        else:
            # Calculate incremental amounts vectorized
            incremental_paid = period_transactions[config.paid_amount_col].sum()
            incremental_expense = period_transactions[config.expense_amount_col].sum()
            incremental_recovery = period_transactions[config.recovery_amount_col].sum() if config.recovery_amount_col in claim_group.columns else 0.0
            incremental_reserve = period_transactions[config.reserve_amount_col].sum() if config.reserve_amount_col in claim_group.columns else 0.0
            
            # # Calculate normalized amounts
            # if normalization_computed:
            #     incremental_paid_normalized = period_transactions[f'{config.paid_amount_col}_normalized'].sum()
            #     incremental_expense_normalized = period_transactions[f'{config.expense_amount_col}_normalized'].sum()
            # else:
            #     incremental_paid_normalized = 0.0
            #     incremental_expense_normalized = 0.0
        
        # Update cumulative amounts
        cumulative_paid += incremental_paid
        cumulative_expense += incremental_expense
        cumulative_recovery += incremental_recovery
        cumulative_reserve += incremental_reserve
        cumulative_incurred = cumulative_paid + cumulative_expense + cumulative_recovery + cumulative_reserve
        cumulative_paid_normalized += incremental_paid_normalized
        cumulative_expense_normalized += incremental_expense_normalized
        
        # Create period data dictionary
        period_data = {
            'clmNum': claim_num,
            'period': period,
            'days_from_receipt': period_start_days,
            'period_start_date': period_start,
            'period_end_date': period_end,
            'incremental_paid': incremental_paid,
            'incremental_expense': incremental_expense,
            'incremental_recovery': incremental_recovery,
            'incremental_reserve': incremental_reserve,
            'incremental_paid_normalized': incremental_paid_normalized,
            'incremental_expense_normalized': incremental_expense_normalized,
            'cumulative_paid': cumulative_paid,
            'cumulative_expense': cumulative_expense,
            'cumulative_recovery': cumulative_recovery,
            'cumulative_reserve': cumulative_reserve,
            'cumulative_incurred': cumulative_incurred,
            'cumulative_paid_normalized': cumulative_paid_normalized,
            'cumulative_expense_normalized': cumulative_expense_normalized,
            'num_transactions': len(period_transactions),
            'has_payment': incremental_paid > 0,
            'has_expense': incremental_expense > 0
        }
        
        periods.append(period_data)
    
    return periods



def create_period_column_fast(df):
    """
    Optimized version of create_period_column for better speed.
    Uses vectorized operations and avoids Python loops.
    Assumes 'datetxn' is datetime and 'clmNum' exists. PERIOD_LENGTH_DAYS must be defined.
    """
    df = df.copy()
    # Calculate min transaction date per claim
    min_dates = df.groupby('clmNum')['datetxn'].transform('min')
    days_from_min = (df['datetxn'] - min_dates).dt.days
    df['period'] = days_from_min // PERIOD_LENGTH_DAYS

    # Aggregate paid and expense per claim/period
    incremental_cols = ['paid','expense','reserve','incurred']
    cumulative_cols = ['paid_cumsum','expense_cumsum','reserve_cumsum','incurred_cumsum']
    agg_cols = {**{col: 'sum' for col in incremental_cols}, **{col: 'max' for col in cumulative_cols}}

    period_agg = df.groupby(['clmNum', 'period']).agg(agg_cols).reset_index()

    # Get min date per claim for period start calculation
    min_date_per_claim = df.groupby('clmNum')['datetxn'].min()
    max_periods = period_agg.groupby('clmNum')['period'].max()

    # Build all combinations of clmNum and period using reindexing (faster than Python loop)
    clmNums = max_periods.index.values
    max_ps = max_periods.values
    period_ranges = [np.arange(0, p+1) for p in max_ps]
    all_clmNums = np.repeat(clmNums, [len(r) for r in period_ranges])
    all_periods = np.concatenate(period_ranges)
    full_periods = pd.DataFrame({'clmNum': all_clmNums, 'period': all_periods})

    # Map min_date to each clmNum
    full_periods = full_periods.merge(min_date_per_claim.rename('min_date'), left_on='clmNum', right_index=True, how='left')
    full_periods['period_start_date'] = full_periods['min_date'] + pd.to_timedelta(full_periods['period'] * PERIOD_LENGTH_DAYS, unit='D')
    full_periods['period_end_date'] = full_periods['period_start_date'] + pd.to_timedelta(PERIOD_LENGTH_DAYS - 1, unit='D')
    full_periods = full_periods.drop(columns='min_date')

    # Merge with actual data, fill missing with 0
    result = pd.merge(
        full_periods,
        period_agg,
        on=['clmNum', 'period'],
        how='left'
    )
    result = result.fillna(0)

    return result

