"""
Standardized Claims Transformer for Micro-Level Reserving

Transforms raw transaction data into standardized 30-day periods following the framework
described in "Micro-level reserving for general insurance claims using a long short-term memory network"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import warnings
from .standardized_claims_schema import (
    StandardizedClaim, StaticClaimContext, DynamicClaimPeriod, 
    StandardizedClaimsDataset, StandardizationConfig, validate_standardized_claim
)

warnings.filterwarnings('ignore')


class StandardizedClaimsTransformer:
    """
    Transforms raw transaction data into standardized claims format
    """
    
    def __init__(self, config: Optional[StandardizationConfig] = None):
        self.config = config or StandardizationConfig()
        
    def transform_claims_data(self, df_txn: pd.DataFrame) -> StandardizedClaimsDataset:
        """
        Transform raw transaction data into standardized format
        
        Args:
            df_txn: Raw transaction dataframe with columns matching StandardizationConfig
            
        Returns:
            StandardizedClaimsDataset with all claims transformed
        """
        if len(df_txn) == 0:
            return StandardizedClaimsDataset(
                claims=[],
                total_claims=0,
                total_periods=0,
                metadata={'error': 'Empty input dataframe'}
            )
        
        # Validate required columns
        required_cols = [
            self.config.claim_number_col,
            self.config.transaction_date_col,
            self.config.paid_amount_col,
            self.config.expense_amount_col
        ]
        
        missing_cols = [col for col in required_cols if col not in df_txn.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort data by claim number and transaction date
        df_sorted = df_txn.sort_values([self.config.claim_number_col, self.config.transaction_date_col]).copy()
        
        # Get unique claims
        unique_claims = df_sorted[self.config.claim_number_col].unique()
        
        standardized_claims = []
        total_periods = 0
        
        print(f"Transforming {len(unique_claims)} claims...")
        
        for i, claim_num in enumerate(unique_claims):
            if i % 100 == 0:
                print(f"Processing claim {i+1}/{len(unique_claims)}: {claim_num}")
                
            try:
                standardized_claim = self._transform_single_claim(
                    df_sorted[df_sorted[self.config.claim_number_col] == claim_num]
                )
                if standardized_claim:
                    standardized_claims.append(standardized_claim)
                    total_periods += len(standardized_claim.dynamic_periods)
            except Exception as e:
                print(f"Error processing claim {claim_num}: {str(e)}")
                continue
        
        # Create dataset
        dataset = StandardizedClaimsDataset(
            claims=standardized_claims,
            total_claims=len(standardized_claims),
            total_periods=total_periods,
            metadata={
                'config': self.config.dict(),
                'transformation_date': datetime.now().isoformat(),
                'input_rows': len(df_txn),
                'successful_transforms': len(standardized_claims)
            }
        )
        
        print(f"Transformation completed: {len(standardized_claims)} claims, {total_periods} total periods")
        return dataset
    
    def _transform_single_claim(self, claim_df: pd.DataFrame) -> Optional[StandardizedClaim]:
        """
        Transform a single claim's transaction data into standardized format
        """
        if len(claim_df) == 0:
            return None
        
        # Get claim number
        claim_num = claim_df[self.config.claim_number_col].iloc[0]
        
        # Create static context
        static_context = self._create_static_context(claim_df)
        
        # Create dynamic periods
        dynamic_periods = self._create_dynamic_periods(claim_df, static_context.dateReceived)
        
        if len(dynamic_periods) < self.config.min_periods:
            print(f"Claim {claim_num}: Not enough periods ({len(dynamic_periods)} < {self.config.min_periods})")
            return None
        
        # Calculate summary metrics - use LAST period's cumulative values (not max)
        # This correctly represents the final cash flow position for reserving
        if dynamic_periods:
            last_period = dynamic_periods[-1]
            total_paid = last_period.cumulative_paid
            total_expense = last_period.cumulative_expense
            total_recovery = last_period.cumulative_recovery
            final_incurred = last_period.cumulative_incurred
            final_reserve = last_period.cumulative_reserve
        else:
            # No periods available
            total_paid = 0.0
            total_expense = 0.0
            total_recovery = 0.0
            final_incurred = 0.0
            final_reserve = 0.0
        
        # Create standardized claim
        standardized_claim = StandardizedClaim(
            static_context=static_context,
            dynamic_periods=dynamic_periods,
            total_periods=len(dynamic_periods),
            max_period=max([p.period for p in dynamic_periods], default=0),
            total_paid=total_paid,
            total_expense=total_expense,
            total_recovery=total_recovery,
            final_incurred=final_incurred,
            final_reserve=final_reserve
        )
        
        # Validate the claim
        if not validate_standardized_claim(standardized_claim):
            print(f"Validation failed for claim {claim_num}")
            return None
        
        return standardized_claim
    
    def _create_static_context(self, claim_df: pd.DataFrame) -> StaticClaimContext:
        """
        Create static context for a claim
        """
        # Get the first transaction date as the claim receipt date
        date_received = claim_df[self.config.transaction_date_col].min()
        
        # Get static information (use first occurrence for most fields)
        clm_cause = claim_df[self.config.claim_cause_col].iloc[0] if self.config.claim_cause_col in claim_df.columns else "UNKNOWN"
        booknum = claim_df[self.config.booking_number_col].iloc[0] if self.config.booking_number_col in claim_df.columns else "UNKNOWN"
        cidpol = claim_df[self.config.policy_id_col].iloc[0] if self.config.policy_id_col in claim_df.columns else "UNKNOWN"
        clm_status = claim_df[self.config.claim_status_col].iloc[-1] if self.config.claim_status_col in claim_df.columns else "UNKNOWN"
        
        # Get completion and reopen dates (use last occurrence)
        date_completed = None
        date_reopened = None
        
        if self.config.date_completed_col in claim_df.columns:
            completed_dates = claim_df[self.config.date_completed_col].dropna()
            date_completed = completed_dates.iloc[-1] if len(completed_dates) > 0 else None
            
        if self.config.date_reopened_col in claim_df.columns:
            reopened_dates = claim_df[self.config.date_reopened_col].dropna()
            date_reopened = reopened_dates.iloc[-1] if len(reopened_dates) > 0 else None
        
        # Derived features
        is_reopened = date_reopened is not None
        policy_has_open_claims = claim_df.get('policy_has_open_claims', pd.Series([False])).iloc[-1]
        policy_has_reopen_claims = claim_df.get('policy_has_reopen_claims', pd.Series([False])).iloc[-1]
        
        return StaticClaimContext(
            clmNum=claim_df[self.config.claim_number_col].iloc[0],
            clmCause=clm_cause,
            booknum=booknum,
            cidpol=cidpol,
            dateReceived=date_received,
            clmStatus=clm_status,
            dateCompleted=date_completed,
            dateReopened=date_reopened,
            isReopened=is_reopened,
            policy_has_open_claims=policy_has_open_claims,
            policy_has_reopen_claims=policy_has_reopen_claims
        )
    
    def _create_dynamic_periods(self, claim_df: pd.DataFrame, date_received: datetime) -> List[DynamicClaimPeriod]:
        """
        Create dynamic periods for a claim
        """
        periods = []
        
        # Calculate the maximum period needed
        max_date = claim_df[self.config.transaction_date_col].max()
        max_days = (max_date - date_received).days
        max_period_needed = min(max_days // self.config.period_length_days + 1, self.config.max_periods)
        
        print(f"Creating periods: max_days={max_days}, period_length={self.config.period_length_days}, max_period_needed={max_period_needed}")
        
        # Initialize cumulative amounts
        cumulative_paid = 0.0
        cumulative_expense = 0.0
        cumulative_recovery = 0.0
        cumulative_reserve = 0.0
        cumulative_incurred = 0.0
        
        # Create periods
        for period in range(max_period_needed + 1):
            period_start = date_received + timedelta(days=period * self.config.period_length_days)
            period_end = period_start + timedelta(days=self.config.period_length_days - 1)
            days_from_receipt = period * self.config.period_length_days
            
            # Get transactions in this period
            period_mask = (
                (claim_df[self.config.transaction_date_col] >= period_start) & 
                (claim_df[self.config.transaction_date_col] <= period_end)
            )
            period_transactions = claim_df[period_mask]
            
            # Calculate incremental amounts for this period
            incremental_paid = period_transactions[self.config.paid_amount_col].sum()
            incremental_expense = period_transactions[self.config.expense_amount_col].sum()
            incremental_recovery = period_transactions[self.config.recovery_amount_col].sum() if self.config.recovery_amount_col in claim_df.columns else 0.0
            incremental_reserve = period_transactions[self.config.reserve_amount_col].sum() if self.config.reserve_amount_col in claim_df.columns else 0.0
            
            # Update cumulative amounts
            cumulative_paid += incremental_paid
            cumulative_expense += incremental_expense
            cumulative_recovery += incremental_recovery
            cumulative_reserve += incremental_reserve
            cumulative_incurred = cumulative_paid + cumulative_expense + cumulative_recovery + cumulative_reserve
            
            # Period-specific features
            num_transactions = len(period_transactions)
            has_payment = incremental_paid > 0
            has_expense = incremental_expense > 0
            
            # Time-based features
            days_since_first_txn = days_from_receipt
            development_stage = min(days_since_first_txn / self.config.development_stage_max_days, 1.0)
            
            # Create period
            dynamic_period = DynamicClaimPeriod(
                period=period,
                days_from_receipt=days_from_receipt,
                period_start_date=period_start,
                period_end_date=period_end,
                incremental_paid=incremental_paid,
                incremental_expense=incremental_expense,
                incremental_recovery=incremental_recovery,
                incremental_reserve=incremental_reserve,
                cumulative_paid=cumulative_paid,
                cumulative_expense=cumulative_expense,
                cumulative_recovery=cumulative_recovery,
                cumulative_reserve=cumulative_reserve,
                cumulative_incurred=cumulative_incurred,
                num_transactions=num_transactions,
                has_payment=has_payment,
                has_expense=has_expense,
                days_since_first_txn=days_since_first_txn,
                development_stage=development_stage
            )
            
            periods.append(dynamic_period)
        
        return periods
    
    def get_claim_summary(self, standardized_claim: StandardizedClaim) -> Dict[str, Any]:
        """
        Get a summary of a standardized claim
        """
        return {
            'claim_number': standardized_claim.static_context.clmNum,
            'claim_cause': standardized_claim.static_context.clmCause,
            'claim_status': standardized_claim.static_context.clmStatus,
            'date_received': standardized_claim.static_context.dateReceived,
            'total_periods': standardized_claim.total_periods,
            'max_period': standardized_claim.max_period,
            'total_paid': standardized_claim.total_paid,
            'total_expense': standardized_claim.total_expense,
            'final_incurred': standardized_claim.final_incurred,
            'final_reserve': standardized_claim.final_reserve,
            'is_reopened': standardized_claim.static_context.isReopened,
            'periods_with_payments': sum(1 for p in standardized_claim.dynamic_periods if p.has_payment),
            'periods_with_expenses': sum(1 for p in standardized_claim.dynamic_periods if p.has_expense)
        }
    
    def compare_with_original(self, standardized_claim: StandardizedClaim, original_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare standardized claim with original transaction data
        """
        claim_num = standardized_claim.static_context.clmNum
        original_claim = original_df[original_df[self.config.claim_number_col] == claim_num]
        
        # Calculate totals from original data
        original_total_paid = original_claim[self.config.paid_amount_col].sum()
        original_total_expense = original_claim[self.config.expense_amount_col].sum()
        original_total_recovery = original_claim[self.config.recovery_amount_col].sum() if self.config.recovery_amount_col in original_df.columns else 0.0
        original_total_reserve = original_claim[self.config.reserve_amount_col].sum() if self.config.reserve_amount_col in original_df.columns else 0.0
        
        return {
            'claim_number': claim_num,
            'original_transactions': len(original_claim),
            'standardized_periods': len(standardized_claim.dynamic_periods),
            'total_paid_match': abs(standardized_claim.total_paid - original_total_paid) < 0.01,
            'total_expense_match': abs(standardized_claim.total_expense - original_total_expense) < 0.01,
            'total_recovery_match': abs(standardized_claim.total_recovery - original_total_recovery) < 0.01,
            'total_reserve_match': abs(standardized_claim.final_reserve - original_total_reserve) < 0.01,
            'original_total_paid': original_total_paid,
            'standardized_total_paid': standardized_claim.total_paid,
            'original_total_expense': original_total_expense,
            'standardized_total_expense': standardized_claim.total_expense
        }
