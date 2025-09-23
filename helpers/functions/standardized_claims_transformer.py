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
        
        # Normalization parameters for incremental payments
        self.normalization_params = {
            'paid': {'mean': None, 'std': None},
            'expense': {'mean': None, 'std': None}
        }
        
        # Flag to track if normalization has been calculated
        self.normalization_computed = False
    
    def calculate_normalization_parameters(self, df_txn: pd.DataFrame, df_final: pd.DataFrame) -> None:
        """
        Calculate normalization parameters from completed claims only
        
        Args:
            df_txn: Raw transaction data
            df_final: Final claim status data
        """
        # Step 1: Filter to completed claims only
        completed_statuses = ['PAID', 'DENIED', 'CLOSED']
        completed_claims = df_final[df_final['clmStatus'].isin(completed_statuses)]['clmNum'].unique()
        
        # Get transactions for completed claims only
        df_completed_txn = df_txn[df_txn['clmNum'].isin(completed_claims)].copy()
        
        if len(df_completed_txn) == 0:
            print("Warning: No completed claims found for normalization")
            return
        
        # Step 2: Calculate normalization parameters for each payment type
        payment_types = ['paid', 'expense']
        
        for payment_type in payment_types:
            # Get only positive incremental payments (exclude zeros)
            positive_payments = df_completed_txn[df_completed_txn[payment_type] > 0][payment_type]
            
            if len(positive_payments) > 0:
                mean_val = positive_payments.mean()
                std_val = positive_payments.std()
                
                self.normalization_params[payment_type]['mean'] = mean_val
                self.normalization_params[payment_type]['std'] = std_val
                
                print(f"Normalization parameters for {payment_type}:")
                print(f"  Mean: {mean_val:.2f}")
                print(f"  Std:  {std_val:.2f}")
                print(f"  Sample size: {len(positive_payments)}")
            else:
                print(f"Warning: No positive {payment_type} payments found")
                self.normalization_params[payment_type]['mean'] = 0.0
                self.normalization_params[payment_type]['std'] = 1.0
        
        self.normalization_computed = True
    
    def apply_normalization(self, df_txn: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization to raw transaction data
        
        Args:
            df_txn: Raw transaction dataframe
            
        Returns:
            Dataframe with normalized payment columns added
        """
        if not self.normalization_computed:
            print("Warning: Normalization parameters not computed. Run calculate_normalization_parameters first.")
            return df_txn
        
        df_normalized = df_txn.copy()
        
        # Apply z-score normalization to each payment type
        for payment_type in ['paid', 'expense']:
            params = self.normalization_params[payment_type]
            
            if params['mean'] is not None and params['std'] is not None and params['std'] > 0:
                # Apply normalization: (x - mean) / std
                df_normalized[f'{payment_type}_normalized'] = (
                    (df_normalized[payment_type] - params['mean']) / params['std']
                )
            else:
                # If no valid parameters, set normalized values to 0
                df_normalized[f'{payment_type}_normalized'] = 0.0
        
        return df_normalized
    
    def create_normalization_visualization(self, df_txn: pd.DataFrame, df_final: pd.DataFrame) -> None:
        """
        Create sanity check visualization for normalization parameters
        
        Args:
            df_txn: Raw transaction data
            df_final: Final claim status data
        """
        if not self.normalization_computed:
            print("Warning: Normalization parameters not computed. Run calculate_normalization_parameters first.")
            return
        
        # Filter to completed claims only (same logic as normalization calculation)
        completed_statuses = ['PAID', 'DENIED', 'CLOSED']
        completed_claims = df_final[df_final['clmStatus'].isin(completed_statuses)]['clmNum'].unique()
        df_completed_txn = df_txn[df_txn['clmNum'].isin(completed_claims)].copy()
        
        if len(df_completed_txn) == 0:
            print("Warning: No completed claims found for visualization")
            return
        
        # Apply normalization to get normalized values
        df_normalized = self.apply_normalization(df_completed_txn)
        
        # Create visualization
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Original Paid Distribution',
                'Normalized Paid Distribution', 
                'Original Expense Distribution',
                'Normalized Expense Distribution'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot original and normalized distributions
        payment_types = ['paid', 'expense']
        colors = ['blue', 'red']
        
        for i, payment_type in enumerate(payment_types):
            # Get positive values only
            original_positive = df_completed_txn[df_completed_txn[payment_type] > 0][payment_type]
            normalized_positive = df_normalized[df_normalized[payment_type] > 0][f'{payment_type}_normalized']
            
            # Original distribution
            fig.add_trace(
                go.Histogram(
                    x=original_positive,
                    name=f'Original {payment_type}',
                    marker_color=colors[i],
                    opacity=0.7,
                    nbinsx=50
                ),
                row=1 + i, col=1
            )
            
            # Normalized distribution
            fig.add_trace(
                go.Histogram(
                    x=normalized_positive,
                    name=f'Normalized {payment_type}',
                    marker_color=colors[i],
                    opacity=0.7,
                    nbinsx=50
                ),
                row=1 + i, col=2
            )
        
        # Add vertical lines for means
        for i, payment_type in enumerate(payment_types):
            params = self.normalization_params[payment_type]
            
            # Original mean
            fig.add_vline(
                x=params['mean'],
                line_dash="dash",
                line_color=colors[i],
                row=1 + i, col=1,
                annotation_text=f"Mean: {params['mean']:.2f}"
            )
            
            # Normalized mean (should be ~0)
            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color=colors[i],
                row=1 + i, col=2,
                annotation_text="Mean: ~0"
            )
        
        fig.update_layout(
            height=800,
            title_text="Normalization Sanity Check: Before vs After",
            title_x=0.5,
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Amount ($)", row=1, col=1)
        fig.update_xaxes(title_text="Normalized Value", row=1, col=2)
        fig.update_xaxes(title_text="Amount ($)", row=2, col=1)
        fig.update_xaxes(title_text="Normalized Value", row=2, col=2)
        
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("NORMALIZATION SANITY CHECK SUMMARY")
        print("="*60)
        
        for payment_type in payment_types:
            params = self.normalization_params[payment_type]
            original_positive = df_completed_txn[df_completed_txn[payment_type] > 0][payment_type]
            normalized_positive = df_normalized[df_normalized[payment_type] > 0][f'{payment_type}_normalized']
            
            print(f"\n{payment_type.upper()}:")
            print(f"  Original - Mean: {original_positive.mean():.2f}, Std: {original_positive.std():.2f}")
            print(f"  Normalized - Mean: {normalized_positive.mean():.4f}, Std: {normalized_positive.std():.4f}")
            print(f"  Sample size: {len(original_positive)}")
            print(f"  Normalization params - Mean: {params['mean']:.2f}, Std: {params['std']:.2f}")
        
        print("="*60)
        
        return fig
        
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
        
        # Apply normalization if parameters have been computed
        if self.normalization_computed:
            df_txn = self.apply_normalization(df_txn)
        
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
    
    def transform_claims_data_vectorized(self, df_txn: pd.DataFrame) -> pd.DataFrame:
        """
        Fast vectorized transformation to get period data for all claims
        
        Returns a DataFrame with all periods from all claims for fast analysis
        """
        if len(df_txn) == 0:
            return pd.DataFrame()
        
        # Apply normalization if parameters have been computed
        if self.normalization_computed:
            df_txn = self.apply_normalization(df_txn)
        
        # Sort by claim and transaction date
        df_sorted = df_txn.sort_values([self.config.claim_number_col, self.config.transaction_date_col])
        
        # Vectorized approach: Group by claim and process in parallel
        all_periods = []
        
        # Group by claim number
        grouped = df_sorted.groupby(self.config.claim_number_col)
        
        for claim_num, claim_group in grouped:
            try:
                # Get date received (first transaction date for this claim)
                date_received = claim_group[self.config.transaction_date_col].min()
                
                # Calculate periods vectorized
                periods_data = self._create_periods_vectorized(claim_group, date_received, claim_num)
                all_periods.extend(periods_data)
                
            except Exception as e:
                # Skip problematic claims silently
                continue
        
        # Convert to DataFrame
        if all_periods:
            return pd.DataFrame(all_periods)
        else:
            return pd.DataFrame()
    
    def _create_periods_vectorized(self, claim_group: pd.DataFrame, date_received: pd.Timestamp, claim_num: str) -> List[Dict]:
        """
        Create periods for a single claim using vectorized operations
        """
        periods = []
        
        # Calculate the maximum period needed
        max_date = claim_group[self.config.transaction_date_col].max()
        max_days = (max_date - date_received).days
        max_period_needed = min(max_days // self.config.period_length_days + 1, self.config.max_periods)
        
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
            period_start_days = period * self.config.period_length_days
            period_end_days = (period + 1) * self.config.period_length_days
            
            period_start = date_received + timedelta(days=period_start_days)
            period_end = date_received + timedelta(days=period_end_days)
            
            # Vectorized period calculation
            period_mask = (
                (claim_group[self.config.transaction_date_col] >= period_start) &
                (claim_group[self.config.transaction_date_col] < period_end)
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
                incremental_paid = period_transactions[self.config.paid_amount_col].sum()
                incremental_expense = period_transactions[self.config.expense_amount_col].sum()
                incremental_recovery = period_transactions[self.config.recovery_amount_col].sum() if self.config.recovery_amount_col in claim_group.columns else 0.0
                incremental_reserve = period_transactions[self.config.reserve_amount_col].sum() if self.config.reserve_amount_col in claim_group.columns else 0.0
                
                # Calculate normalized amounts
                if self.normalization_computed:
                    incremental_paid_normalized = period_transactions[f'{self.config.paid_amount_col}_normalized'].sum()
                    incremental_expense_normalized = period_transactions[f'{self.config.expense_amount_col}_normalized'].sum()
                else:
                    incremental_paid_normalized = 0.0
                    incremental_expense_normalized = 0.0
            
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
        
        # Initialize normalized cumulative amounts
        cumulative_paid_normalized = 0.0
        cumulative_expense_normalized = 0.0
        
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
            
            # Calculate normalized incremental amounts for this period
            if self.normalization_computed:
                incremental_paid_normalized = period_transactions[f'{self.config.paid_amount_col}_normalized'].sum()
                incremental_expense_normalized = period_transactions[f'{self.config.expense_amount_col}_normalized'].sum()
            else:
                incremental_paid_normalized = 0.0
                incremental_expense_normalized = 0.0
            
            # Update cumulative amounts
            cumulative_paid += incremental_paid
            cumulative_expense += incremental_expense
            cumulative_recovery += incremental_recovery
            cumulative_reserve += incremental_reserve
            cumulative_incurred = cumulative_paid + cumulative_expense + cumulative_recovery + cumulative_reserve
            
            # Update normalized cumulative amounts
            cumulative_paid_normalized += incremental_paid_normalized
            cumulative_expense_normalized += incremental_expense_normalized
            
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
                incremental_paid_normalized=incremental_paid_normalized,
                incremental_expense_normalized=incremental_expense_normalized,
                cumulative_paid=cumulative_paid,
                cumulative_expense=cumulative_expense,
                cumulative_recovery=cumulative_recovery,
                cumulative_reserve=cumulative_reserve,
                cumulative_incurred=cumulative_incurred,
                cumulative_paid_normalized=cumulative_paid_normalized,
                cumulative_expense_normalized=cumulative_expense_normalized,
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
