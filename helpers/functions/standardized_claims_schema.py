"""
Pydantic schemas for standardized claims data following micro-level reserving framework.

Based on the paper: "Micro-level reserving for general insurance claims using a long short-term memory network"
Section 2.2: Dynamic information Dk,j and static context Ck,0
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd


class StaticClaimContext(BaseModel):
    """
    Static context Ck,0 - Claim-level information that doesn't change over time
    """
    clmNum: str = Field(..., description="Claim number - unique identifier")
    clmCause: str = Field(..., description="Claim cause/category")
    booknum: str = Field(..., description="Booking number")
    cidpol: str = Field(..., description="Policy ID")
    dateReceived: datetime = Field(..., description="Date when claim was first received")
    clmStatus: str = Field(..., description="Current claim status")
    dateCompleted: Optional[datetime] = Field(None, description="Date when claim was completed")
    dateReopened: Optional[datetime] = Field(None, description="Date when claim was reopened")
    
    # Derived static features
    isReopened: bool = Field(False, description="Whether claim was reopened")
    policy_has_open_claims: bool = Field(False, description="Whether policy has other open claims")
    policy_has_reopen_claims: bool = Field(False, description="Whether policy has other reopened claims")


class DynamicClaimPeriod(BaseModel):
    """
    Dynamic information Dk,j - Period-specific information that changes over time
    """
    period: int = Field(..., description="Period number (0, 1, 2, ...) representing 30-day windows")
    days_from_receipt: int = Field(..., description="Days from claim receipt (0, 30, 60, ...)")
    period_start_date: datetime = Field(..., description="Start date of the period")
    period_end_date: datetime = Field(..., description="End date of the period")
    
    # Incremental amounts for this period
    incremental_paid: float = Field(0.0, description="Incremental paid amount in this period")
    incremental_expense: float = Field(0.0, description="Incremental expense amount in this period")
    incremental_recovery: float = Field(0.0, description="Incremental recovery amount in this period")
    incremental_reserve: float = Field(0.0, description="Incremental reserve change in this period")
    
    # Normalized incremental amounts for this period (z-score normalized)
    incremental_paid_normalized: float = Field(0.0, description="Normalized incremental paid amount in this period")
    incremental_expense_normalized: float = Field(0.0, description="Normalized incremental expense amount in this period")
    
    # Cumulative amounts up to this period
    cumulative_paid: float = Field(0.0, description="Cumulative paid amount up to this period")
    cumulative_expense: float = Field(0.0, description="Cumulative expense amount up to this period")
    cumulative_recovery: float = Field(0.0, description="Cumulative recovery amount up to this period")
    cumulative_reserve: float = Field(0.0, description="Cumulative reserve amount up to this period")
    cumulative_incurred: float = Field(0.0, description="Cumulative incurred amount up to this period")
    
    # Normalized cumulative amounts up to this period (z-score normalized)
    cumulative_paid_normalized: float = Field(0.0, description="Normalized cumulative paid amount up to this period")
    cumulative_expense_normalized: float = Field(0.0, description="Normalized cumulative expense amount up to this period")
    
    # Period-specific features
    num_transactions: int = Field(0, description="Number of transactions in this period")
    has_payment: bool = Field(False, description="Whether there was any payment in this period")
    has_expense: bool = Field(False, description="Whether there was any expense in this period")
    
    # Time-based features
    days_since_first_txn: int = Field(0, description="Days since first transaction")
    development_stage: float = Field(0.0, description="Development stage (0-1, where 1 = 5 years)")


class StandardizedClaim(BaseModel):
    """
    Complete standardized claim with static context and dynamic periods
    """
    static_context: StaticClaimContext = Field(..., description="Static claim context")
    dynamic_periods: List[DynamicClaimPeriod] = Field(..., description="List of dynamic periods")
    
    # Summary metrics
    total_periods: int = Field(..., description="Total number of periods")
    max_period: int = Field(..., description="Maximum period number")
    total_paid: float = Field(0.0, description="Total paid amount across all periods")
    total_expense: float = Field(0.0, description="Total expense amount across all periods")
    total_recovery: float = Field(0.0, description="Total recovery amount across all periods")
    final_incurred: float = Field(0.0, description="Final incurred amount")
    final_reserve: float = Field(0.0, description="Final reserve amount")


class StandardizedClaimsDataset(BaseModel):
    """
    Complete dataset of standardized claims
    """
    claims: List[StandardizedClaim] = Field(..., description="List of standardized claims")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")
    
    # Dataset summary
    total_claims: int = Field(..., description="Total number of claims")
    total_periods: int = Field(..., description="Total number of periods across all claims")
    date_created: datetime = Field(default_factory=datetime.now, description="Date when dataset was created")
    
    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Convert the standardized claims to pandas DataFrames for analysis
        """
        # Static context DataFrame
        static_data = []
        for claim in self.claims:
            static_data.append(claim.static_context.dict())
        static_df = pd.DataFrame(static_data)
        
        # Dynamic periods DataFrame
        dynamic_data = []
        for claim in self.claims:
            for period in claim.dynamic_periods:
                period_data = period.dict()
                period_data['clmNum'] = claim.static_context.clmNum
                dynamic_data.append(period_data)
        dynamic_df = pd.DataFrame(dynamic_data)
        
        return {
            'static_context': static_df,
            'dynamic_periods': dynamic_df
        }


class StandardizationConfig(BaseModel):
    """
    Configuration for the standardization process
    """
    period_length_days: int = Field(30, description="Length of each period in days")
    max_periods: int = Field(60, description="Maximum number of periods to track (60 = 5 years)")
    min_periods: int = Field(1, description="Minimum number of periods required")
    
    # Column mappings for input data
    claim_number_col: str = Field("clmNum", description="Column name for claim number")
    transaction_date_col: str = Field("datetxn", description="Column name for transaction date")
    paid_amount_col: str = Field("paid", description="Column name for paid amount")
    expense_amount_col: str = Field("expense", description="Column name for expense amount")
    recovery_amount_col: str = Field("recovery", description="Column name for recovery amount")
    reserve_amount_col: str = Field("reserve", description="Column name for reserve amount")
    
    # Static context columns
    claim_cause_col: str = Field("clmCause", description="Column name for claim cause")
    booking_number_col: str = Field("booknum", description="Column name for booking number")
    policy_id_col: str = Field("cidpol", description="Column name for policy ID")
    claim_status_col: str = Field("clmStatus", description="Column name for claim status")
    date_completed_col: str = Field("dateCompleted", description="Column name for completion date")
    date_reopened_col: str = Field("dateReopened", description="Column name for reopen date")
    
    # Derived features
    include_derived_features: bool = Field(True, description="Whether to include derived features")
    development_stage_max_days: int = Field(1825, description="Maximum days for development stage calculation (5 years)")


def validate_standardized_claim(claim: StandardizedClaim) -> bool:
    """
    Validate a standardized claim for consistency
    """
    # Check that periods are sequential
    periods = [p.period for p in claim.dynamic_periods]
    if periods != sorted(periods):
        return False
    
    # Check that days_from_receipt are correct
    for period in claim.dynamic_periods:
        expected_days = period.period * 30
        if period.days_from_receipt != expected_days:
            return False
    
    # Check cumulative amounts are non-decreasing
    for i in range(1, len(claim.dynamic_periods)):
        prev_period = claim.dynamic_periods[i-1]
        curr_period = claim.dynamic_periods[i]
        
        if (curr_period.cumulative_paid < prev_period.cumulative_paid or
            curr_period.cumulative_expense < prev_period.cumulative_expense or
            curr_period.cumulative_recovery < prev_period.cumulative_recovery):
            return False
    
    return True
