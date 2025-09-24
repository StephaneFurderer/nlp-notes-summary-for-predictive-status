"""
Data schema definitions for claims data processing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Define the expected data types for claims data columns
CLAIMS_DATA_SCHEMA = {
    # Date/Time columns - should be datetime or None
    'datetxn': 'datetime64[ns]',
    'dateeff': 'datetime64[ns]', 
    'dateterm': 'datetime64[ns]',
    'dateReceived': 'datetime64[ns]',
    'dateOpened': 'datetime64[ns]',
    'dateCompleted': 'datetime64[ns]',
    'dateReopened': 'datetime64[ns]',
    'dateIncurred': 'datetime64[ns]',
    'dateAssigned': 'datetime64[ns]',
    'dateRepInitialContact': 'datetime64[ns]',
    'whenOpenStatus': 'datetime64[ns]',
    
    # String/Categorical columns
    'cidpol': 'string',
    'licstate': 'string',
    'clmType': 'string',
    'booknum': 'string',
    'clmNum': 'string',
    'clmReason': 'string',
    'prod': 'string',
    'clmStatus': 'string',
    'processStatus': 'string',
    'clmCause': 'string',
    'whereIncur': 'string',
    'Country': 'string',
    'state': 'string',
    'City': 'string',
    'payCategory': 'string',
    'paymentType': 'string',
    'AcceptedLOB': 'string',
    'clmStatusReason': 'string',
    'processCategory': 'string',
    'whoRepAssigned': 'string',
    'whoAssigned': 'string',
    'payee': 'string',
    'whoResponsible': 'string',
    'whoReviewer': 'string',
    'reasonClmPay': 'string',
    
    # Numeric columns
    'PostalCode': 'string',  # Keep as string to handle leading zeros
    'amt': 'float64',
    'subroOpportunity': 'int64',
}

# Define date parsing format for the specific format in the data
DATE_PARSE_FORMAT = '%H:%M.%S'

def clean_and_convert_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and convert DataFrame according to the claims data schema - raw data
    """
    df_clean = df.copy()
    
    # Handle date columns - convert '00:00.0' to None, proper dates to datetime
    date_columns = [col for col in CLAIMS_DATA_SCHEMA.keys() if 'date' in col.lower() or col == 'whenOpenStatus']
    
    for col in date_columns:
        if col in df_clean.columns:
            # Replace '00:00.0' with None
            df_clean[col] = df_clean[col].replace(['00:00.0', 'NULL'], None)
            
            # Convert remaining valid dates
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Handle string columns - replace 'NULL' with None
    string_columns = [col for col in CLAIMS_DATA_SCHEMA.keys() if CLAIMS_DATA_SCHEMA[col] == 'string']
    for col in string_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(['NULL'], None)
    
    # Handle numeric columns
    if 'amt' in df_clean.columns:
        df_clean['amt'] = pd.to_numeric(df_clean['amt'], errors='coerce')
    
    if 'subroOpportunity' in df_clean.columns:
        df_clean['subroOpportunity'] = pd.to_numeric(df_clean['subroOpportunity'], errors='coerce').astype('Int64')
    
    # Handle PostalCode - keep as string to preserve leading zeros
    if 'PostalCode' in df_clean.columns:
        df_clean['PostalCode'] = df_clean['PostalCode'].astype(str)
        df_clean['PostalCode'] = df_clean['PostalCode'].replace(['nan', 'None'], None)
    

    # Process the data
    df_clean['booknum'] = np.where(df_clean['booknum'].isnull(), "NO_BOOKING_NUM", df_clean['booknum'])
    df_clean['dateCompleted'] = pd.to_datetime(df_clean['dateCompleted'], errors='coerce')
    df_clean['dateReopened'] = pd.to_datetime(df_clean['dateReopened'], errors='coerce')
    df_clean['datetxn'] = pd.to_datetime(df_clean['datetxn'], errors='coerce')

    return df_clean

def get_claims_data_types() -> Dict[str, str]:
    """Get the data types dictionary for claims data"""
    return CLAIMS_DATA_SCHEMA.copy()

def validate_claims_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate claims data and return validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_columns': [],
        'extra_columns': [],
        'data_type_issues': [],
        'null_counts': {}
    }
    
    # Check for missing expected columns
    expected_columns = set(CLAIMS_DATA_SCHEMA.keys())
    actual_columns = set(df.columns)
    
    validation_results['missing_columns'] = list(expected_columns - actual_columns)
    validation_results['extra_columns'] = list(actual_columns - expected_columns)
    
    # Check null counts
    for col in df.columns:
        null_count = df[col].isnull().sum()
        validation_results['null_counts'][col] = null_count
    
    return validation_results
