#!/usr/bin/env python3
"""
Simple Load and Cache Data Module for Claims Processing

This module handles data loading and caching for the claims processing pipeline
using structured folders by extraction date.
"""

import os
import glob
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from .claims_data_schema import clean_and_convert_dataframe, get_claims_data_types
from .CONST import BASE_DATA_DIR

# instead of print, use logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def get_available_data_versions() -> List[Dict[str, Any]]:
    """
    Get list of available data versions (extraction dates)
    
    Returns:
        List of dictionaries with extraction date info
    """
    date_folders = glob.glob(os.path.join(BASE_DATA_DIR, "*") + os.path.sep)
    extraction_dates = []
    
    for folder in date_folders:
        date_name = os.path.basename(folder.rstrip(os.path.sep))
        # Validate date format (YYYY-MM-DD)
        try:
            datetime.strptime(date_name, '%Y-%m-%d')
            # Check if claims file exists
            claims_file = os.path.join(folder, "clm_with_amt.csv")
            if os.path.exists(claims_file):
                extraction_dates.append({
                    'extraction_date': date_name,
                    'claims_file': claims_file,
                    'folder_path': folder
                })
        except ValueError:
            # Skip folders that don't match date format (like 'cache')
            continue
    
    # Sort by date (newest first)
    extraction_dates.sort(key=lambda x: x['extraction_date'], reverse=True)
    return extraction_dates




class CacheManager:
    """
    Simple cache manager for periodized data using structured folders
    """

    def __init__(self, base_data_dir: str = None):
        if base_data_dir is None:
            # Find the project root directory (where this file is located)
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            base_data_dir = os.path.join(project_root, "_data")

        self.base_data_dir = base_data_dir
    
    def get_cache_path(self, extraction_date: str) -> str:
        """Get cache file path for extraction date"""
        date_dir = os.path.join(self.base_data_dir, extraction_date)
        os.makedirs(date_dir, exist_ok=True)
        return os.path.join(date_dir, "period_clm.parquet")
    
    def load_cache(self, df_txn: pd.DataFrame, extraction_date: str) -> Optional[pd.DataFrame]:
        """Load cached periodized data if it exists"""
        cache_file = self.get_cache_path(extraction_date)
        
        if os.path.exists(cache_file):
            cached_df = pd.read_parquet(cache_file)
            print(f"📁 Loaded cached period data: {len(cached_df):,} periods from {cached_df['clmNum'].nunique():,} claims")
            return cached_df
        else:
            return None
    
    def save_cache(self, periods_df: pd.DataFrame, df_txn: pd.DataFrame, extraction_date: str) -> bool:
        """Save periodized data to cache"""
        cache_file = self.get_cache_path(extraction_date)
        
        periods_df.to_parquet(cache_file, index=False)
        print(f"💾 Cached period data: {len(periods_df):,} periods from {periods_df['clmNum'].nunique():,} claims")
        return True


class DataLoader:
    """
    Handles loading of raw data from structured folders
    """

    def __init__(self, base_data_dir: str = None):
        if base_data_dir is None:
            # Find the project root directory (where this file is located)
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            base_data_dir = os.path.join(project_root, "_data")

        self.base_data_dir = base_data_dir
        self.cache_manager = CacheManager(base_data_dir)
    
    
    
    
    
    def load_notes_data(self, extraction_date: Optional[str] = None,
                       notes_file: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load notes data from structured folders with parquet caching
        
        Args:
            extraction_date: Date string for structured approach
            notes_file: Direct file path (if not using structured approach)
            
        Returns:
            Notes DataFrame or None if not found
        """
        if extraction_date:
            # Load directly from date directory
            date_dir = os.path.join(self.base_data_dir, extraction_date)
            notes_file = os.path.join(date_dir, "notes_summary.csv")
            if not os.path.exists(notes_file):
                print(f"ℹ️ No notes file found for extraction date {extraction_date}")
                return None
        
        if notes_file and os.path.exists(notes_file):
            # Check for parquet cache
            parquet_file = notes_file.replace('.csv', '.parquet')
            
            if os.path.exists(parquet_file):
                df = pd.read_parquet(parquet_file)
                print(f"📁 Loaded cached notes data: {len(df):,} notes")
                return df
            
            # Load from CSV and save as parquet
            df = pd.read_csv(notes_file)
            
            # Clean and convert data according to schema
            df = clean_and_convert_dataframe(df)
            
            df.to_parquet(parquet_file)
            print(f"📁 Loaded and cached notes data: {len(df):,} notes")
            return df
        else:
            print(f"ℹ️ Notes file not found: {notes_file}")
            return None