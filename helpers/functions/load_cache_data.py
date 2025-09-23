#!/usr/bin/env python3
"""
Load and Cache Data Module for Claims Processing

This module handles all data loading, caching, and organization logic for the
claims processing pipeline. It supports both structured folder organization
by extraction date and legacy hash-based caching.
"""

import os
import glob
import pickle
import shutil
import hashlib
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from .standardized_claims_schema import StandardizationConfig


class DataOrganizer:
    """
    Handles organization of input data into structured folders by extraction date
    """
    
    def __init__(self, base_data_dir: str = "_data"):
        self.base_data_dir = base_data_dir
    
    def create_folder_structure(self, extraction_date: str) -> str:
        """
        Create folder structure for a given extraction date
        
        Args:
            extraction_date: Date string in YYYY-MM-DD format
            
        Returns:
            Path to the date directory
        """
        date_dir = f"{self.base_data_dir}/{extraction_date}"
        os.makedirs(date_dir, exist_ok=True)
        return date_dir
    
    def organize_files(self, extraction_date: str, claims_file: str, 
                      notes_file: Optional[str] = None, 
                      policy_file: Optional[str] = None) -> Dict[str, str]:
        """
        Organize input files into structured folders by extraction date
        
        Args:
            extraction_date: Date string in YYYY-MM-DD format
            claims_file: Path to claims CSV file
            notes_file: Optional path to notes CSV file  
            policy_file: Optional path to policy CSV file
            
        Returns:
            Dictionary mapping file types to their new organized paths
        """
        date_dir = self.create_folder_structure(extraction_date)
        organized_files = {}
        
        # Copy claims file
        if os.path.exists(claims_file):
            clm_dest = f"{date_dir}/clm_with_amt.csv"
            shutil.copy2(claims_file, clm_dest)
            organized_files['claims'] = clm_dest
        
        # Copy notes file if provided
        if notes_file and os.path.exists(notes_file):
            notes_dest = f"{date_dir}/notes_summary.csv"
            shutil.copy2(notes_file, notes_dest)
            organized_files['notes'] = notes_dest
        
        # Copy policy file if provided
        if policy_file and os.path.exists(policy_file):
            policy_dest = f"{date_dir}/policy_info.csv"
            shutil.copy2(policy_file, policy_dest)
            organized_files['policy'] = policy_dest
        
        return organized_files
    
    def get_organized_files(self, extraction_date: str) -> List[str]:
        """
        Get list of organized input files for a given extraction date
        
        Args:
            extraction_date: Date string in YYYY-MM-DD format
            
        Returns:
            List of file paths that exist for this extraction date
        """
        date_dir = f"{self.base_data_dir}/{extraction_date}"
        input_files = []
        
        # Check for standard file locations directly in date folder
        potential_files = [
            f"{date_dir}/clm_with_amt.csv",
            f"{date_dir}/notes_summary.csv", 
            f"{date_dir}/policy_info.csv"
        ]
        
        for file_path in potential_files:
            if os.path.exists(file_path):
                input_files.append(file_path)
        
        return input_files
    
    def list_extraction_dates(self) -> List[str]:
        """
        List all available extraction dates (folder names)
        
        Returns:
            List of extraction date strings sorted by date (newest first)
        """
        date_folders = glob.glob(f"{self.base_data_dir}/*/")
        extraction_dates = []
        
        for folder in date_folders:
            date_name = os.path.basename(folder.rstrip('/'))
            # Validate date format (YYYY-MM-DD)
            try:
                datetime.strptime(date_name, '%Y-%m-%d')
                extraction_dates.append(date_name)
            except ValueError:
                # Skip folders that don't match date format
                continue
        
        # Sort by date (newest first)
        extraction_dates.sort(reverse=True)
        return extraction_dates


class CacheManager:
    """
    Handles all caching operations for processed claims data
    """
    
    def __init__(self, base_data_dir: str = "_data"):
        self.base_data_dir = base_data_dir
        self.data_organizer = DataOrganizer(base_data_dir)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of a file's content"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return "error"
    
    def _create_data_hash(self, df_txn: pd.DataFrame, input_files: Optional[List[str]] = None) -> str:
        """
        Create a hash of the input data for cache invalidation
        Now supports multiple input files (CSV files, etc.)
        """
        # Create a summary of the data for hashing
        data_summary = {
            'num_rows': len(df_txn),
            'num_claims': df_txn['clmNum'].nunique(),  # Assuming clmNum is the claim number column
            'date_range': {
                'min': str(df_txn['datetxn'].min()),  # Assuming datetxn is the transaction date column
                'max': str(df_txn['datetxn'].max())
            },
            'columns': sorted(df_txn.columns.tolist()),
            'data_hash': hashlib.md5(
                df_txn.sort_values(['clmNum', 'datetxn'])
                .to_string().encode()
            ).hexdigest()[:16]  # Use first 16 chars for shorter filenames
        }
        
        # Add input file information if provided
        if input_files:
            file_info = {}
            for file_path in input_files:
                if os.path.exists(file_path):
                    file_stat = os.stat(file_path)
                    file_info[os.path.basename(file_path)] = {
                        'size': file_stat.st_size,
                        'modified': file_stat.st_mtime,
                        'content_hash': self._get_file_hash(file_path)[:16]
                    }
            data_summary['input_files'] = file_info
        
        # Create hash from summary
        summary_str = str(sorted(data_summary.items()))
        return hashlib.md5(summary_str.encode()).hexdigest()[:12]
    
    def _create_cache_metadata(self, df_txn: pd.DataFrame, result_df: pd.DataFrame, 
                              input_files: Optional[List[str]] = None, 
                              extraction_date: Optional[str] = None,
                              config: Optional[StandardizationConfig] = None) -> dict:
        """
        Create metadata for cache validation
        """
        # Create human-readable description
        data_hash = self._create_data_hash(df_txn, input_files)
        date_range = {
            'min': df_txn['datetxn'].min().strftime('%Y-%m-%d'),
            'max': df_txn['datetxn'].max().strftime('%Y-%m-%d')
        }
        
        if extraction_date:
            description = f"Extracted {extraction_date} - {date_range['min']} to {date_range['max']} - {len(df_txn):,} transactions"
        else:
            description = f"{date_range['min']} to {date_range['max']} - {len(df_txn):,} transactions"
        
        if input_files:
            file_names = [os.path.basename(f) for f in input_files if os.path.exists(f)]
            if file_names:
                description += f" from {', '.join(file_names)}"
        
        metadata = {
            'data_version': data_hash,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'extraction_date': extraction_date,
            'num_transactions': len(df_txn),
            'num_claims': df_txn['clmNum'].nunique(),
            'num_periods': len(result_df),
            'date_range': date_range,
            'input_files': input_files or [],
            'normalization_computed': False  # Will be updated by transformer
        }
        
        if config:
            metadata['config'] = {
                'period_length_days': config.period_length_days,
                'max_periods': config.max_periods,
                'min_periods': config.min_periods
            }
        
        return metadata
    
    def _validate_cache_metadata(self, cache_meta: dict, df_txn: pd.DataFrame, 
                                input_files: Optional[List[str]] = None,
                                config: Optional[StandardizationConfig] = None) -> bool:
        """
        Validate cache metadata against current data and config
        """
        # Check if data hash matches (including input files)
        current_hash = self._create_data_hash(df_txn, input_files)
        if cache_meta['data_version'] != current_hash:
            return False
        
        # Check if config matches
        if config and 'config' in cache_meta:
            if cache_meta['config']['period_length_days'] != config.period_length_days:
                return False
            if cache_meta['config']['max_periods'] != config.max_periods:
                return False
            if cache_meta['config']['min_periods'] != config.min_periods:
                return False
        
        return True
    
    def get_cache_paths(self, extraction_date: str) -> Tuple[str, str]:
        """
        Get cache file paths based on extraction date
        
        Args:
            extraction_date: Date string for structured approach
            
        Returns:
            Tuple of (cache_file_path, metadata_file_path)
        """
        # Use structured folder approach - cache files go directly in date folder
        date_dir = f"{self.base_data_dir}/{extraction_date}"
        os.makedirs(date_dir, exist_ok=True)
        cache_file = f"{date_dir}/period_clm.parquet"
        meta_file = f"{date_dir}/period_clm_meta.pkl"
        
        return cache_file, meta_file
    
    def save_cache(self, periods_df: pd.DataFrame, df_txn: pd.DataFrame,
                  extraction_date: str,
                  input_files: Optional[List[str]] = None,
                  config: Optional[StandardizationConfig] = None) -> bool:
        """
        Save processed data to cache
        
        Args:
            periods_df: Processed periods DataFrame
            df_txn: Original transaction DataFrame
            extraction_date: Date string for structured approach
            input_files: List of input files used
            config: Standardization configuration
            
        Returns:
            True if cache was saved successfully, False otherwise
        """
        try:
            # Get cache paths
            cache_file, meta_file = self.get_cache_paths(extraction_date)
            
            # Create metadata
            cache_meta = self._create_cache_metadata(df_txn, periods_df, input_files, extraction_date, config)
            
            # Save data and metadata
            periods_df.to_parquet(cache_file, index=False)
            with open(meta_file, 'wb') as f:
                pickle.dump(cache_meta, f)
            
            print(f"💾 Cache saved: {cache_file}")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")
            return False
    
    def load_cache(self, df_txn: pd.DataFrame,
                  extraction_date: str,
                  input_files: Optional[List[str]] = None,
                  config: Optional[StandardizationConfig] = None) -> Optional[pd.DataFrame]:
        """
        Load processed data from cache if valid
        
        Args:
            df_txn: Current transaction DataFrame for validation
            extraction_date: Date string for structured approach
            input_files: List of input files used
            config: Standardization configuration
            
        Returns:
            Cached DataFrame if valid, None otherwise
        """
        try:
            # Get cache paths
            cache_file, meta_file = self.get_cache_paths(extraction_date)
            
            # Check if cache files exist
            if not os.path.exists(cache_file) or not os.path.exists(meta_file):
                return None
            
            # Load metadata
            with open(meta_file, 'rb') as f:
                cache_meta = pickle.load(f)
            
            # Validate cache metadata
            if not self._validate_cache_metadata(cache_meta, df_txn, input_files, config):
                print("⚠️ Cache metadata validation failed, recomputing...")
                return None
            
            # Load cached data
            cached_df = pd.read_parquet(cache_file)
            
            # Basic validation: check if cache has expected columns
            expected_cols = ['clmNum', 'period', 'incremental_paid', 'incremental_expense', 
                           'cumulative_paid', 'cumulative_expense']
            if all(col in cached_df.columns for col in expected_cols):
                print(f"📁 Loaded cached data: {len(cached_df):,} periods from {cached_df['clmNum'].nunique():,} claims")
                print(f"📊 Cache info: {cache_meta['description']} | {cache_meta['created_at']}")
                return cached_df
            else:
                print("⚠️ Cached data missing expected columns, recomputing...")
                return None
                
        except Exception as e:
            print(f"⚠️ Error loading cached data: {e}")
            return None
    
    def list_available_caches(self) -> List[Dict[str, Any]]:
        """
        List all available cache files with their metadata for cache selection UI
        Only supports structured folder approach by extraction date
        """
        cache_info = []
        
        # Check structured folders (extraction_date/)
        date_folders = glob.glob(f"{self.base_data_dir}/*/")
        for date_dir in date_folders:
            cache_file = os.path.join(date_dir, "period_clm.parquet")
            meta_file = os.path.join(date_dir, "period_clm_meta.pkl")
            
            if os.path.exists(cache_file) and os.path.exists(meta_file):
                try:
                    with open(meta_file, 'rb') as f:
                        meta = pickle.load(f)
                    
                    # Extract date from folder path
                    extraction_date = os.path.basename(date_dir.rstrip('/'))
                    
                    # Calculate file size
                    cache_size = os.path.getsize(cache_file) / (1024 * 1024)  # MB
                    
                    # Parse creation date
                    try:
                        created_date = datetime.fromisoformat(meta['created_at']).strftime('%Y-%m-%d %H:%M')
                    except:
                        created_date = "Unknown"
                    
                    cache_info.append({
                        'extraction_date': extraction_date,
                        'file_path': cache_file,
                        'meta_file': meta_file,
                        'description': meta.get('description', 'Unknown'),
                        'created_at': created_date,
                        'num_transactions': meta.get('num_transactions', 0),
                        'num_claims': meta.get('num_claims', 0),
                        'cache_size_mb': round(cache_size, 1),
                        'input_files': meta.get('input_files', [])
                    })
                except Exception as e:
                    # Skip corrupted metadata files
                    continue
        
        # Sort by extraction date (newest first)
        cache_info.sort(key=lambda x: x['extraction_date'], reverse=True)
        return cache_info
    
    def cleanup_old_caches(self, keep_extraction_dates: Optional[List[str]] = None) -> None:
        """
        Clean up old cache files, optionally keeping specific extraction dates
        
        Args:
            keep_extraction_dates: List of extraction dates to keep
        """
        if keep_extraction_dates is None:
            keep_extraction_dates = []
        
        # Clean up date folders not in keep list
        date_folders = glob.glob(f"{self.base_data_dir}/*/")
        for date_dir in date_folders:
            extraction_date = os.path.basename(date_dir.rstrip('/'))
            if extraction_date not in keep_extraction_dates:
                try:
                    shutil.rmtree(date_dir)
                    print(f"🗑️ Removed old cache: {extraction_date}")
                except Exception as e:
                    print(f"⚠️ Could not remove cache {extraction_date}: {e}")


class DataLoader:
    """
    Handles loading of raw data from various sources
    """
    
    def __init__(self, base_data_dir: str = "_data"):
        self.base_data_dir = base_data_dir
        self.data_organizer = DataOrganizer(base_data_dir)
        self.cache_manager = CacheManager(base_data_dir)
    
    def load_claims_data(self, extraction_date: Optional[str] = None, 
                        claims_file: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load claims data from organized structure or direct file path
        
        Args:
            extraction_date: Date string for organized structure
            claims_file: Direct file path (if not using organized structure)
            
        Returns:
            Claims DataFrame or None if not found
        """
        if extraction_date:
            # Load from organized structure
            organized_files = self.data_organizer.get_organized_files(extraction_date)
            claims_files = [f for f in organized_files if 'clm_with_amt.csv' in f]
            if claims_files:
                claims_file = claims_files[0]
            else:
                print(f"⚠️ No claims file found for extraction date {extraction_date}")
                return None
        
        if claims_file and os.path.exists(claims_file):
            try:
                df = pd.read_csv(claims_file)
                print(f"📁 Loaded claims data: {len(df):,} transactions from {df['clmNum'].nunique():,} claims")
                return df
            except Exception as e:
                print(f"⚠️ Error loading claims data from {claims_file}: {e}")
                return None
        else:
            print(f"⚠️ Claims file not found: {claims_file}")
            return None
    
    def load_notes_data(self, extraction_date: Optional[str] = None,
                       notes_file: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load notes data from organized structure or direct file path
        
        Args:
            extraction_date: Date string for organized structure
            notes_file: Direct file path (if not using organized structure)
            
        Returns:
            Notes DataFrame or None if not found
        """
        if extraction_date:
            # Load from organized structure
            organized_files = self.data_organizer.get_organized_files(extraction_date)
            notes_files = [f for f in organized_files if 'notes_summary.csv' in f]
            if notes_files:
                notes_file = notes_files[0]
            else:
                print(f"ℹ️ No notes file found for extraction date {extraction_date}")
                return None
        
        if notes_file and os.path.exists(notes_file):
            try:
                df = pd.read_csv(notes_file)
                print(f"📁 Loaded notes data: {len(df):,} notes")
                return df
            except Exception as e:
                print(f"⚠️ Error loading notes data from {notes_file}: {e}")
                return None
        else:
            print(f"ℹ️ Notes file not found: {notes_file}")
            return None
    
    def get_available_data_versions(self) -> List[Dict[str, Any]]:
        """
        Get list of all available data versions (extraction dates)
        
        Returns:
            List of dictionaries with extraction date info
        """
        extraction_dates = self.data_organizer.list_extraction_dates()
        versions = []
        
        for date in extraction_dates:
            files = self.data_organizer.get_organized_files(date)
            file_types = []
            if any('clm_with_amt.csv' in f for f in files):
                file_types.append('claims')
            if any('notes_summary.csv' in f for f in files):
                file_types.append('notes')
            if any('policy_info.csv' in f for f in files):
                file_types.append('policy')
            
            versions.append({
                'extraction_date': date,
                'file_types': file_types,
                'num_files': len(files),
                'files': files
            })
        
        return versions
