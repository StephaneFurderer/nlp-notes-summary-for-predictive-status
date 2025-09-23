"""
Standardized Claims Transformer for Micro-Level Reserving

Transforms raw transaction data into standardized 30-day periods following the framework
described in "Micro-level reserving for general insurance claims using a long short-term memory network"
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import warnings
from .standardized_claims_schema import (
    StandardizedClaim, StaticClaimContext, DynamicClaimPeriod, 
    StandardizedClaimsDataset, StandardizationConfig, validate_standardized_claim
)
from .load_cache_data import CacheManager, DataOrganizer

warnings.filterwarnings('ignore')


class StandardizedClaimsTransformer:
    """
    Transforms raw transaction data into standardized claims format
    """
    
    def __init__(self, config: Optional[StandardizationConfig] = None, debug_logging: bool = False):
        self.config = config or StandardizationConfig()
        self.debug_logging = debug_logging
        
        # Storage for processed claims
        self.standardized_claims: Dict[str, StandardizedClaim] = {}
        
        # Normalization parameters
        self.normalization_params = {
            'paid': {'mean': None, 'std': None},
            'expense': {'mean': None, 'std': None}
        }
        self.period_normalization_params: Optional[Dict[str, Any]] = None
        self.normalization_computed = False
        self.period_normalization_computed = False
        
        # Cache and data management
        self.cache_manager = CacheManager()
        self.data_organizer = DataOrganizer()
    
    def list_available_caches(self) -> List[Dict[str, Any]]:
        """Delegate to cache manager"""
        return self.cache_manager.list_available_caches()
    
    def get_organized_input_files(self, extraction_date: str) -> List[str]:
        """Delegate to data organizer"""
        return self.data_organizer.get_organized_files(extraction_date)
    
    def organize_data_by_extraction_date(self, extraction_date: str, claims_file: str, 
                                       notes_file: Optional[str] = None, 
                                       policy_file: Optional[str] = None) -> Dict[str, str]:
        """Delegate to data organizer"""
        return self.data_organizer.organize_files(extraction_date, claims_file, notes_file, policy_file)
    
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
        # Use the correct column names from the standardized schema
        payment_types = ['incremental_paid', 'incremental_expense']

        # Check which payment columns actually exist in the dataframe
        available_columns = df_completed_txn.columns.tolist()
        print(f"Available columns in df_completed_txn: {available_columns}")

        for payment_type in payment_types:
            if payment_type not in df_completed_txn.columns:
                print(f"Warning: Column '{payment_type}' not found in dataframe. Skipping normalization for this column.")
                continue

            # Get only positive incremental payments (exclude zeros)
            positive_payments = df_completed_txn[df_completed_txn[payment_type] > 0][payment_type]
            
            if len(positive_payments) > 0:
                mean_val = positive_payments.mean()
                std_val = positive_payments.std()
                
                # Store parameters using the base name (without 'incremental_')
                base_name = payment_type.replace('incremental_', '')
                self.normalization_params[base_name]['mean'] = mean_val
                self.normalization_params[base_name]['std'] = std_val
                
                print(f"Normalization parameters for {payment_type}:")
                print(f"  Mean: {mean_val:.2f}")
                print(f"  Std:  {std_val:.2f}")
                print(f"  Sample size: {len(positive_payments)}")
            else:
                print(f"Warning: No positive {payment_type} payments found")
                base_name = payment_type.replace('incremental_', '')
                self.normalization_params[base_name]['mean'] = 0.0
                self.normalization_params[base_name]['std'] = 1.0
        
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
        for payment_type in ['incremental_paid', 'incremental_expense']:
            if payment_type not in df_normalized.columns:
                print(f"Warning: Column '{payment_type}' not found. Skipping normalization.")
                continue

            base_name = payment_type.replace('incremental_', '')
            params = self.normalization_params[base_name]

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
        payment_types = ['incremental_paid', 'incremental_expense']
        colors = ['blue', 'red']
        
        for i, payment_type in enumerate(payment_types):
            if payment_type not in df_completed_txn.columns:
                print(f"Warning: Column '{payment_type}' not found in dataframe. Skipping visualization.")
                continue

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
            if payment_type not in df_completed_txn.columns:
                print(f"Warning: Column '{payment_type}' not found. Skipping summary statistics.")
                continue

            base_name = payment_type.replace('incremental_', '')
            params = self.normalization_params[base_name]
            original_positive = df_completed_txn[df_completed_txn[payment_type] > 0][payment_type]
            normalized_positive = df_normalized[df_normalized[payment_type] > 0][f'{payment_type}_normalized']
            
            print(f"\n{payment_type.upper()}:")
            print(f"  Original - Mean: {original_positive.mean():.2f}, Std: {original_positive.std():.2f}")
            print(f"  Normalized - Mean: {normalized_positive.mean():.4f}, Std: {normalized_positive.std():.4f}")
            print(f"  Sample size: {len(original_positive)}")
            print(f"  Normalization params - Mean: {params['mean']:.2f}, Std: {params['std']:.2f}")
        
        print("="*60)
        
        return fig
    
    def _create_data_hash(self, df_txn: pd.DataFrame, input_files: Optional[List[str]] = None) -> str:
        """
        Create a hash of the input data for cache invalidation
        Now supports multiple input files (CSV files, etc.)
        """
        import hashlib
        import os
        
        # Create a summary of the data for hashing
        data_summary = {
            'num_rows': len(df_txn),
            'num_claims': df_txn[self.config.claim_number_col].nunique(),
            'date_range': {
                'min': str(df_txn[self.config.transaction_date_col].min()),
                'max': str(df_txn[self.config.transaction_date_col].max())
            },
            'columns': sorted(df_txn.columns.tolist()),
            'data_hash': hashlib.md5(
                df_txn.sort_values([self.config.claim_number_col, self.config.transaction_date_col])
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
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of a file's content"""
        import hashlib
        
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return "error"
    
    def _create_cache_metadata(self, df_txn: pd.DataFrame, result_df: pd.DataFrame, input_files: Optional[List[str]] = None, extraction_date: Optional[str] = None) -> dict:
        """
        Create metadata for cache validation
        """
        from datetime import datetime
        
        # Create human-readable description
        data_hash = self._create_data_hash(df_txn, input_files)
        date_range = {
            'min': df_txn[self.config.transaction_date_col].min().strftime('%Y-%m-%d'),
            'max': df_txn[self.config.transaction_date_col].max().strftime('%Y-%m-%d')
        }
        
        if extraction_date:
            description = f"Extracted {extraction_date} - {date_range['min']} to {date_range['max']} - {len(df_txn):,} transactions"
        else:
            description = f"{date_range['min']} to {date_range['max']} - {len(df_txn):,} transactions"
        
        if input_files:
            file_names = [os.path.basename(f) for f in input_files if os.path.exists(f)]
            if file_names:
                description += f" from {', '.join(file_names)}"
        
        return {
            'data_version': data_hash,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'extraction_date': extraction_date,
            'num_transactions': len(df_txn),
            'num_claims': df_txn[self.config.claim_number_col].nunique(),
            'num_periods': len(result_df),
            'date_range': date_range,
            'input_files': input_files or [],
            'config': {
                'period_length_days': self.config.period_length_days,
                'max_periods': self.config.max_periods,
                'min_periods': self.config.min_periods
            },
            'normalization_computed': self.normalization_computed
        }
    
    def _validate_cache_metadata(self, cache_meta: dict, df_txn: pd.DataFrame, input_files: Optional[List[str]] = None) -> bool:
        """
        Validate cache metadata against current data and config
        """
        # Check if data hash matches (including input files)
        current_hash = self._create_data_hash(df_txn, input_files)
        if cache_meta['data_version'] != current_hash:
            return False
        
        # Check if config matches
        if cache_meta['config']['period_length_days'] != self.config.period_length_days:
            return False
        if cache_meta['config']['max_periods'] != self.config.max_periods:
            return False
        if cache_meta['config']['min_periods'] != self.config.min_periods:
            return False
        
        # Check normalization status
        if cache_meta['normalization_computed'] != self.normalization_computed:
            return False
        
        return True
    
    def list_available_caches(self) -> List[Dict[str, Any]]:
        """
        List all available cache files with their metadata for cache selection UI
        Supports both structured folder approach and legacy hash-based approach
        """
        import os
        import glob
        import pickle
        from datetime import datetime
        
        cache_info = []
        
        # Method 1: Check structured folders (extraction_date/cache/)
        date_folders = glob.glob(os.path.join("_data", "*", "cache") + os.path.sep)
        for cache_dir in date_folders:
            cache_file = os.path.join(cache_dir, "period_clm.parquet")
            meta_file = os.path.join(cache_dir, "period_clm_meta.pkl")
            
            if os.path.exists(cache_file) and os.path.exists(meta_file):
                try:
                    with open(meta_file, 'rb') as f:
                        meta = pickle.load(f)
                    
                    # Extract date from folder path
                    extraction_date = os.path.basename(os.path.dirname(cache_dir))
                    
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
                        'input_files': meta.get('input_files', []),
                        'type': 'structured'
                    })
                except Exception:
                    # Skip corrupted metadata files
                    continue
        
        # Method 2: Check legacy hash-based files
        hash_files = glob.glob(os.path.join("_data", "period_clm_*.parquet"))
        for cache_file in hash_files:
            # Extract hash from filename
            hash_part = os.path.basename(cache_file).replace("period_clm_", "").replace(".parquet", "")
            meta_file = os.path.join("_data", f"period_clm_{hash_part}_meta.pkl")
            
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, 'rb') as f:
                        meta = pickle.load(f)
                    
                    # Calculate file size
                    cache_size = os.path.getsize(cache_file) / (1024 * 1024)  # MB
                    
                    # Parse creation date
                    try:
                        created_date = datetime.fromisoformat(meta['created_at']).strftime('%Y-%m-%d %H:%M')
                    except:
                        created_date = "Unknown"
                    
                    cache_info.append({
                        'hash': hash_part,
                        'file_path': cache_file,
                        'meta_file': meta_file,
                        'description': meta.get('description', 'Unknown'),
                        'created_at': created_date,
                        'num_transactions': meta.get('num_transactions', 0),
                        'num_claims': meta.get('num_claims', 0),
                        'cache_size_mb': round(cache_size, 1),
                        'input_files': meta.get('input_files', []),
                        'type': 'legacy'
                    })
                except Exception:
                    # Skip corrupted metadata files
                    continue
        
        # Sort by creation date (newest first), prioritizing structured folders
        cache_info.sort(key=lambda x: (x.get('extraction_date', ''), x['created_at']), reverse=True)
        return cache_info
    
    
    def _cleanup_old_cache_files(self, current_hash: str) -> None:
        """
        Clean up old cache files, keeping only the current one
        """
        import glob
        
        try:
            # Find all cache files
            cache_pattern = os.path.join("_data", "period_clm_*.parquet")
            meta_pattern = os.path.join("_data", "period_clm_*_meta.pkl")
            
            cache_files = glob.glob(cache_pattern)
            meta_files = glob.glob(meta_pattern)
            
            # Remove old files (not current)
            for file_path in cache_files:
                if current_hash not in file_path:
                    os.remove(file_path)
            
            for file_path in meta_files:
                if current_hash not in file_path:
                    os.remove(file_path)
                    
        except Exception as e:
            print(f"⚠️ Could not cleanup old cache files: {e}")
        
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
    
    def transform_claims_data_vectorized(self, df_txn: pd.DataFrame, force_recompute: bool = False, input_files: Optional[List[str]] = None, extraction_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fast vectorized transformation to get period data for all claims with structured caching
        
        Returns a DataFrame with all periods from all claims for fast analysis.
        Uses structured folder caching organized by extraction date.
        
        Args:
            df_txn: Raw transaction dataframe
            force_recompute: If True, recompute even if cache exists
            input_files: List of input CSV files used to create df_txn (for cache invalidation)
            extraction_date: Date string (YYYY-MM-DD) for organized folder structure
        """
        import os
        import hashlib
        import pickle
        
        # Determine cache paths based on extraction date or fallback to hash-based
        if extraction_date:
            # Use structured folder approach
            cache_dir = os.path.join("_data", extraction_date, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "period_clm.parquet")
            cache_meta_file = os.path.join(cache_dir, "period_clm_meta.pkl")
        else:
            # Fallback to hash-based approach
            data_hash = self._create_data_hash(df_txn, input_files)
            cache_file = os.path.join("_data", f"period_clm_{data_hash}.parquet")
            cache_meta_file = os.path.join("_data", f"period_clm_{data_hash}_meta.pkl")
        
        # Check if cache exists and is valid
        if not force_recompute and os.path.exists(cache_file) and os.path.exists(cache_meta_file):
            try:
                # Load metadata
                with open(cache_meta_file, 'rb') as f:
                    cache_meta = pickle.load(f)
                
                # Validate cache metadata
                if self._validate_cache_metadata(cache_meta, df_txn, input_files):
                    # Load from cache
                    cached_df = pd.read_parquet(cache_file)
                    
                    # Basic validation: check if cache has expected columns
                    expected_cols = ['clmNum', 'period', 'incremental_paid', 'incremental_expense', 'cumulative_paid', 'cumulative_expense']
                    if all(col in cached_df.columns for col in expected_cols):
                        print(f"📁 Loaded cached period data: {len(cached_df):,} periods from {cached_df['clmNum'].nunique():,} claims")
                        print(f"📊 Cache info: {cache_meta['data_version']} | {cache_meta['created_at']} | {cache_meta['num_transactions']:,} transactions")
                        return cached_df
                    else:
                        print("⚠️ Cached data missing expected columns, recomputing...")
                else:
                    print("⚠️ Cache metadata validation failed (data changed), recomputing...")
            except Exception as e:
                print(f"⚠️ Error loading cached data: {e}, recomputing...")
        
        # Compute fresh data
        print("🔄 Computing period data (this may take a while for large datasets)...")
        
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
            result_df = pd.DataFrame(all_periods)
            
            # Save to cache with metadata
            try:
                # Ensure _data directory exists
                os.makedirs("_data", exist_ok=True)
                
                # Create cache metadata
                cache_meta = self._create_cache_metadata(df_txn, result_df, input_files, extraction_date)
                
                # Save data as parquet
                result_df.to_parquet(cache_file, index=False)
                
                # Save metadata
                with open(cache_meta_file, 'wb') as f:
                    pickle.dump(cache_meta, f)
                
                print(f"💾 Cached period data: {len(result_df):,} periods from {result_df['clmNum'].nunique():,} claims")
                print(f"📊 Cache saved: {cache_meta['data_version']} | {cache_meta['num_transactions']:,} transactions")
                
                # Clean up old cache files
                self._cleanup_old_cache_files(data_hash)
                
            except Exception as e:
                print(f"⚠️ Could not save cache: {e}")
            
            return result_df
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
