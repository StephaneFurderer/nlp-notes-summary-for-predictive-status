#!/usr/bin/env python3
"""
Example: Simplified Data Structure

This example demonstrates the simplified data organization:
- No subfolders, just _data/<date>/ with CSV files and cache directly inside
- No legacy hash-based caching support
- Clean and simple structure
"""

import pandas as pd
from helpers.functions.load_cache_data import DataOrganizer, CacheManager, DataLoader
from helpers.functions.standardized_claims_transformer import StandardizedClaimsTransformer
from helpers.functions.standardized_claims_schema import StandardizationConfig

def main():
    print("🔧 Simplified Data Structure Example")
    print("=" * 45)
    
    # Initialize components
    data_organizer = DataOrganizer()
    cache_manager = CacheManager()
    data_loader = DataLoader()
    
    print("\n📁 Expected Structure:")
    print("""
_data/
├── 2024-01-15/
│   ├── clm_with_amt.csv          # Claims data
│   ├── notes_summary.csv         # Notes data (optional)
│   ├── policy_info.csv           # Policy data (optional)
│   ├── period_clm.parquet        # Processed cache
│   └── period_clm_meta.pkl       # Cache metadata
├── 2024-02-01/
│   ├── clm_with_amt.csv
│   ├── period_clm.parquet
│   └── period_clm_meta.pkl
└── 2024-03-10/
    └── ... (same structure)
""")
    
    # Example 1: Organize data
    print("\n📁 Example 1: Organize Data")
    print("-" * 30)
    
    extraction_date = "2024-01-15"
    claims_file = "_data/clm_with_amt.csv"  # Your actual claims file
    
    print(f"Organizing data for extraction date: {extraction_date}")
    
    try:
        organized_files = data_organizer.organize_files(
            extraction_date=extraction_date,
            claims_file=claims_file,
            notes_file=None,  # Add if you have notes
            policy_file=None  # Add if you have policy data
        )
        
        print("✅ Files organized successfully:")
        for file_type, file_path in organized_files.items():
            print(f"  • {file_type}: {file_path}")
            
    except Exception as e:
        print(f"⚠️ Error organizing files: {e}")
    
    # Example 2: List available data versions
    print("\n📅 Example 2: List Available Data Versions")
    print("-" * 45)
    
    available_versions = data_loader.get_available_data_versions()
    
    if available_versions:
        print("Available data versions:")
        for version in available_versions:
            print(f"  📅 {version['extraction_date']}")
            print(f"     Files: {', '.join(version['file_types'])}")
            print(f"     Count: {version['num_files']} files")
            print()
    else:
        print("No organized data versions found")
    
    # Example 3: Load data
    print("\n📊 Example 3: Load Data")
    print("-" * 25)
    
    if available_versions:
        latest_date = available_versions[0]['extraction_date']
        print(f"Loading data for latest extraction date: {latest_date}")
        
        df_claims = data_loader.load_claims_data(extraction_date=latest_date)
        if df_claims is not None:
            print(f"✅ Loaded {len(df_claims):,} transactions from {df_claims['clmNum'].nunique():,} claims")
        else:
            print("⚠️ Could not load claims data")
    else:
        print("No data available to load")
    
    # Example 4: Cache management
    print("\n💾 Example 4: Cache Management")
    print("-" * 30)
    
    available_caches = cache_manager.list_available_caches()
    
    if available_caches:
        print("Available caches:")
        for cache in available_caches:
            print(f"  📅 {cache['extraction_date']} - {cache['description']}")
            print(f"    Size: {cache['cache_size_mb']} MB")
            print(f"    Created: {cache['created_at']}")
            print()
    else:
        print("No cached data available")
    
    # Example 5: Process with caching
    print("\n🔄 Example 5: Process Data with Caching")
    print("-" * 40)
    
    if available_versions:
        latest_date = available_versions[0]['extraction_date']
        df_claims = data_loader.load_claims_data(extraction_date=latest_date)
        
        if df_claims is not None:
            config = StandardizationConfig()
            transformer = StandardizedClaimsTransformer(config=config, debug_logging=False)
            
            input_files = data_organizer.get_organized_files(latest_date)
            print(f"Processing with input files: {input_files}")
            
            try:
                periods_df = transformer.transform_claims_data_vectorized(
                    df_claims, 
                    force_recompute=False,
                    input_files=input_files,
                    extraction_date=latest_date
                )
                
                if not periods_df.empty:
                    print(f"✅ Processed {len(periods_df):,} periods from {periods_df['clmNum'].nunique():,} claims")
                else:
                    print("⚠️ No periods generated")
                    
            except Exception as e:
                print(f"⚠️ Error processing data: {e}")
    
    print("\n✅ Simplified structure example completed!")
    print("\nKey Benefits:")
    print("• 🗂️ No unnecessary subfolders")
    print("• 📅 Clear extraction date organization")
    print("• 💾 Cache files alongside data")
    print("• 🚫 No legacy hash-based confusion")
    print("• 🎯 One simple approach")

if __name__ == "__main__":
    main()
