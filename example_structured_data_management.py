#!/usr/bin/env python3
"""
Example: Structured Data Management with Load Cache Data Module

This example demonstrates how to use the new load_cache_data module for
organized data management by extraction date.
"""

import pandas as pd
from helpers.functions.load_cache_data import DataOrganizer, CacheManager, DataLoader
from helpers.functions.standardized_claims_transformer import StandardizedClaimsTransformer
from helpers.functions.standardized_claims_schema import StandardizationConfig

def main():
    print("🔧 Structured Data Management Example")
    print("=" * 50)
    
    # Initialize components
    data_organizer = DataOrganizer()
    cache_manager = CacheManager()
    data_loader = DataLoader()
    
    # Example 1: Organize data by extraction date
    print("\n📁 Example 1: Organize Data by Extraction Date")
    print("-" * 45)
    
    extraction_date = "2024-01-15"
    claims_file = "_data/clm_with_amt.csv"  # Your actual claims file
    
    print(f"Organizing data for extraction date: {extraction_date}")
    
    try:
        # Organize files into structured folders
        organized_files = data_organizer.organize_files(
            extraction_date=extraction_date,
            claims_file=claims_file,
            notes_file=None,  # Add if you have notes data
            policy_file=None  # Add if you have policy data
        )
        
        print("✅ Files organized successfully:")
        for file_type, file_path in organized_files.items():
            print(f"  • {file_type}: {file_path}")
            
    except Exception as e:
        print(f"⚠️ Error organizing files: {e}")
    
    # Example 2: List available data versions
    print("\n📅 Example 2: List Available Data Versions")
    print("-" * 40)
    
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
    
    # Example 3: Load data from organized structure
    print("\n📊 Example 3: Load Data from Organized Structure")
    print("-" * 50)
    
    if available_versions:
        latest_date = available_versions[0]['extraction_date']
        print(f"Loading data for latest extraction date: {latest_date}")
        
        # Load claims data
        df_claims = data_loader.load_claims_data(extraction_date=latest_date)
        if df_claims is not None:
            print(f"✅ Loaded {len(df_claims):,} transactions from {df_claims['clmNum'].nunique():,} claims")
        
        # Load notes data (if available)
        df_notes = data_loader.load_notes_data(extraction_date=latest_date)
        if df_notes is not None:
            print(f"✅ Loaded {len(df_notes):,} notes")
        else:
            print("ℹ️ No notes data available")
    
    # Example 4: Cache management
    print("\n💾 Example 4: Cache Management")
    print("-" * 30)
    
    available_caches = cache_manager.list_available_caches()
    
    if available_caches:
        print("Available caches:")
        for cache in available_caches:
            cache_type = "📅 Structured" if cache.get('type') == 'structured' else "🔧 Legacy"
            print(f"  {cache_type} - {cache['description']}")
            print(f"    Size: {cache['cache_size_mb']} MB")
            print(f"    Created: {cache['created_at']}")
            if cache.get('extraction_date'):
                print(f"    Date: {cache['extraction_date']}")
            print()
    else:
        print("No cached data available")
    
    # Example 5: Process data with caching
    print("\n🔄 Example 5: Process Data with Caching")
    print("-" * 40)
    
    if available_versions:
        latest_date = available_versions[0]['extraction_date']
        df_claims = data_loader.load_claims_data(extraction_date=latest_date)
        
        if df_claims is not None:
            # Initialize transformer
            config = StandardizationConfig()
            transformer = StandardizedClaimsTransformer(config=config, debug_logging=False)
            
            # Get input files for this extraction date
            input_files = data_organizer.get_organized_files(latest_date)
            
            print(f"Processing with input files: {input_files}")
            
            # Process with caching (this will use the new cache manager)
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
    
    print("\n✅ Structured data management example completed!")
    print("\nKey Benefits:")
    print("• 📁 Organized folder structure by extraction date")
    print("• 💾 Smart caching with metadata")
    print("• 🔄 Easy data version management")
    print("• 🎛️ Human-readable cache selection")

if __name__ == "__main__":
    main()
