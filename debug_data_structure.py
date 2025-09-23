#!/usr/bin/env python3
"""
Debug script to check data structure recognition
"""

import os
import glob
from helpers.functions.load_cache_data import DataOrganizer, DataLoader

def main():
    print("🔍 Debugging Data Structure Recognition")
    print("=" * 50)
    
    # Check what folders exist
    print("\n📁 Checking _data folder structure:")
    data_dir = "_data"
    if os.path.exists(data_dir):
        print(f"✅ _data folder exists")
        
        # List all subdirectories
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"📂 Subdirectories in _data: {subdirs}")
        
        # Check each subdirectory
        for subdir in subdirs:
            subdir_path = os.path.join(data_dir, subdir)
            files = os.listdir(subdir_path)
            print(f"   📁 {subdir}: {files}")
            
            # Check if clm_with_amt.csv exists
            clm_file = os.path.join(subdir_path, "clm_with_amt.csv")
            if os.path.exists(clm_file):
                print(f"      ✅ clm_with_amt.csv exists")
                print(f"      📊 File size: {os.path.getsize(clm_file)} bytes")
            else:
                print(f"      ❌ clm_with_amt.csv not found")
    else:
        print(f"❌ _data folder does not exist")
    
    # Test DataOrganizer
    print("\n🔧 Testing DataOrganizer:")
    data_organizer = DataOrganizer()
    
    # Test list_extraction_dates
    extraction_dates = data_organizer.list_extraction_dates()
    print(f"📅 Extraction dates found: {extraction_dates}")
    
    # Test get_organized_files for each date
    for date in extraction_dates:
        files = data_organizer.get_organized_files(date)
        print(f"   📁 {date}: {files}")
    
    # Test DataLoader
    print("\n📊 Testing DataLoader:")
    data_loader = DataLoader()
    
    # Test get_available_data_versions
    versions = data_loader.get_available_data_versions()
    print(f"📋 Available versions: {versions}")
    
    # Test loading claims data
    if extraction_dates:
        test_date = extraction_dates[0]
        print(f"\n🧪 Testing load_claims_data for {test_date}:")
        df = data_loader.load_claims_data(extraction_date=test_date)
        if df is not None:
            print(f"✅ Successfully loaded data: {len(df)} rows")
            print(f"📊 Columns: {list(df.columns)}")
        else:
            print(f"❌ Failed to load data")

if __name__ == "__main__":
    main()
