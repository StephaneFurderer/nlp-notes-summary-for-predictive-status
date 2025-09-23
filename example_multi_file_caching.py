#!/usr/bin/env python3
"""
Example: Multi-File Caching System for Claims Processing

This example demonstrates how to use the enhanced caching system that tracks
multiple input CSV files (claims, notes, etc.) and provides human-readable
cache selection.
"""

import pandas as pd
from helpers.functions.standardized_claims_transformer import StandardizedClaimsTransformer
from helpers.functions.standardized_claims_schema import StandardizationConfig

def main():
    print("🔧 Multi-File Caching Example")
    print("=" * 50)
    
    # Create transformer with custom config
    config = StandardizationConfig(
        period_length_days=30,
        max_periods=20
    )
    transformer = StandardizedClaimsTransformer(config=config, debug_logging=False)
    
    # Example 1: Process with single CSV file
    print("\n📁 Example 1: Single CSV file")
    print("-" * 30)
    
    # Simulate loading claims data (replace with your actual CSV path)
    input_files_1 = ["_data/clm_with_amt.csv"]
    
    # Check available caches before processing
    print("Available caches before processing:")
    caches = transformer.list_available_caches()
    for cache in caches:
        print(f"  • {cache['created_at']} - {cache['description']} ({cache['cache_size_mb']} MB)")
        print(f"    Files: {cache['input_files']}")
        print(f"    Hash: {cache['hash'][:16]}...")
    
    # Example 2: Process with multiple CSV files
    print("\n📁 Example 2: Multiple CSV files (claims + notes)")
    print("-" * 50)
    
    # Simulate having multiple input files
    input_files_2 = [
        "_data/clm_with_amt.csv",
        "_data/notes_summary.csv",  # Hypothetical notes file
        "_data/policy_info.csv"     # Hypothetical policy file
    ]
    
    print("Input files for this processing run:")
    for file_path in input_files_2:
        print(f"  • {file_path}")
    
    # Example 3: Cache selection UI simulation
    print("\n🎛️ Example 3: Cache Selection UI")
    print("-" * 35)
    
    available_caches = transformer.list_available_caches()
    
    if available_caches:
        print("Available cache versions:")
        for i, cache in enumerate(available_caches):
            print(f"  [{i+1}] {cache['created_at']} - {cache['description']}")
            print(f"      Files: {', '.join(cache['input_files'])}")
            print(f"      Size: {cache['cache_size_mb']} MB")
            print(f"      Hash: {cache['hash'][:16]}...")
            print()
        
        # Simulate user selection
        selected_cache = available_caches[0]  # Select newest
        print(f"✅ Selected: {selected_cache['description']}")
        print(f"   Files: {', '.join(selected_cache['input_files'])}")
    else:
        print("No cached data available")
    
    # Example 4: How to use in your Streamlit app
    print("\n🚀 Example 4: Usage in Streamlit App")
    print("-" * 40)
    
    print("""
# In your Streamlit app, you can now do:

# 1. Get available caches
available_caches = transformer.list_available_caches()

# 2. Create selection UI
if available_caches:
    cache_options = {}
    for cache in available_caches:
        cache_options[cache['hash']] = f"{cache['created_at']} - {cache['description']} ({cache['cache_size_mb']} MB)"
    
    selected_cache = st.selectbox(
        "📁 Select Cache Version:",
        options=list(cache_options.keys()),
        format_func=lambda x: cache_options[x],
        index=0
    )

# 3. Process with input file tracking
input_files = ["_data/clm_with_amt.csv", "_data/notes_summary.csv"]
periods_df = transformer.transform_claims_data_vectorized(
    df_raw_txn, 
    force_recompute=False,
    input_files=input_files  # This creates a unique cache for this file combination
)

# 4. Show cache details
selected_cache_info = next((c for c in available_caches if c['hash'] == selected_cache), None)
if selected_cache_info:
    st.json({
        "Description": selected_cache_info['description'],
        "Files": selected_cache_info['input_files'],
        "Created": selected_cache_info['created_at'],
        "Size": f"{selected_cache_info['cache_size_mb']} MB"
    })
""")
    
    print("\n✅ Multi-file caching system ready!")
    print("\nKey Benefits:")
    print("• 📁 Tracks multiple input CSV files")
    print("• 🔍 Human-readable cache descriptions")
    print("• ⚡ Automatic cache invalidation when files change")
    print("• 🎛️ Easy cache selection UI")
    print("• 💾 Efficient storage with metadata")

if __name__ == "__main__":
    main()
