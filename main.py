from helpers.functions.claims_utils import transform_claims_raw_data

# Load data from a specific date
extraction_date = "2025-09-21"
data = transform_claims_raw_data(extraction_date=extraction_date)

# Unpack the data
df_raw_txn, closed_txn, open_txn, paid_txn, df_raw_final, closed_final, paid_final, open_final = data

print(f"Loaded {len(df_raw_txn)} transactions from {df_raw_txn['clmNum'].nunique()} claims")
print(f"Open claims: {len(open_final)}, Closed claims: {len(closed_final)}, Paid claims: {len(paid_final)}")
