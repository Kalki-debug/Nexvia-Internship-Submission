
import pandas as pd
import os

print("=" * 60)
print("VALIDATING COMBINED DATASETS")
print("=" * 60)

stores = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3"
]

expected_rows = 3049 * 1913
total_rows = 0

for store in stores:

    file_path = f"data/processed/combined_{store}.csv"

    print(f"\nChecking {store}...")

    row_count = 0
    missing_dates = 0
    missing_weeks = 0
    missing_prices = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=["date", "wm_yr_wk", "sell_price"],
        chunksize=500000
    ):

        row_count += len(chunk)

        missing_dates += chunk["date"].isna().sum()
        missing_weeks += chunk["wm_yr_wk"].isna().sum()
        missing_prices += chunk["sell_price"].isna().sum()

    total_rows += row_count

    print("Rows:", f"{row_count:,}")
    print("Expected rows:", f"{expected_rows:,}")
    print("Missing dates:", f"{missing_dates:,}")
    print("Missing week IDs:", f"{missing_weeks:,}")
    print("Missing selling prices:", f"{missing_prices:,}")

    if row_count == expected_rows:
        print("Row count: PASS")
    else:
        print("Row count: CHECK")

print("\n" + "=" * 60)
print("OVERALL VALIDATION")
print("=" * 60)

print("Total rows:", f"{total_rows:,}")
print("Expected total:", f"{expected_rows * len(stores):,}")

print("\nValidation scan completed!")
