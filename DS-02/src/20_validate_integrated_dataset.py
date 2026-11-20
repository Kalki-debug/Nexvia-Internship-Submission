
import pandas as pd
import os

print("=" * 65)
print("VALIDATING FINAL MODELLING DATASETS")
print("=" * 65)

stores = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3"
]

expected_rows = 5_832_737
expected_columns = 35

feature_columns = [
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]

all_passed = True

for store in stores:

    print(f"\nChecking {store}...")

    path = f"data/processed/modeling_{store}.csv"

    if not os.path.exists(path):
        print("FAIL: File not found")
        all_passed = False
        continue

    # Read only the header first
    columns = pd.read_csv(path, nrows=0).columns.tolist()

    # Count rows without loading the entire dataset
    row_count = sum(
        len(chunk)
        for chunk in pd.read_csv(path, usecols=["id"], chunksize=200_000)
    )

    missing_features = [
        col for col in feature_columns
        if col not in columns
    ]

    print("Rows:", f"{row_count:,}")
    print("Columns:", len(columns))
    print("Missing feature columns:", missing_features)

    if row_count != expected_rows:
        print("Status: FAIL - Incorrect row count")
        all_passed = False

    elif len(columns) != expected_columns:
        print("Status: FAIL - Incorrect column count")
        all_passed = False

    elif missing_features:
        print("Status: FAIL - Missing features")
        all_passed = False

    else:
        print("Status: PASS")

print("\n" + "=" * 65)

if all_passed:
    print("ALL MODELLING DATASETS PASSED VALIDATION!")
else:
    print("VALIDATION ISSUES FOUND. CHECK THE OUTPUT.")

print("=" * 65)
