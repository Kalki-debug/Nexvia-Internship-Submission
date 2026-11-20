
import pandas as pd

stores = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3"
]

total_missing = 0
total_missing_positive = 0
total_missing_zero = 0

print("=" * 65)
print("MISSING PRICE VALIDATION - ALL STORES")
print("=" * 65)

for store in stores:

    file_path = f"data/processed/combined_{store}.csv"

    missing_count = 0
    missing_positive = 0
    missing_zero = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=["sales", "sell_price"],
        chunksize=500000
    ):

        missing = chunk["sell_price"].isna()

        missing_count += missing.sum()

        missing_positive += (
            missing & (chunk["sales"] > 0)
        ).sum()

        missing_zero += (
            missing & (chunk["sales"] == 0)
        ).sum()

    total_missing += missing_count
    total_missing_positive += missing_positive
    total_missing_zero += missing_zero

    print(f"\nStore: {store}")
    print(f"Missing prices: {missing_count:,}")
    print(f"Missing + positive sales: {missing_positive:,}")
    print(f"Missing + zero sales: {missing_zero:,}")

print("\n" + "=" * 65)
print("OVERALL RESULTS")
print("=" * 65)

print("Total missing prices:", f"{total_missing:,}")
print("Missing + positive sales:", f"{total_missing_positive:,}")
print("Missing + zero sales:", f"{total_missing_zero:,}")

print("\nInvestigation completed!")
