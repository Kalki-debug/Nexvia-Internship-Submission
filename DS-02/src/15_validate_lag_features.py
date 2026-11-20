
import pandas as pd

print("=" * 60)
print("LAG AND ROLLING FEATURE VALIDATION")
print("=" * 60)

stores = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3"
]

expected_rows = 3049 * 1913

for store in stores:

    path = f"data/processed/lag_features_{store}.csv"

    print(f"\nChecking store: {store}")

    df = pd.read_csv(
        path,
        nrows=1000
    )

    print("Columns:", df.columns.tolist())

    print("\nFirst 5 records:")
    print(
        df[
            [
                "item_id", "d",
                "lag_1", "lag_7", "lag_28",
                "rolling_mean_7", "rolling_mean_28"
            ]
        ].head()
    )

    # Count rows using a lightweight chunked scan
    row_count = sum(
        len(chunk)
        for chunk in pd.read_csv(
            path,
            usecols=["d"],
            chunksize=500000
        )
    )

    print("\nActual rows:", f"{row_count:,}")
    print("Expected rows:", f"{expected_rows:,}")

    if row_count == expected_rows:
        print("Row count: PASS")
    else:
        print("Row count: CHECK")

    print("Sample inspection completed.")

print("\n" + "=" * 60)
print("VALIDATION SCAN COMPLETED!")
print("=" * 60)
