
import pandas as pd
import os

print("=" * 65)
print("INTEGRATING LAG AND ROLLING FEATURES")
print("=" * 65)

os.makedirs("data/processed", exist_ok=True)

stores = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3"
]

feature_columns = [
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]

chunksize = 200_000

for store in stores:

    print(f"\nProcessing store: {store}")

    combined_path = f"data/processed/combined_{store}.csv"
    lag_path = f"data/processed/lag_features_{store}.csv"
    output_path = f"data/processed/modeling_{store}.csv"

    combined_reader = pd.read_csv(
        combined_path,
        chunksize=chunksize
    )

    lag_reader = pd.read_csv(
        lag_path,
        usecols=["id", "d"] + feature_columns,
        chunksize=chunksize
    )

    total_rows = 0
    first_chunk = True

    try:
        for combined_chunk, lag_chunk in zip(
            combined_reader, lag_reader
        ):

            # Check row counts
            if len(combined_chunk) != len(lag_chunk):
                raise ValueError(
                    f"Row count mismatch in {store}"
                )

            # Verify product and day alignment
            if not combined_chunk["id"].equals(lag_chunk["id"]):
                raise ValueError(
                    f"ID mismatch in {store}, "
                    f"near row {total_rows}"
                )

            if not combined_chunk["d"].equals(lag_chunk["d"]):
                raise ValueError(
                    f"Day mismatch in {store}, "
                    f"near row {total_rows}"
                )

            # Attach lag and rolling features
            combined_chunk[feature_columns] = (
                lag_chunk[feature_columns].to_numpy()
            )

            # Save integrated chunk
            combined_chunk.to_csv(
                output_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            first_chunk = False
            total_rows += len(combined_chunk)

        # Ensure neither file has leftover chunks
        if next(combined_reader, None) is not None:
            raise ValueError(
                f"Extra rows found in combined file: {store}"
            )

        if next(lag_reader, None) is not None:
            raise ValueError(
                f"Extra rows found in lag file: {store}"
            )

    finally:
        combined_reader.close()
        lag_reader.close()

    print("Integrated rows:", f"{total_rows:,}")
    print("Saved:", output_path)
    print("Status: PASS")

print("\n" + "=" * 65)
print("FEATURE INTEGRATION COMPLETED!")
print("=" * 65)
