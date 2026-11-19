
import pandas as pd

stores = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3"
]

print("=" * 65)
print("CHECKING COMBINED AND LAG FEATURE ALIGNMENT")
print("=" * 65)

all_passed = True
expected_rows = 3049 * 1913

for store in stores:

    combined_path = (
        f"data/processed/combined_{store}.csv"
    )

    lag_path = (
        f"data/processed/lag_features_{store}.csv"
    )

    print(f"\nChecking {store}...")

    combined_reader = pd.read_csv(
        combined_path,
        usecols=["id", "d"],
        chunksize=200000
    )

    lag_reader = pd.read_csv(
        lag_path,
        usecols=["id", "d"],
        chunksize=200000
    )

    rows_checked = 0
    store_passed = True

    for combined_chunk, lag_chunk in zip(
        combined_reader, lag_reader
    ):

        # Explicitly use the same column order
        combined_chunk = combined_chunk[["id", "d"]]
        lag_chunk = lag_chunk[["id", "d"]]

        combined_chunk = combined_chunk.reset_index(drop=True)
        lag_chunk = lag_chunk.reset_index(drop=True)

        if not combined_chunk.equals(lag_chunk):
            print("Mismatch found near row:", rows_checked)
            store_passed = False
            break

        rows_checked += len(combined_chunk)

    if rows_checked != expected_rows:
        store_passed = False

    print("Rows checked:", f"{rows_checked:,}")
    print("Expected rows:", f"{expected_rows:,}")
    print("Alignment:", "PASS" if store_passed else "FAIL")

    if not store_passed:
        all_passed = False

print("\n" + "=" * 65)

if all_passed:
    print("ALL STORES PASSED ALIGNMENT CHECK!")
else:
    print("ALIGNMENT ISSUE FOUND. DO NOT INTEGRATE YET.")

print("=" * 65)
