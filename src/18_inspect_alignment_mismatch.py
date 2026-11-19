
import pandas as pd

store = "CA_1"

combined_path = (
    f"data/processed/combined_{store}.csv"
)

lag_path = (
    f"data/processed/lag_features_{store}.csv"
)

combined = pd.read_csv(
    combined_path,
    usecols=["id", "item_id", "d"],
    nrows=10
)

lag = pd.read_csv(
    lag_path,
    usecols=["id", "item_id", "d"],
    nrows=10
)

print("=" * 60)
print("FIRST 10 ROWS - COMBINED DATASET")
print("=" * 60)
print(combined.to_string(index=False))

print("\n" + "=" * 60)
print("FIRST 10 ROWS - LAG FEATURES")
print("=" * 60)
print(lag.to_string(index=False))

print("\n" + "=" * 60)
print("CHECKING FIRST RECORD KEYS")
print("=" * 60)

print("Combined first key:")
print(combined.iloc[0][["id", "d"]].to_dict())

print("\nLag first key:")
print(lag.iloc[0][["id", "d"]].to_dict())

print("\nSame first key:",
      combined.iloc[0]["id"] == lag.iloc[0]["id"]
      and combined.iloc[0]["d"] == lag.iloc[0]["d"])
