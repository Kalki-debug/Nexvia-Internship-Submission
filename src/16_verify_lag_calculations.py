
import pandas as pd
import numpy as np

print("=" * 60)
print("VERIFYING LAG FEATURES WITH NON-ZERO SALES")
print("=" * 60)

store = "CA_1"

# Load original sales
sales = pd.read_csv(
    "data/raw/sales_train_validation.csv"
)

# Find a product with sales during its first 28 days
day_columns = [f"d_{i}" for i in range(1, 1914)]

selected = None

for _, row in sales[sales["store_id"] == store].iterrows():

    values = row[day_columns].to_numpy(dtype=float)

    if values[:28].sum() > 0:
        selected = row
        break

if selected is None:
    raise ValueError("No suitable product found.")

item = selected["item_id"]
values = selected[day_columns].to_numpy(dtype=float)

print("\nSelected product:", item)
print("Store:", store)
print("Sales in first 28 days:", values[:28].sum())

# Load generated features
features = pd.read_csv(
    f"data/processed/lag_features_{store}.csv"
)

features = features[
    features["item_id"] == item
].set_index("d")

# Verify calculations on day 29
checks = {
    "lag_1": (
        features.loc["d_29", "lag_1"],
        values[27]
    ),
    "lag_7": (
        features.loc["d_29", "lag_7"],
        values[21]
    ),
    "lag_28": (
        features.loc["d_29", "lag_28"],
        values[0]
    ),
    "rolling_mean_7": (
        features.loc["d_29", "rolling_mean_7"],
        np.mean(values[21:28])
    ),
    "rolling_mean_28": (
        features.loc["d_29", "rolling_mean_28"],
        np.mean(values[:28])
    )
}

all_passed = True

for name, (actual, expected) in checks.items():

    passed = np.isclose(actual, expected)

    print(f"\n{name}")
    print("Generated:", actual)
    print("Expected: ", expected)
    print("Status:", "PASS" if passed else "FAIL")

    if not passed:
        all_passed = False

print("\n" + "=" * 60)

if all_passed:
    print("ALL NON-ZERO NUMERICAL CHECKS PASSED!")
else:
    print("VALIDATION FAILED!")

print("=" * 60)
