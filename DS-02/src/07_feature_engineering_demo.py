
import pandas as pd

print("=" * 65)
print("FEATURE ENGINEERING - LAG AND ROLLING FEATURES")
print("=" * 65)
import os

os.makedirs("data/processed", exist_ok=True)
# 1. Load sales dataset
sales = pd.read_csv("data/raw/sales_train_validation.csv")

# 2. Select one product-store combination
sample = sales.iloc[0]

print("\nSELECTED PRODUCT")
print("Product:", sample["item_id"])
print("Store:", sample["store_id"])
print("Category:", sample["cat_id"])

# 3. Extract daily sales columns
day_columns = [
    col for col in sales.columns if col.startswith("d_")
]

# 4. Convert wide format into time-series format
df = pd.DataFrame({
    "d": day_columns,
    "sales": sample[day_columns].values
})

# 5. Create numerical day index
df["day_number"] = (
    df["d"].str.replace("d_", "", regex=False).astype(int)
)

df = df.sort_values("day_number").reset_index(drop=True)

# 6. Create lag features
df["lag_1"] = df["sales"].shift(1)
df["lag_7"] = df["sales"].shift(7)
df["lag_28"] = df["sales"].shift(28)

# 7. Create rolling average features
# Shift first to prevent using the current day's sales
df["rolling_mean_7"] = (
    df["sales"].shift(1).rolling(window=7).mean()
)

df["rolling_mean_28"] = (
    df["sales"].shift(1).rolling(window=28).mean()
)

# 8. Display selected rows
print("\nFEATURE ENGINEERING RESULTS")

print(
    df[
        [
            "d",
            "sales",
            "lag_1",
            "lag_7",
            "lag_28",
            "rolling_mean_7",
            "rolling_mean_28"
        ]
    ].iloc[25:40].to_string(index=False)
)

# 9. Save output
df.to_csv(
    "data/processed/feature_engineering_demo.csv",
    index=False
)

print("\nSaved: data/processed/feature_engineering_demo.csv")

print("\n" + "=" * 65)
print("FEATURE ENGINEERING DEMO COMPLETED")
print("=" * 65)
