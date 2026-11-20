
import pandas as pd
import os

print("=" * 60)
print("BUILDING COMBINED MODELLING DATASET")
print("=" * 60)

os.makedirs("data/processed", exist_ok=True)

# 1. Load datasets
print("\nLoading datasets...")

sales = pd.read_csv("data/raw/sales_train_validation.csv")
calendar = pd.read_csv("data/processed/calendar_features.csv")
prices = pd.read_csv("data/processed/price_features.csv")

print("Sales shape:", sales.shape)
print("Calendar shape:", calendar.shape)
print("Prices shape:", prices.shape)

# 2. Identify columns
metadata_columns = [
    "id", "item_id", "dept_id",
    "cat_id", "store_id", "state_id"
]

day_columns = [
    col for col in sales.columns
    if col.startswith("d_")
]

# 3. Process one store at a time
stores = sales["store_id"].unique()

print("\nStores to process:", len(stores))

for store in stores:

    print(f"\nProcessing store: {store}")

    store_sales = sales[sales["store_id"] == store].copy()

    # Convert wide sales data into long format
    store_long = store_sales.melt(
        id_vars=metadata_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="sales"
    )

    print("Long sales shape:", store_long.shape)

    # Merge calendar features
    combined = store_long.merge(
        calendar,
        on="d",
        how="left",
        sort=False
    )

    # Merge price features
    combined = combined.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        sort=False
    )

    # Preserve original product order
    series_order = {
        series_id: index
        for index, series_id in enumerate(store_sales["id"])
    }

    combined["series_order"] = combined["id"].map(series_order)

    # Extract numeric day
    combined["day_number"] = (
        combined["d"].str.replace("d_", "", regex=False).astype(int)
    )

    # Sort product-wise, then day-wise
    combined = combined.sort_values(
        ["series_order", "day_number"]
    )

    # Remove temporary columns
    combined = combined.drop(
        columns=["series_order", "day_number"]
    )

    # Save one file per store
    output_path = f"data/processed/combined_{store}.csv"

    combined.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print("Final shape:", combined.shape)

    del store_sales, store_long, combined

print("\n" + "=" * 60)
print("COMBINED DATASET CREATION COMPLETED!")
print("=" * 60)
