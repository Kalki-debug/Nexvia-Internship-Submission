
import os
import pandas as pd

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

INPUT_FILE = (
    "reports/baseline/"
    "lightgbm_recursive_predictions_CA_1_100.csv"
)

OUTPUT_DIR = "data/processed/inventory"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "inventory_demand_CA_1_100.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD FORECAST DATA
# --------------------------------------------------

print("Loading LightGBM forecast data...")

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)

# --------------------------------------------------
# 3. AGGREGATE DEMAND
# --------------------------------------------------

inventory_data = (
    df.groupby("item_id")
    .agg(
        total_forecast_demand=("predicted_sales", "sum"),
        average_daily_demand=("predicted_sales", "mean"),
        total_actual_demand=("actual_sales", "sum"),
        forecast_days=("day_number", "nunique")
    )
    .reset_index()
)

# --------------------------------------------------
# 4. CALCULATE SAFETY STOCK
# --------------------------------------------------

# Initial demonstration assumption:
# Safety stock = 20% of forecast demand

inventory_data["safety_stock"] = (
    inventory_data["total_forecast_demand"] * 0.20
).round().astype(int)

# --------------------------------------------------
# 5. CALCULATE TARGET INVENTORY
# --------------------------------------------------

inventory_data["target_inventory"] = (
    inventory_data["total_forecast_demand"].round().astype(int)
    + inventory_data["safety_stock"]
)

# --------------------------------------------------
# 6. SAVE OUTPUT
# --------------------------------------------------

inventory_data.to_csv(OUTPUT_FILE, index=False)

print("\nInventory demand preparation completed!")

print("Products:", len(inventory_data))
print("Forecast horizon:", inventory_data["forecast_days"].unique())

print("\nSample output:")
print(inventory_data.head())

print("\nSaved:", OUTPUT_FILE)
