
import pandas as pd
import numpy as np

print("=" * 60)
print("M5 DATA QUALITY ANALYSIS")
print("=" * 60)

# Load datasets
sales = pd.read_csv("data/raw/sales_train_validation.csv")
calendar = pd.read_csv("data/raw/calendar.csv")
prices = pd.read_csv("data/raw/sell_prices.csv")

# 1. Dataset dimensions
print("\n1. DATASET SHAPES")
print("Sales:", sales.shape)
print("Calendar:", calendar.shape)
print("Prices:", prices.shape)

# 2. Duplicate checks
print("\n2. DUPLICATE CHECKS")
print("Duplicate sales IDs:", sales["id"].duplicated().sum())
print(
    "Duplicate item-store combinations:",
    sales.duplicated(["item_id", "store_id"]).sum()
)
print(
    "Duplicate price combinations:",
    prices.duplicated(["item_id", "store_id", "wm_yr_wk"]).sum()
)

# 3. Missing values
print("\n3. MISSING VALUES")
print("Sales missing values:", sales.isnull().sum().sum())
print("Calendar missing values:", calendar.isnull().sum().sum())
print("Price missing values:", prices.isnull().sum().sum())

# 4. Sales statistics
day_columns = [
    col for col in sales.columns if col.startswith("d_")
]

sales_values = sales[day_columns]

print("\n4. SALES STATISTICS")
print("Total units sold:", sales_values.sum().sum())
print("Zero-sales observations:", (sales_values == 0).sum().sum())
print("Non-zero observations:", (sales_values > 0).sum().sum())
print("Average daily sales:", sales_values.mean().mean())

# 5. Product and store counts
print("\n5. PRODUCT AND STORE COUNTS")
print("Unique products:", sales["item_id"].nunique())
print("Unique stores:", sales["store_id"].nunique())
print("Unique departments:", sales["dept_id"].nunique())
print("Unique categories:", sales["cat_id"].nunique())
print("States:", sales["state_id"].unique())

# 6. Price statistics
print("\n6. PRICE STATISTICS")
print("Minimum price:", prices["sell_price"].min())
print("Maximum price:", prices["sell_price"].max())
print("Average price:", prices["sell_price"].mean())

print("\n" + "=" * 60)
print("DATA QUALITY ANALYSIS COMPLETED")
print("=" * 60)
