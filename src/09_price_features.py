
import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

# Load price dataset
prices = pd.read_csv("data/raw/sell_prices.csv")

print("=" * 60)
print("PRICE FEATURE ENGINEERING")
print("=" * 60)

print("\nOriginal price dataset shape:", prices.shape)

# Sort by item, store and week
prices = prices.sort_values(
    ["store_id", "item_id", "wm_yr_wk"]
).copy()

# Previous week's price
prices["previous_price"] = (
    prices.groupby(["store_id", "item_id"])["sell_price"]
    .shift(1)
)

# Price change
prices["price_change"] = (
    prices["sell_price"] - prices["previous_price"]
)

# Percentage price change
prices["price_change_pct"] = (
    prices.groupby(["store_id", "item_id"])["sell_price"]
    .pct_change() * 100
)

# Save processed price features
prices.to_csv(
    "data/processed/price_features.csv",
    index=False
)

print("\nNew features created:")
print([
    "previous_price",
    "price_change",
    "price_change_pct"
])

print("\nSample output:")
print(prices.head(15).to_string(index=False))

print("\nMissing previous prices:", prices["previous_price"].isna().sum())

print("\nProcessed dataset shape:", prices.shape)

print("\nSaved: data/processed/price_features.csv")
print("\nPrice feature engineering completed!")
