
import pandas as pd

print("=" * 60)
print("INVESTIGATING MISSING SELLING PRICES - CA_1")
print("=" * 60)

file_path = "data/processed/combined_CA_1.csv"

total_rows = 0
missing_prices = 0
missing_with_sales = 0
missing_with_zero_sales = 0

date_summary = {}
item_summary = {}

for chunk in pd.read_csv(
    file_path,
    usecols=[
        "item_id", "date", "sales", "sell_price"
    ],
    chunksize=500000
):

    total_rows += len(chunk)

    missing = chunk["sell_price"].isna()

    missing_prices += missing.sum()

    missing_with_sales += (
        missing & (chunk["sales"] > 0)
    ).sum()

    missing_with_zero_sales += (
        missing & (chunk["sales"] == 0)
    ).sum()

    # Missing prices by date
    daily = (
        chunk.assign(missing_price=missing)
        .groupby("date")
        .agg(
            missing_count=("missing_price", "sum"),
            total_count=("missing_price", "size")
        )
    )

    for date, row in daily.iterrows():
        if date not in date_summary:
            date_summary[date] = [0, 0]

        date_summary[date][0] += row["missing_count"]
        date_summary[date][1] += row["total_count"]

    # Missing prices by product
    products = chunk.groupby("item_id").agg(
        missing_count=("sell_price", lambda x: x.isna().sum()),
        total_count=("sell_price", "size")
    )

    for item, row in products.iterrows():
        if item not in item_summary:
            item_summary[item] = [0, 0]

        item_summary[item][0] += row["missing_count"]
        item_summary[item][1] += row["total_count"]

print("\nOVERALL SUMMARY")
print("-" * 40)

print("Total rows:", f"{total_rows:,}")
print("Missing prices:", f"{missing_prices:,}")
print("Missing with positive sales:", f"{missing_with_sales:,}")
print("Missing with zero sales:", f"{missing_with_zero_sales:,}")

print("\nMISSING PRICES BY DATE")
print("-" * 40)

daily_df = pd.DataFrame.from_dict(
    date_summary,
    orient="index",
    columns=["missing_count", "total_count"]
)

daily_df["missing_pct"] = (
    daily_df["missing_count"] /
    daily_df["total_count"] * 100
)

daily_df.index.name = "date"

print("\nFirst 10 dates:")
print(daily_df.sort_index().head(10))

print("\nLast 10 dates:")
print(daily_df.sort_index().tail(10))

print("\nMISSING PRICES BY PRODUCT")
print("-" * 40)

item_df = pd.DataFrame.from_dict(
    item_summary,
    orient="index",
    columns=["missing_count", "total_count"]
)

item_df.index.name = "item_id"

item_df["missing_pct"] = (
    item_df["missing_count"] /
    item_df["total_count"] * 100
)

print("\nProducts with the most missing prices:")
print(item_df.sort_values(
    "missing_count",
    ascending=False
).head(10))

print("\nInvestigation completed!")
