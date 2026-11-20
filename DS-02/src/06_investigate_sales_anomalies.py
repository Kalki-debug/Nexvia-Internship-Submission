
import pandas as pd

sales = pd.read_csv("data/raw/sales_train_validation.csv")
calendar = pd.read_csv("data/raw/calendar.csv")

day_columns = [col for col in sales.columns if col.startswith("d_")]

# Identify unusually low-sales days
daily_totals = sales[day_columns].sum()
anomaly_days = daily_totals[daily_totals < 1000].index.tolist()

print("=" * 65)
print("INVESTIGATING UNUSUALLY LOW SALES DAYS")
print("=" * 65)

# Match anomaly days with calendar dates
print("\n1. CALENDAR DETAILS")

anomaly_calendar = calendar[calendar["d"].isin(anomaly_days)]

print(
    anomaly_calendar[
        ["d", "date", "weekday", "event_name_1", "event_type_1"]
    ].to_string(index=False)
)

# Investigate each store separately
print("\n2. STORE-WISE SALES")

for day in anomaly_days:
    store_totals = sales.groupby("store_id")[day].sum()

    print(f"\n{day} | Total sales: {daily_totals[day]}")
    print(store_totals.to_string())

# Count products with positive sales
print("\n3. PRODUCT AVAILABILITY / POSITIVE SALES")

for day in anomaly_days:
    positive_products = (sales[day] > 0).sum()
    zero_products = (sales[day] == 0).sum()

    print(f"\n{day}")
    print("Products with sales:", positive_products)
    print("Products with zero sales:", zero_products)

print("\n" + "=" * 65)
print("ANOMALY INVESTIGATION COMPLETED")
print("=" * 65)
