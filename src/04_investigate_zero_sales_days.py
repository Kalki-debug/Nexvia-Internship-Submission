
import pandas as pd

# Load datasets
sales = pd.read_csv("data/raw/sales_train_validation.csv")
calendar = pd.read_csv("data/raw/calendar.csv")

# Identify daily sales columns
day_columns = [col for col in sales.columns if col.startswith("d_")]

# Calculate total sales for each day
daily_totals = sales[day_columns].sum()

# Find days where total sales across all products and stores are zero
zero_days = daily_totals[daily_totals == 0].index

# Match day identifiers with calendar dates
result = calendar[calendar["d"].isin(zero_days)]

print("=" * 60)
print("INVESTIGATION OF ZERO-SALES DAYS")
print("=" * 60)

print(result[
    ["d", "date", "weekday", "event_name_1", "event_type_1"]
].to_string(index=False))

print("\nNumber of zero-sales days:", len(zero_days))
