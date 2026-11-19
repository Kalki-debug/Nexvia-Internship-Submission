
import pandas as pd

sales = pd.read_csv("data/raw/sales_train_validation.csv")

day_columns = [col for col in sales.columns if col.startswith("d_")]

# Ensure all daily sales columns are numeric
sales[day_columns] = sales[day_columns].apply(
    pd.to_numeric, errors="coerce"
)

daily_totals = sales[day_columns].sum(axis=0)

print("=" * 60)
print("DAILY SALES DIAGNOSTIC")
print("=" * 60)

print("Number of daily columns:", len(day_columns))
print("Total daily observations:", len(daily_totals))
print("Minimum daily total:", daily_totals.min())
print("Maximum daily total:", daily_totals.max())
print("Number of exact zero-sales days:", (daily_totals == 0).sum())
print("Number of missing daily totals:", daily_totals.isna().sum())

print("\n15 LOWEST SALES DAYS:")
print(daily_totals.nsmallest(15).to_string())

print("\nDAYS WITH SALES BELOW 1000:")
print(daily_totals[daily_totals < 1000].to_string())

print("\n" + "=" * 60)
