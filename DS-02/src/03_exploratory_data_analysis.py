
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("M5 RETAIL DEMAND FORECASTING - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Create output folder for charts
os.makedirs("reports/figures", exist_ok=True)

# Load datasets
sales = pd.read_csv("data/raw/sales_train_validation.csv")
calendar = pd.read_csv("data/raw/calendar.csv")

# Identify daily sales columns
day_columns = [
    col for col in sales.columns if col.startswith("d_")
]

# --------------------------------------------------
# 1. DAILY SALES TREND
# --------------------------------------------------

daily_sales = sales[day_columns].sum(axis=0)

daily_sales.index = range(1, len(daily_sales) + 1)

plt.figure(figsize=(14, 5))
plt.plot(daily_sales.index, daily_sales.values)

plt.title("Total Daily Sales Across All Products and Stores")
plt.xlabel("Day")
plt.ylabel("Units Sold")
plt.grid(True)

plt.tight_layout()
plt.savefig("reports/figures/daily_sales_trend.png")
plt.show()

print("\n1. DAILY SALES ANALYSIS")
print("Average daily total sales:", daily_sales.mean())
print("Highest daily sales:", daily_sales.max())
print("Lowest daily sales:", daily_sales.min())

# --------------------------------------------------
# 2. WEEKDAY SALES PATTERN
# --------------------------------------------------

sales_transposed = sales[day_columns].T

sales_long = sales_transposed.stack().reset_index()
sales_long.columns = ["d", "product_index", "sales"]

sales_long = sales_long.merge(
    calendar[["d", "weekday", "month", "year"]],
    on="d",
    how="left"
)

weekday_sales = (
    sales_long.groupby("weekday")["sales"]
    .mean()
)

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_sales = weekday_sales.reindex(weekday_order)

plt.figure(figsize=(10, 5))
weekday_sales.plot(kind="bar")

plt.title("Average Product Sales by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Average Units Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/figures/weekday_sales.png")
plt.show()

print("\n2. WEEKDAY SALES ANALYSIS")
print(weekday_sales)

# --------------------------------------------------
# 3. MONTHLY SALES PATTERN
# --------------------------------------------------

monthly_sales = (
    sales_long.groupby("month")["sales"]
    .mean()
)

plt.figure(figsize=(10, 5))
monthly_sales.plot(kind="bar")

plt.title("Average Product Sales by Month")
plt.xlabel("Month")
plt.ylabel("Average Units Sold")
plt.tight_layout()
plt.savefig("reports/figures/monthly_sales.png")
plt.show()

print("\n3. MONTHLY SALES ANALYSIS")
print(monthly_sales)

# --------------------------------------------------
# 4. CATEGORY SALES
# --------------------------------------------------

category_sales = sales.groupby("cat_id")[day_columns].sum().sum(axis=1)

plt.figure(figsize=(8, 5))
category_sales.plot(kind="bar")

plt.title("Total Sales by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Units Sold")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/figures/category_sales.png")
plt.show()

print("\n4. CATEGORY SALES")
print(category_sales.sort_values(ascending=False))

# --------------------------------------------------
# 5. STORE SALES
# --------------------------------------------------

store_sales = sales.groupby("store_id")[day_columns].sum().sum(axis=1)

plt.figure(figsize=(10, 5))
store_sales.plot(kind="bar")

plt.title("Total Sales by Store")
plt.xlabel("Store")
plt.ylabel("Total Units Sold")
plt.tight_layout()
plt.savefig("reports/figures/store_sales.png")
plt.show()

print("\n5. STORE SALES")
print(store_sales.sort_values(ascending=False))

# --------------------------------------------------
# 6. DEPARTMENT SALES
# --------------------------------------------------

department_sales = (
    sales.groupby("dept_id")[day_columns]
    .sum()
    .sum(axis=1)
)

plt.figure(figsize=(12, 5))
department_sales.plot(kind="bar")

plt.title("Total Sales by Department")
plt.xlabel("Department")
plt.ylabel("Total Units Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/figures/department_sales.png")
plt.show()

print("\n6. DEPARTMENT SALES")
print(department_sales.sort_values(ascending=False))

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("Charts saved in reports/figures/")
print("=" * 60)
