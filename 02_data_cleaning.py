
import pandas as pd
from pathlib import Path

# File paths
DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_PATH = Path("data/telco_cleaned.csv")

# Load original data
df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

# Remove accidental whitespace in column names and text fields
df.columns = df.columns.str.strip()
for column in df.select_dtypes(include=["object", "str"]).columns:
    df[column] = df[column].str.strip()

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Data quality checks
print("\nDuplicate customer IDs:", df["customerID"].duplicated().sum())
print("Missing TotalCharges:", df["TotalCharges"].isna().sum())
print("Invalid tenure values:", (~df["tenure"].between(0, 72)).sum())
print("Invalid MonthlyCharges:", (df["MonthlyCharges"] < 0).sum())

# Inspect zero-tenure customers
zero_tenure = df[df["tenure"] == 0]
print("\nZero-tenure customers:", len(zero_tenure))
print("Their churn distribution:")
print(zero_tenure["Churn"].value_counts(dropna=False))

# Save a separate cleaned copy.
# We retain the 11 zero-tenure customers and their missing TotalCharges.
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved to:", OUTPUT_PATH)
print("Cleaned dataset shape:", df.shape)
print("Remaining missing values:")
print(df.isna().sum()[df.isna().sum() > 0])
