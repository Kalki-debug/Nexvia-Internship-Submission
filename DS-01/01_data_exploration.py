
import pandas as pd
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# Load the telecom customer churn dataset
file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

# 1. Dataset dimensions
print("\n--- Dataset Shape ---")
print(df.shape)

# 2. First five customer records
print("\n--- First 5 Rows ---")
print(df.head())

# 3. Column names
print("\n--- Column Names ---")
print(df.columns.tolist())

# 4. Data types and non-null values
print("\n--- Dataset Information ---")
df.info()

# 5. Churn distribution
print("\n--- Churn Distribution ---")
print(df["Churn"].value_counts())

# 6. Missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# --- Data Quality and Initial Analysis ---

print("\n--- TotalCharges Data Cleaning ---")

# Convert TotalCharges from text to numeric.
# Blank or invalid values become NaN.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

print("Invalid or blank TotalCharges:", df["TotalCharges"].isna().sum())

print("\n--- Churn Percentage ---")
churn_percentage = df["Churn"].value_counts(normalize=True) * 100
print(churn_percentage.round(2))

print("\n--- Churn by Contract Type ---")
print(
    df.groupby("Contract", observed=True)["Churn"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .mul(100)
    .round(2)
)

print("\n--- Average Monthly Charges by Churn ---")
print(df.groupby("Churn", observed=True)["MonthlyCharges"].mean().round(2))

print("\n--- Average Tenure by Churn ---")
print(df.groupby("Churn", observed=True)["tenure"].mean().round(2))

print("\n--- Tenure Zero Records ---")
print((df["tenure"] == 0).sum())


# --- Investigate Invalid TotalCharges ---

print("\n--- Records with Missing TotalCharges ---")

invalid_charges = df[df["TotalCharges"].isna()]

print(
    invalid_charges[
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ].to_string(index=False)
)

print("\n--- Check Invalid Records ---")
print("Number of invalid records:", len(invalid_charges))
print("All have zero tenure:", (invalid_charges["tenure"] == 0).all())


# --- Visualise Churn Patterns ---

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# Chart 1: Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("churn_distribution.png", dpi=150)
plt.close()

# Chart 2: Churn rate by contract type
contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
contract_churn.plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("churn_by_contract.png", dpi=150)
plt.close()

print("\nCharts saved successfully:")
print(" - churn_distribution.png")
print(" - churn_by_contract.png")
