
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/telco_cleaned.csv")

# Convert numeric columns safely
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("Original dataset shape:", df.shape)

# Inspect the rows with missing TotalCharges
missing_mask = df["TotalCharges"].isna()

print("Missing TotalCharges:", missing_mask.sum())
print(
    "Missing rows with tenure = 0:",
    (df.loc[missing_mask, "tenure"] == 0).all()
)
print(
    "Missing rows with Churn = No:",
    (df.loc[missing_mask, "Churn"] == "No").all()
)

# Apply the documented assumption only if it is valid
if (
    (df.loc[missing_mask, "tenure"] == 0).all()
    and (df.loc[missing_mask, "Churn"] == "No").all()
):
    df.loc[missing_mask, "TotalCharges"] = 0
else:
    raise ValueError(
        "Unexpected missing TotalCharges rows. Review before filling."
    )

# Confirm no missing TotalCharges remain
print(
    "Missing TotalCharges after handling:",
    df["TotalCharges"].isna().sum()
)

# Save a separate modeling-preparation dataset
df.to_csv("data/telco_ml_prepared.csv", index=False)

print("Prepared dataset shape:", df.shape)
print("Saved to data/telco_ml_prepared.csv")


# Inspect data types for machine-learning preparation
print("\nColumn data types:")
print(df.dtypes)

print("\nRemaining missing values:")
print(df.isnull().sum().sum())

print("\nChurn distribution:")
print(df["Churn"].value_counts(normalize=True).mul(100).round(2))
