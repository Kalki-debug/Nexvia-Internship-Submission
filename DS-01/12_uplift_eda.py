
import pandas as pd
from pathlib import Path

# ==========================================
# 1. LOAD DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "data" / "hillstrom.csv")

print("Original dataset shape:", df.shape)

# ==========================================
# 2. SELECT TREATMENT AND CONTROL
# ==========================================

uplift_df = df[
    df["segment"].isin(["Womens E-Mail", "No E-Mail"])
].copy()

# Treatment: 1 = Women's Email
# Control:  0 = No Email

uplift_df["treatment"] = (
    uplift_df["segment"] == "Womens E-Mail"
).astype(int)

print("\nSelected dataset shape:", uplift_df.shape)

print("\nTreatment distribution:")
print(uplift_df["treatment"].value_counts())

# ==========================================
# 3. CHECK DATA QUALITY
# ==========================================

print("\nMissing values:")
print(uplift_df.isnull().sum())

print("\nDuplicate rows:", uplift_df.duplicated().sum())

# ==========================================
# 4. COMPARE CONVERSION RATES
# ==========================================

conversion_rates = uplift_df.groupby("treatment")["conversion"].agg(
    ["count", "mean", "sum"]
)

conversion_rates["conversion_rate_percent"] = (
    conversion_rates["mean"] * 100
)

print("\nConversion comparison:")
print(conversion_rates)

control_rate = conversion_rates.loc[0, "mean"]
treatment_rate = conversion_rates.loc[1, "mean"]

absolute_uplift = treatment_rate - control_rate

relative_uplift = (
    absolute_uplift / control_rate
) * 100

print("\nUPLIFT RESULTS")
print("--------------------------------")
print(f"Control conversion rate: {control_rate:.4%}")
print(f"Treatment conversion rate: {treatment_rate:.4%}")
print(f"Absolute uplift: {absolute_uplift:.4%}")
print(f"Relative uplift: {relative_uplift:.2f}%")

# ==========================================
# 5. PREPARE MODELING DATA
# ==========================================

# Only pre-campaign customer information is used
# Do NOT use visit, conversion or spend as features.

feature_columns = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel"
]

model_df = uplift_df[
    feature_columns + ["treatment", "conversion"]
].copy()

# Save prepared dataset
output_path = BASE_DIR / "data" / "hillstrom_uplift_prepared.csv"

model_df.to_csv(output_path, index=False)

print("\nPrepared modeling dataset:", model_df.shape)
print("Saved to:", output_path)

print("\nStage 5 data preparation completed.")
