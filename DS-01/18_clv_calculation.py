
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data" / "telco_ml_prepared.csv"
)

print("Dataset shape:", df.shape)

# ==========================================
# 2. CLEAN CLV INPUTS
# ==========================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["MonthlyCharges"] = pd.to_numeric(
    df["MonthlyCharges"], errors="coerce"
)

df["tenure"] = pd.to_numeric(
    df["tenure"], errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)

df["MonthlyCharges"] = df["MonthlyCharges"].fillna(
    df["MonthlyCharges"].median()
)

df["tenure"] = df["tenure"].fillna(0)

# ==========================================
# 3. CALCULATE HISTORICAL CLV
# ==========================================

df["historical_clv"] = df["TotalCharges"]

df["average_monthly_revenue"] = np.where(
    df["tenure"] > 0,
    df["TotalCharges"] / df["tenure"],
    0
)

# ==========================================
# 4. CUSTOMER VALUE SEGMENTS
# ==========================================

df["clv_segment"] = pd.qcut(
    df["historical_clv"].rank(method="first"),
    q=4,
    labels=[
        "Low Value",
        "Medium Value",
        "High Value",
        "Very High Value"
    ]
)

# ==========================================
# 5. SUMMARY
# ==========================================

print("\nCLV SUMMARY")
print("--------------------------------")

print("Total customers:", len(df))

print(
    "Average historical CLV:",
    round(df["historical_clv"].mean(), 2)
)

print(
    "Median historical CLV:",
    round(df["historical_clv"].median(), 2)
)

print(
    "Average monthly revenue:",
    round(df["average_monthly_revenue"].mean(), 2)
)

print("\nCustomer value distribution:")
print(df["clv_segment"].value_counts())

# ==========================================
# 6. CLV BY CHURN STATUS
# ==========================================

if "Churn" in df.columns:

    print("\nAverage historical CLV by churn:")
    print(
        df.groupby("Churn", observed=True)[
            "historical_clv"
        ].mean().round(2)
    )

# ==========================================
# 7. SAVE RESULTS
# ==========================================

output_path = (
    BASE_DIR / "data" / "telco_clv_results.csv"
)

df.to_csv(output_path, index=False)

print("\nSaved:", output_path)

# ==========================================
# 8. VISUALIZATION
# ==========================================

plt.figure(figsize=(9, 5))

df["clv_segment"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Customer Distribution by Historical CLV")
plt.xlabel("Customer Value Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.tight_layout()

chart_path = BASE_DIR / "models" / "clv_distribution.png"

plt.savefig(chart_path, dpi=300)
plt.close()

print("Chart saved:", chart_path)

print("\nCLV calculation completed.")
