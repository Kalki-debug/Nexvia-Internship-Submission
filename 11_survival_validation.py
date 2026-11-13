
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# ==========================================
# 1. LOAD DATA
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "data" / "telco_ml_prepared.csv")

df["event"] = (df["Churn"] == "Yes").astype(int)

# ==========================================
# 2. PREPARE COX DATA
# ==========================================

cox_df = df[
    [
        "tenure",
        "event",
        "Contract",
        "InternetService",
        "MonthlyCharges",
        "SeniorCitizen",
        "OnlineSecurity",
        "TechSupport"
    ]
].copy()

cox_df = pd.get_dummies(
    cox_df,
    columns=[
        "Contract",
        "InternetService",
        "OnlineSecurity",
        "TechSupport"
    ],
    drop_first=True,
    dtype=int
)

# Remove zero-tenure records
cox_df = cox_df[cox_df["tenure"] > 0].copy()

# Remove redundant columns
cox_df.drop(
    columns=[
        "OnlineSecurity_No internet service",
        "TechSupport_No internet service"
    ],
    inplace=True
)

print("Prepared dataset:", cox_df.shape)

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

train_df, test_df = train_test_split(
    cox_df,
    test_size=0.20,
    random_state=42,
    stratify=cox_df["event"]
)

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

# ==========================================
# 4. BASELINE COX MODEL
# ==========================================

baseline_model = CoxPHFitter(penalizer=0.01)

baseline_model.fit(
    train_df,
    duration_col="tenure",
    event_col="event"
)

baseline_risk = baseline_model.predict_partial_hazard(test_df)

# Higher risk means shorter survival, so reverse the score
baseline_c_index = concordance_index(
    test_df["tenure"],
    -baseline_risk.values.ravel(),
    test_df["event"]
)

# ==========================================
# 5. STRATIFIED COX MODEL
# ==========================================

adjusted_model = CoxPHFitter(penalizer=0.01)

adjusted_model.fit(
    train_df,
    duration_col="tenure",
    event_col="event",
    strata=["Contract_One year"]
)

adjusted_risk = adjusted_model.predict_partial_hazard(test_df)

adjusted_c_index = concordance_index(
    test_df["tenure"],
    -adjusted_risk.values.ravel(),
    test_df["event"]
)

# ==========================================
# 6. RESULTS
# ==========================================

print("\nSURVIVAL MODEL TEST RESULTS")
print("--------------------------------")

print(f"Baseline Cox Test C-index: {baseline_c_index:.4f}")
print(f"Stratified Cox Test C-index: {adjusted_c_index:.4f}")

print("\nValidation completed.")
