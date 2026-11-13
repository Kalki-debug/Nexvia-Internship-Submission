
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from pathlib import Path

# 1. Load prepared customer data
df = pd.read_csv("data/telco_ml_prepared.csv")

# 2. Define survival duration and event
# tenure = observed duration in months
# Churn = Yes means event observed
df["event"] = (df["Churn"] == "Yes").astype(int)

print("Dataset shape:", df.shape)
print("\nEvent distribution:")
print(df["event"].value_counts().rename(
    index={0: "No churn (censored)", 1: "Churn event"}
))

# 3. Kaplan-Meier survival curve
kmf = KaplanMeierFitter()

kmf.fit(
    durations=df["tenure"],
    event_observed=df["event"],
    label="Customer retention"
)

print("\nKaplan-Meier survival estimates:")
print(kmf.survival_function_.head(15))

# Save Kaplan-Meier plot
output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(9, 5))
kmf.plot_survival_function()
plt.title("Kaplan-Meier Customer Survival Curve")
plt.xlabel("Tenure (months)")
plt.ylabel("Probability of Remaining a Customer")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    output_dir / "kaplan_meier_survival.png",
    dpi=150
)
plt.close()

# 4. Prepare data for Cox Proportional Hazards model
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

# Encode categorical variables
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

# Remove zero-duration records for Cox model
# These records have no positive observed follow-up duration.
cox_df = cox_df[cox_df["tenure"] > 0].copy()

# Remove redundant dummy variables
cox_df = cox_df.drop(
    columns=[
        "OnlineSecurity_No internet service",
        "TechSupport_No internet service"
    ]
)
# Investigate nonlinear effect of MonthlyCharges
cox_df["MonthlyCharges_squared"] = (
    cox_df["MonthlyCharges"] ** 2
)

print("\nAdded nonlinear MonthlyCharges feature.")
print(
    cox_df[
        ["MonthlyCharges", "MonthlyCharges_squared"]
    ].head()
)
# Check event distribution for PH-violating categorical variables

ph_variables = [
    "Contract_One year",
    "InternetService_Fiber optic",
    "OnlineSecurity_Yes",
    "TechSupport_Yes"
]

print("\nPH variable event distribution:")

for col in ph_variables:
    print(f"\n{col}:")
    print(
        cox_df.groupby(col)["event"]
        .agg(["count", "sum"])
        .rename(columns={
            "count": "total_customers",
            "sum": "churn_events"
        })
    )
print("\nCox dataset shape after removing redundant columns:")
print(cox_df.shape)
# 5. Fit Cox model
cph = CoxPHFitter(penalizer=0.01)

cph.fit(
    cox_df,
    duration_col="tenure",
    event_col="event"
)

# Adjusted Cox model with stratification
adjusted_cph = CoxPHFitter(penalizer=0.01)

adjusted_cph.fit(
    cox_df,
    duration_col="tenure",
    event_col="event",
    strata=["Contract_One year"]
)

print("\nAdjusted Cox model summary:")
print(
    adjusted_cph.summary[
        ["coef", "exp(coef)", "p",
         "exp(coef) lower 95%", "exp(coef) upper 95%"]
    ].sort_values("p")
)

# Save adjusted model summary
adjusted_cph.summary.to_csv(
    output_dir / "adjusted_cox_model_summary.csv"
)

print("\nChecking adjusted model PH assumption...")

adjusted_cph.check_assumptions(
    cox_df,
    p_value_threshold=0.05,
    show_plots=True
)

print("\nCox Proportional Hazards summary:")
print(cph.summary[
    ["coef", "exp(coef)", "p", "exp(coef) lower 95%", "exp(coef) upper 95%"]
].sort_values("p"))

# Save Cox model summary
cph.summary.to_csv(output_dir / "cox_model_summary.csv")

# Save a coefficient plot
plt.figure(figsize=(9, 6))
cph.plot()
plt.title("Cox Proportional Hazards Model")
plt.tight_layout()
plt.savefig(
    output_dir / "cox_coefficients.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# Check the proportional hazards assumption
# Check the proportional hazards assumption
print("\nChecking Cox proportional hazards assumption...")

diagnostic_dir = output_dir / "cox_diagnostics"
diagnostic_dir.mkdir(parents=True, exist_ok=True)

# Run the diagnostic only once
cph.check_assumptions(
    cox_df,
    p_value_threshold=0.05,
    show_plots=True
)

# Save generated diagnostic plots
for number, figure_number in enumerate(plt.get_fignums(), start=1):
    fig = plt.figure(figure_number)

    fig.savefig(
        diagnostic_dir / f"cox_diagnostic_{number}.png",
        dpi=150,
        bbox_inches="tight"
    )

plt.close("all")

print("Saved Cox diagnostic plots to:", diagnostic_dir)

print("\nSaved outputs:")
print("- models/kaplan_meier_survival.png")
print("- models/cox_model_summary.csv")
print("- models/cox_coefficients.png")
print("- models/cox_diagnostics/")
