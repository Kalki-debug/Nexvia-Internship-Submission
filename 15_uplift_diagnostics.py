
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data" / "hillstrom_uplift_predictions.csv"
)

print("UPLIFT MODEL DIAGNOSTICS")
print("--------------------------------")

# Actual conversion rates
print("\nActual conversion rates:")
print(
    df.groupby("actual_treatment")["actual_conversion"].mean()
)

# Predicted probabilities
print("\nPredicted probabilities:")
print("Control:", df["p_control"].mean())
print("Treatment:", df["p_treatment"].mean())

# Uplift distribution
print("\nUplift score statistics:")
print(df["uplift_score"].describe())

# Compare actual and predicted rates by group
print("\nActual vs predicted:")
for group, name in [(0, "Control"), (1, "Treatment")]:
    subset = df[df["actual_treatment"] == group]

    predicted_column = (
        "p_control" if group == 0 else "p_treatment"
    )

    print(f"\n{name}:")
    print("Actual conversion:", subset["actual_conversion"].mean())
    print("Predicted conversion:", subset[predicted_column].mean())

print("\nDiagnostics completed.")
