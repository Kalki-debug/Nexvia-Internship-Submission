
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency

# Load prepared data
BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data" / "hillstrom_uplift_prepared.csv"
)

# Create contingency table
table = pd.crosstab(
    df["treatment"],
    df["conversion"]
)

print("Contingency Table:")
print(table)

# Chi-square test
chi2, p_value, dof, expected = chi2_contingency(table)

print("\nSTATISTICAL SIGNIFICANCE")
print("------------------------------")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.8f}")

if p_value < 0.05:
    print("Result: Statistically significant difference.")
else:
    print("Result: No statistically significant difference.")

print("\nSignificance test completed.")
