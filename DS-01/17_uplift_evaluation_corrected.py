
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# ==========================================
# 1. LOAD TEST PREDICTIONS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data" / "hillstrom_uplift_predictions.csv"
)

df = df.sort_values(
    "uplift_score",
    ascending=False
).reset_index(drop=True)

treatment = df["actual_treatment"].to_numpy()
outcome = df["actual_conversion"].to_numpy()

n = len(df)
p_treatment = treatment.mean()
p_control = 1 - p_treatment

# ==========================================
# 2. IPW CUSTOMER CONTRIBUTION
# ==========================================

contribution = (
    treatment * outcome / p_treatment
    - (1 - treatment) * outcome / p_control
)

df["ipw_contribution"] = contribution

# ==========================================
# 3. CUMULATIVE UPLIFT CURVE
# ==========================================

cumulative_effect = np.cumsum(contribution)

x = np.arange(1, n + 1) / n

# Average incremental outcome per customer targeted
uplift_curve = cumulative_effect / n

# Random targeting baseline
random_curve = x * uplift_curve[-1]

# ==========================================
# 4. AUUC
# ==========================================

model_auuc = np.trapezoid(
    uplift_curve,
    x
)

random_auuc = np.trapezoid(
    random_curve,
    x
)

qini_coefficient = model_auuc - random_auuc

# ==========================================
# 5. RESULTS
# ==========================================

print("CORRECTED UPLIFT EVALUATION")
print("--------------------------------")

print("Test customers:", n)
print(f"Treatment probability: {p_treatment:.4f}")
print(f"Control probability: {p_control:.4f}")

print(f"\nModel AUUC: {model_auuc:.8f}")
print(f"Random AUUC: {random_auuc:.8f}")
print(f"Qini coefficient: {qini_coefficient:.8f}")

print(
    "\nFinal estimated treatment effect per customer:",
    round(uplift_curve[-1], 8)
)

# ==========================================
# 6. PLOT
# ==========================================

plt.figure(figsize=(9, 6))

plt.plot(
    x,
    uplift_curve,
    label="T-Learner"
)

plt.plot(
    x,
    random_curve,
    linestyle="--",
    label="Random Targeting"
)

plt.xlabel("Fraction of Customers Targeted")
plt.ylabel("Estimated Incremental Conversions per Customer")
plt.title("Corrected Uplift Curve")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

output_path = BASE_DIR / "models" / "uplift_curve_corrected.png"

plt.savefig(output_path, dpi=300)
plt.close()

print("\nSaved:", output_path)
print("Corrected evaluation completed.")
