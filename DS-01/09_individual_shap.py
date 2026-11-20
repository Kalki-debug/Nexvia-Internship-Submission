
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

# 1. Load the saved model pipeline
model = joblib.load("models/lightgbm_churn_pipeline.joblib")

# 2. Load prepared customer data
df = pd.read_csv("data/telco_ml_prepared.csv")

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 3. Re-create the same test split used during training
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Find the customer with the highest predicted churn probability
probabilities = model.predict_proba(X_test)[:, 1]
customer_position = int(np.argmax(probabilities))

customer = X_test.iloc[[customer_position]]
actual_churn = int(y_test.iloc[customer_position])
churn_probability = float(probabilities[customer_position])

# 5. Transform using the saved, fitted preprocessor
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

encoded_customer = preprocessor.transform(customer)
feature_names = preprocessor.get_feature_names_out()

# SHAP works with a dense row for the individual waterfall plot
if hasattr(encoded_customer, "toarray"):
    encoded_customer = encoded_customer.toarray()

# 6. Calculate SHAP values
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(encoded_customer)

# Support different SHAP output formats
if isinstance(shap_values, list):
    shap_row = np.asarray(shap_values[1])[0]
else:
    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_row = shap_values[0, :, 1]
    else:
        shap_row = shap_values[0]

# Get the baseline output for the churn class
base_values = np.asarray(explainer.expected_value)

if base_values.ndim == 0:
    base_value = float(base_values)
elif base_values.size == 1:
    base_value = float(base_values.reshape(-1)[0])
else:
    base_value = float(base_values.reshape(-1)[-1])

# 7. Print customer summary
print("Individual customer SHAP analysis")
print("---------------------------------")
print("Test row position:", customer_position)
print("Actual churn label:", actual_churn)
print(f"Predicted churn probability: {churn_probability:.2%}")

if actual_churn == 1:
    print("Actual outcome: Customer churned")
else:
    print("Actual outcome: Customer did not churn")

# 8. Show the strongest feature contributions
contributions = pd.DataFrame({
    "feature": feature_names,
    "shap_value": shap_row
})

contributions["absolute_impact"] = contributions["shap_value"].abs()
contributions = contributions.sort_values(
    "absolute_impact", ascending=False
)

print("\nTop 10 feature contributions:")
print(
    contributions[["feature", "shap_value"]]
    .head(10)
    .to_string(index=False)
)

# 9. Save a waterfall plot
output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

explanation = shap.Explanation(
    values=shap_row,
    base_values=base_value,
    data=encoded_customer[0],
    feature_names=feature_names
)

shap.plots.waterfall(explanation, max_display=12, show=False)
plt.savefig(
    output_dir / "individual_shap_waterfall.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("\nSaved plot to models/individual_shap_waterfall.png")
