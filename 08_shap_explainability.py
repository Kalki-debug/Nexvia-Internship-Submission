
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the saved LightGBM pipeline
model = joblib.load("models/lightgbm_churn_pipeline.joblib")

# Load prepared customer data
df = pd.read_csv("data/telco_ml_prepared.csv")

# Separate features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Use the same split as model training
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Get the fitted preprocessing step and trained LightGBM model
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Transform test features using the already-fitted preprocessor
X_test_encoded = preprocessor.transform(X_test)

# Get readable feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out()

# Create SHAP explainer
explainer = shap.TreeExplainer(classifier)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_encoded)

# Handle SHAP output formats for binary classification
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Plot global feature importance
shap.summary_plot(
    shap_values,
    X_test_encoded,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()
plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print("SHAP analysis completed.")
print("Saved summary plot to models/shap_summary.png")
