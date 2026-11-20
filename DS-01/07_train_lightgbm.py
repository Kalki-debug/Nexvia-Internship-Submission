
import pandas as pd
import joblib
from pathlib import Path

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Load prepared dataset
df = pd.read_csv("data/telco_ml_prepared.csv")

# Separate target and features
y = df["Churn"].map({"No": 0, "Yes": 1})
X = df.drop(columns=["customerID", "Churn"])

# Identify categorical and numerical features
categorical_cols = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Use the same split settings as the baseline
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Encode categorical columns; keep numeric columns unchanged
preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_cols
    ),
    ("numerical", "passthrough", numerical_cols)
])

# Build LightGBM pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
     ("classifier", LGBMClassifier(
     n_estimators=300,
     learning_rate=0.05,
     num_leaves=31,
     class_weight="balanced",
     random_state=42,
     verbosity=-1
    ))

])

# Train the model
model.fit(X_train, y_train)

# Save the fitted model pipeline
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "lightgbm_churn_pipeline.joblib"
joblib.dump(model, model_path)
print(f"\nSaved model pipeline to: {model_path}")

# Predict on held-out test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(
    y_test, y_pred,
    target_names=["No churn", "Churn"]
))
