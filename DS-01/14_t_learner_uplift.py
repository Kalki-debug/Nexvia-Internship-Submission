
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

# ==========================================
# 1. LOAD DATA
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data" / "hillstrom_uplift_prepared.csv"
)

print("Dataset shape:", df.shape)

# ==========================================
# 2. DEFINE FEATURES AND TARGET
# ==========================================

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

X = df[feature_columns]
treatment = df["treatment"]
y = df["conversion"]

# ==========================================
# 3. SPLIT DATA
# ==========================================

X_train, X_test, t_train, t_test, y_train, y_test = (
    train_test_split(
        X,
        treatment,
        y,
        test_size=0.20,
        random_state=42,
        stratify=treatment
    )
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# ==========================================
# 4. PREPROCESSING
# ==========================================

categorical_features = [
    "history_segment",
    "zip_code",
    "channel"
]

numeric_features = [
    "recency",
    "history",
    "mens",
    "womens",
    "newbie"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ==========================================
# 5. CONTROL MODEL
# ==========================================

control_train = t_train == 0
control_test = t_test == 0

control_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

control_model.fit(
    X_train[control_train],
    y_train[control_train]
)

print("\nControl model trained.")

# ==========================================
# 6. TREATMENT MODEL
# ==========================================

treatment_train = t_train == 1
treatment_test = t_test == 1

treatment_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

treatment_model.fit(
    X_train[treatment_train],
    y_train[treatment_train]
)

print("Treatment model trained.")

# ==========================================
# 7. PREDICT CUSTOMER PROBABILITIES
# ==========================================

p_control = control_model.predict_proba(
    X_test
)[:, 1]

p_treatment = treatment_model.predict_proba(
    X_test
)[:, 1]

# Individual treatment effect estimate
uplift = p_treatment - p_control

# ==========================================
# 8. CREATE UPLIFT RESULTS
# ==========================================

results = X_test.copy()

results["actual_treatment"] = t_test.values
results["actual_conversion"] = y_test.values

results["p_control"] = p_control
results["p_treatment"] = p_treatment
results["uplift_score"] = uplift

results["customer_group"] = np.select(
    [
        results["uplift_score"] > 0.01,
        results["uplift_score"] < -0.01
    ],
    [
        "Persuadable",
        "Do Not Target"
    ],
    default="Low Uplift"
)

# ==========================================
# 9. MODEL EVALUATION
# ==========================================

control_auc = roc_auc_score(
    y_test[control_test],
    p_control[control_test.values]
)

treatment_auc = roc_auc_score(
    y_test[treatment_test],
    p_treatment[treatment_test.values]
)

print("\nMODEL PERFORMANCE")
print("--------------------------------")
print(f"Control model ROC-AUC: {control_auc:.4f}")
print(f"Treatment model ROC-AUC: {treatment_auc:.4f}")

# ==========================================
# 10. UPLIFT SUMMARY
# ==========================================

print("\nUPLIFT SUMMARY")
print("--------------------------------")

print("Average control probability:",
      round(p_control.mean(), 4))

print("Average treatment probability:",
      round(p_treatment.mean(), 4))

print("Average predicted uplift:",
      round(uplift.mean(), 4))

print("\nCustomer group distribution:")
print(results["customer_group"].value_counts())

# ==========================================
# 11. SAVE RESULTS AND MODELS
# ==========================================

models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

results.to_csv(
    BASE_DIR / "data" / "hillstrom_uplift_predictions.csv",
    index=False
)

joblib.dump(
    control_model,
    models_dir / "uplift_control_model.joblib"
)

joblib.dump(
    treatment_model,
    models_dir / "uplift_treatment_model.joblib"
)

print("\nSaved:")
print("data/hillstrom_uplift_predictions.csv")
print("models/uplift_control_model.joblib")
print("models/uplift_treatment_model.joblib")

print("\nT-Learner training completed.")
