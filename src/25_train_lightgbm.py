
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

# --------------------------------------------------
# 1. Load prepared training data
# --------------------------------------------------

print("Loading training dataset...")

train_df = pd.read_pickle(
    "data/processed/lightgbm/train_CA_1.pkl"
)

print("Training shape:", train_df.shape)

# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

categorical_features = [
    "item_id",
    "dept_id",
    "cat_id"
]

numerical_features = [
    "wm_yr_wk",
    "wday",
    "month",
    "year",
    "snap_CA",
    "day_of_week",
    "is_weekend",
    "day_of_month",
    "quarter",
    "is_event",
    "sell_price",
    "previous_price",
    "price_change",
    "price_change_pct",
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]

feature_columns = categorical_features + numerical_features

X_train = train_df[feature_columns]
y_train = train_df["sales"]

# --------------------------------------------------
# 3. Create LightGBM model
# --------------------------------------------------

print("\nCreating LightGBM model...")

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

# --------------------------------------------------
# 4. Train model
# --------------------------------------------------

print("\nTraining started...")

model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_features
)

print("Training completed!")

# --------------------------------------------------
# 5. Training performance
# --------------------------------------------------

print("\nCalculating training metrics...")

train_predictions = model.predict(X_train)
train_predictions = np.maximum(train_predictions, 0)

mae = mean_absolute_error(y_train, train_predictions)

rmse = np.sqrt(
    mean_squared_error(y_train, train_predictions)
)

wape = (
    np.sum(np.abs(y_train - train_predictions))
    / np.sum(y_train)
) * 100

print("\nTRAINING PERFORMANCE")
print("--------------------")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%")

# --------------------------------------------------
# 6. Save trained model
# --------------------------------------------------

model_dir = Path("models/lightgbm")
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "lightgbm_CA_1.txt"

model.booster_.save_model(str(model_path))

print("\nModel saved successfully!")
print("Location:", model_path)

# Save feature names for later use
joblib.dump(
    feature_columns,
    model_dir / "feature_columns.pkl"
)

print("Feature list saved successfully!")
