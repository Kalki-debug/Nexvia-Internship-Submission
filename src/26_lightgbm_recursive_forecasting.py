
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

STORE = "CA_1"
SAMPLE_SIZE = 100
TRAIN_END = 1885
VALIDATION_END = 1913

SALES_FILE = "data/raw/sales_train_validation.csv"
VALIDATION_FILE = "data/processed/lightgbm/validation_CA_1.pkl"
MODEL_FILE = "models/lightgbm/lightgbm_CA_1.txt"
FEATURE_FILE = "models/lightgbm/feature_columns.pkl"

CATEGORICAL_FEATURES = ["item_id", "dept_id", "cat_id"]

OUTPUT_DIR = Path("reports/baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Starting recursive LightGBM forecasting...")

# --------------------------------------------------
# 2. Load model and features
# --------------------------------------------------

model = lgb.Booster(model_file=MODEL_FILE)
feature_columns = joblib.load(FEATURE_FILE)

print("Model loaded successfully.")

# Recover the category definitions saved by LightGBM
category_definitions = model.pandas_categorical

if category_definitions is None:
    raise ValueError("Model does not contain categorical definitions.")

print("Categorical definitions loaded.")

# --------------------------------------------------
# 3. Load historical sales
# --------------------------------------------------

print("\nLoading historical sales...")

sales = pd.read_csv(SALES_FILE)

store_sales = sales[sales["store_id"] == STORE].copy()

selected_products = store_sales.head(SAMPLE_SIZE).copy()

day_cols = [
    col for col in selected_products.columns
    if col.startswith("d_")
]

print("Selected products:", len(selected_products))

# --------------------------------------------------
# 4. Load validation data
# --------------------------------------------------

validation_df = pd.read_pickle(VALIDATION_FILE)

selected_ids = selected_products["id"].tolist()

validation_df = validation_df[
    validation_df["id"].isin(selected_ids)
].copy()

validation_df = validation_df.sort_values(
    ["id", "day_number"]
)

# --------------------------------------------------
# 5. Recursive forecasting
# --------------------------------------------------

all_predictions = []

numerical_features = [
    col for col in feature_columns
    if col not in CATEGORICAL_FEATURES
]

for product_number, (_, product) in enumerate(
    selected_products.iterrows(), start=1
):

    product_id = product["id"]

    # Actual historical sales only through d_1885
    history = product[day_cols[:TRAIN_END]].to_numpy(
        dtype=float
    ).tolist()

    future = validation_df[
        validation_df["id"] == product_id
    ].sort_values("day_number").copy()

    if len(future) != VALIDATION_END - TRAIN_END:
        raise ValueError(
            f"{product_id}: expected 28 validation rows, "
            f"found {len(future)}"
        )

    product_predictions = []

    for _, future_row in future.iterrows():

        # Build one prediction row
        X_row = pd.DataFrame([
            future_row[feature_columns].to_dict()
        ])

        # Restore training categorical definitions
        for i, col in enumerate(CATEGORICAL_FEATURES):
            X_row[col] = pd.Categorical(
                X_row[col],
                categories=category_definitions[i]
            )

        # Ensure numerical columns are numeric
        for col in numerical_features:
            X_row[col] = pd.to_numeric(
                X_row[col],
                errors="coerce"
            )

        # Update lag and rolling features using
        # historical actuals and previous predictions
        X_row.loc[:, "lag_1"] = history[-1]
        X_row.loc[:, "lag_7"] = history[-7]
        X_row.loc[:, "lag_28"] = history[-28]

        X_row.loc[:, "rolling_mean_7"] = np.mean(
            history[-7:]
        )

        X_row.loc[:, "rolling_mean_28"] = np.mean(
            history[-28:]
        )

        # Predict next day
        prediction = float(model.predict(X_row)[0])

        prediction = max(0.0, prediction)

        product_predictions.append(prediction)

        # Add prediction to history for the next day
        history.append(prediction)

    # Save actual and predicted sales separately
    for i, (_, future_row) in enumerate(future.iterrows()):

        all_predictions.append({
            "id": product_id,
            "item_id": product["item_id"],
            "day_number": future_row["day_number"],
            "actual_sales": future_row["sales"],
            "predicted_sales": product_predictions[i]
        })

    if product_number % 10 == 0:
        print(
            f"Forecasted {product_number}/{SAMPLE_SIZE} products"
        )

# --------------------------------------------------
# 6. Calculate validation metrics
# --------------------------------------------------

results = pd.DataFrame(all_predictions)

if len(results) != SAMPLE_SIZE * 28:
    raise ValueError(
        f"Unexpected prediction count: {len(results)}"
    )

actual = results["actual_sales"].to_numpy()
predicted = results["predicted_sales"].to_numpy()

mae = mean_absolute_error(actual, predicted)

rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

total_actual = np.sum(actual)

wape = (
    np.sum(np.abs(actual - predicted)) / total_actual
) * 100 if total_actual > 0 else np.nan

print("\nLIGHTGBM RECURSIVE VALIDATION RESULTS")
print("------------------------------------")
print(f"Products: {SAMPLE_SIZE}")
print("Forecast days: 28")
print(f"Forecast records: {len(results)}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%")

# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

prediction_file = (
    OUTPUT_DIR / "lightgbm_recursive_predictions_CA_1_100.csv"
)

metrics_file = (
    OUTPUT_DIR / "lightgbm_recursive_metrics_CA_1_100.csv"
)

results.to_csv(prediction_file, index=False)

metrics = pd.DataFrame([{
    "Model": "LightGBM Recursive",
    "Products": SAMPLE_SIZE,
    "Forecast_Days": 28,
    "MAE": mae,
    "RMSE": rmse,
    "WAPE": wape
}])

metrics.to_csv(metrics_file, index=False)

print("\nFiles saved successfully!")
print(prediction_file)
print(metrics_file)
