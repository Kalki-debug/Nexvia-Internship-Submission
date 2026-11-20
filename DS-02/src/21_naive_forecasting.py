
import pandas as pd
import numpy as np
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error

print("=" * 65)
print("BASELINE FORECASTING - NAIVE VS SEASONAL NAIVE")
print("=" * 65)

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

STORE = "CA_1"
SAMPLE_SIZE = None

TRAIN_END = 1885
FORECAST_HORIZON = 28

os.makedirs("reports/baseline", exist_ok=True)

# --------------------------------------------------
# 2. LOAD SALES DATA
# --------------------------------------------------

print("\nLoading sales data...")

sales = pd.read_csv(
    "data/raw/sales_train_validation.csv"
)

store_sales = sales[
    sales["store_id"] == STORE
].copy()

if SAMPLE_SIZE is not None:
    store_sales = store_sales.head(SAMPLE_SIZE)

day_columns = [
    f"d_{i}" for i in range(1, 1914)
]

print("Store:", STORE)
print("Series selected:", len(store_sales))

# --------------------------------------------------
# 3. TRAINING AND VALIDATION SPLIT
# --------------------------------------------------

train_columns = [
    f"d_{i}" for i in range(1, TRAIN_END + 1)
]

validation_columns = [
    f"d_{i}"
    for i in range(
        TRAIN_END + 1,
        TRAIN_END + FORECAST_HORIZON + 1
    )
]

train_data = store_sales[train_columns].to_numpy()
actual_data = store_sales[validation_columns].to_numpy()

print("Training days:", len(train_columns))
print("Validation days:", len(validation_columns))

# --------------------------------------------------
# 4. NAIVE FORECAST
# --------------------------------------------------

print("\nGenerating Naive Forecast...")

last_sales = train_data[:, -1]

naive_predictions = np.repeat(
    last_sales.reshape(-1, 1),
    FORECAST_HORIZON,
    axis=1
)

# --------------------------------------------------
# 5. SEASONAL NAIVE FORECAST
# --------------------------------------------------

print("Generating Seasonal Naive Forecast...")

seasonal_predictions = np.tile(
    train_data[:, -7:],
    (1, 4)
)

# --------------------------------------------------
# 6. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(actual, predicted, model_name):

    actual_flat = actual.flatten()
    predicted_flat = predicted.flatten()

    mae = mean_absolute_error(
        actual_flat,
        predicted_flat
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_flat,
            predicted_flat
        )
    )

    total_actual = np.sum(actual_flat)
    total_error = np.sum(
        np.abs(actual_flat - predicted_flat)
    )

    wape = (
        total_error / total_actual * 100
        if total_actual != 0
        else np.nan
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "WAPE (%)": wape
    }

# --------------------------------------------------
# 7. COMPARE MODELS
# --------------------------------------------------

print("\nEvaluating models...")

results = [
    evaluate_model(
        actual_data,
        naive_predictions,
        "Naive Forecast"
    ),
    evaluate_model(
        actual_data,
        seasonal_predictions,
        "Seasonal Naive Forecast"
    )
]

results_df = pd.DataFrame(results)

print("\nBASELINE RESULTS")
print("=" * 65)
print(results_df.to_string(index=False))

# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------

results_df.to_csv(
    "reports/baseline/baseline_metrics_CA_1.csv",
    index=False
)

# Save sample predictions for inspection
prediction_rows = []

for index, row in enumerate(store_sales.itertuples(index=False)):

    series_id = store_sales.iloc[index]["id"]

    for day in range(FORECAST_HORIZON):

        prediction_rows.append({
            "id": series_id,
            "forecast_day": day + 1,
            "actual": actual_data[index, day],
            "naive_prediction": naive_predictions[index, day],
            "seasonal_prediction": seasonal_predictions[index, day]
        })

predictions_df = pd.DataFrame(prediction_rows)

predictions_df.to_csv(
    "reports/baseline/baseline_predictions_CA_1.csv",
    index=False
)

print("\nSaved metrics and predictions.")
print("\nBASELINE FORECASTING COMPLETED!")
