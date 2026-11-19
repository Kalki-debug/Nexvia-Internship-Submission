
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

STORE = "CA_1"
SAMPLE_SIZE = 100
TRAIN_END = 1885
FORECAST_HORIZON = 28

OUTPUT_DIR = Path("reports/baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("Loading datasets...")

sales = pd.read_csv("data/raw/sales_train_validation.csv")
calendar = pd.read_csv("data/raw/calendar.csv")

store_sales = sales[
    sales["store_id"] == STORE
].head(SAMPLE_SIZE).copy()

metadata_cols = [
    "id", "item_id", "dept_id",
    "cat_id", "store_id", "state_id"
]

day_cols = [
    col for col in store_sales.columns
    if col.startswith("d_")
]

# Convert to long format
sales_long = store_sales.melt(
    id_vars=metadata_cols,
    value_vars=day_cols,
    var_name="d",
    value_name="y"
)

sales_long = sales_long.merge(
    calendar[["d", "date"]],
    on="d",
    how="left"
)

sales_long["date"] = pd.to_datetime(sales_long["date"])

# --------------------------------------------------
# 2. PREPARE METRIC STORAGE
# --------------------------------------------------

all_actuals = []
all_naive = []
all_seasonal = []
all_prophet = []

prediction_records = []

# --------------------------------------------------
# 3. LOOP THROUGH PRODUCTS
# --------------------------------------------------

print(f"\nStarting Prophet forecasting for {SAMPLE_SIZE} products...")

for index, product_id in enumerate(store_sales["id"], start=1):

    product_data = sales_long[
        sales_long["id"] == product_id
    ][["date", "y"]].copy()

    product_data = product_data.rename(
        columns={"date": "ds"}
    )

    product_data = product_data.sort_values("ds")

    train = product_data.iloc[:TRAIN_END].copy()
    validation = product_data.iloc[
        TRAIN_END:TRAIN_END + FORECAST_HORIZON
    ].copy()

    train_values = train["y"].values

    actuals = validation["y"].values

    # Naive forecast
    naive_forecast = np.repeat(
        train_values[-1],
        FORECAST_HORIZON
    )

    # Seasonal naive forecast
    seasonal_forecast = np.tile(
        train_values[-7:],
        4
    )

    # Prophet model
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(train)

    future = model.make_future_dataframe(
        periods=FORECAST_HORIZON,
        freq="D"
    )

    forecast = model.predict(future)

    prophet_forecast = forecast.tail(
        FORECAST_HORIZON
    )["yhat"].values

    # Demand cannot be negative
    prophet_forecast = np.maximum(
        prophet_forecast,
        0
    )

    # Store results
    all_actuals.extend(actuals)
    all_naive.extend(naive_forecast)
    all_seasonal.extend(seasonal_forecast)
    all_prophet.extend(prophet_forecast)

    for day in range(FORECAST_HORIZON):

        prediction_records.append({
            "id": product_id,
            "date": validation.iloc[day]["ds"],
            "actual": actuals[day],
            "naive": naive_forecast[day],
            "seasonal_naive": seasonal_forecast[day],
            "prophet": prophet_forecast[day]
        })

    if index % 10 == 0:
        print(f"Completed {index}/{SAMPLE_SIZE} products")

# --------------------------------------------------
# 4. CALCULATE METRICS
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    total_actual = np.sum(actual)

    wape = (
        np.sum(np.abs(actual - predicted))
        / total_actual
    ) * 100 if total_actual != 0 else np.nan

    return mae, rmse, wape


results = []

models = {
    "Naive": all_naive,
    "Seasonal Naive": all_seasonal,
    "Prophet": all_prophet
}

for model_name, predictions in models.items():

    mae, rmse, wape = calculate_metrics(
        np.array(all_actuals),
        np.array(predictions)
    )

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape
    })

metrics_df = pd.DataFrame(results)

# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

metrics_path = OUTPUT_DIR / "prophet_comparison_CA_1_100.csv"

predictions_path = OUTPUT_DIR / "prophet_predictions_CA_1_100.csv"

metrics_df.to_csv(metrics_path, index=False)

pd.DataFrame(prediction_records).to_csv(
    predictions_path,
    index=False
)

# --------------------------------------------------
# 6. DISPLAY RESULTS
# --------------------------------------------------

print("\n===================================")
print("100 PRODUCT FORECAST COMPARISON")
print("===================================")

print(metrics_df.to_string(index=False))

print("\nSaved metrics:", metrics_path)
print("Saved predictions:", predictions_path)

print("\nForecasting completed successfully!")
