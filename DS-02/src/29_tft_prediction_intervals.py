
import os
import numpy as np
import pandas as pd
import torch



from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

STORE = "CA_1"
SAMPLE_SIZE = 100

TRAIN_END = 1885
FORECAST_HORIZON = 28
ENCODER_LENGTH = 56
BATCH_SIZE = 128

INPUT_FILE = "data/processed/modeling_CA_1.csv"
MODEL_PATH = "models/tft/tft_CA_1_100.pth"

OUTPUT_DIR = "reports/baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Preparing TFT prediction intervals...")


# --------------------------------------------------
# 2. LOAD AND PREPARE DATA
# --------------------------------------------------

required_columns = [
    "id", "item_id", "dept_id", "cat_id",
    "d", "sales", "weekday", "month",
    "day_of_week", "is_weekend", "is_event",
    "event_type_1", "snap_CA", "sell_price"
]

df = pd.read_csv(
    INPUT_FILE,
    usecols=required_columns,
    nrows=SAMPLE_SIZE * 1913
)

df["time_idx"] = (
    df["d"].str.replace("d_", "", regex=False).astype(int)
)

df = df.sort_values(["id", "time_idx"]).reset_index(drop=True)

categorical_columns = [
    "id", "item_id", "dept_id", "cat_id",
    "weekday", "event_type_1"
]

for col in categorical_columns:
    df[col] = df[col].fillna("Unknown").astype(str)

df["price_available"] = df["sell_price"].notna().astype(int)

df["sell_price"] = (
    df.groupby("id")["sell_price"]
    .transform(lambda x: x.ffill().bfill())
)

df["sell_price"] = df["sell_price"].fillna(
    df["sell_price"].median()
)

numeric_columns = [
    "sales", "month", "day_of_week",
    "is_weekend", "is_event", "snap_CA",
    "sell_price", "price_available"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[numeric_columns] = df[numeric_columns].fillna(0)
df["sales"] = df["sales"].clip(lower=0).astype("float32")


# --------------------------------------------------
# 3. RECREATE TFT DATASET
# --------------------------------------------------

training_data = df[df["time_idx"] <= TRAIN_END].copy()

training_dataset = TimeSeriesDataSet(
    training_data,
    time_idx="time_idx",
    target="sales",
    group_ids=["id"],

    min_encoder_length=ENCODER_LENGTH,
    max_encoder_length=ENCODER_LENGTH,

    min_prediction_length=FORECAST_HORIZON,
    max_prediction_length=FORECAST_HORIZON,

    static_categoricals=[
        "item_id", "dept_id", "cat_id"
    ],

    time_varying_known_categoricals=[
        "weekday", "event_type_1"
    ],

    time_varying_known_reals=[
        "month", "day_of_week", "is_weekend",
        "is_event", "snap_CA", "sell_price",
        "price_available"
    ],

    time_varying_unknown_reals=["sales"],

    target_normalizer=GroupNormalizer(
        groups=["id"],
        transformation="softplus"
    ),

    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True
)

validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset,
    df,
    predict=True,
    stop_randomization=True
)

validation_loader = validation_dataset.to_dataloader(
    train=False,
    batch_size=BATCH_SIZE,
    num_workers=0
)

print("Validation sequences:", len(validation_dataset))


# --------------------------------------------------
# 4. RECREATE TFT MODEL AND LOAD WEIGHTS
# --------------------------------------------------

tft = TemporalFusionTransformer.from_dataset(
    training_dataset,

    learning_rate=0.03,
    hidden_size=8,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=4,

    loss=QuantileLoss(),
    optimizer="adam",

    reduce_on_plateau_patience=2
)

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

tft.load_state_dict(state_dict)
tft.eval()

print("Previously trained TFT weights loaded successfully.")


# --------------------------------------------------
# 5. GENERATE QUANTILE PREDICTIONS
# --------------------------------------------------

print("Generating quantile predictions...")

predictions = tft.predict(
    validation_loader,
    mode="quantiles",
    trainer_kwargs={
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False
    }
)
prediction_array = predictions.detach().cpu().numpy()

if prediction_array.ndim != 3:
    raise ValueError(
        f"Unexpected prediction shape: {prediction_array.shape}"
    )

available_quantiles = np.array(tft.loss.quantiles)

print("Prediction shape:", prediction_array.shape)
print("Available quantiles:", available_quantiles)


# --------------------------------------------------
# 6. SELECT P10, P50 AND P90
# --------------------------------------------------

def nearest_quantile_index(target):
    return int(np.argmin(np.abs(available_quantiles - target)))


p10_idx = nearest_quantile_index(0.10)
p50_idx = nearest_quantile_index(0.50)
p90_idx = nearest_quantile_index(0.90)

p10 = prediction_array[:, :, p10_idx].reshape(-1)
p50 = prediction_array[:, :, p50_idx].reshape(-1)
p90 = prediction_array[:, :, p90_idx].reshape(-1)

p10 = np.maximum(p10, 0)
p50 = np.maximum(p50, 0)
p90 = np.maximum(p90, 0)


# --------------------------------------------------
# 7. MATCH ACTUAL SALES
# --------------------------------------------------

validation_rows = (
    df[df["time_idx"] > TRAIN_END]
    .sort_values(["id", "time_idx"])
    .copy()
)

actuals = validation_rows["sales"].to_numpy()

if len(actuals) != len(p50):
    raise ValueError(
        f"Prediction/actual mismatch: {len(p50)} vs {len(actuals)}"
    )

lower = np.minimum(p10, p50)
upper = np.maximum(p90, p50)


# --------------------------------------------------
# 8. SAVE PREDICTION INTERVALS
# --------------------------------------------------

result = pd.DataFrame({
    "id": validation_rows["id"].to_numpy(),
    "item_id": validation_rows["item_id"].to_numpy(),
    "day_number": validation_rows["time_idx"].to_numpy(),
    "actual_sales": actuals,
    "P10_lower": lower,
    "P50_median": p50,
    "P90_upper": upper
})

output_path = (
    f"{OUTPUT_DIR}/tft_prediction_intervals_CA_1_100.csv"
)

result.to_csv(output_path, index=False)


# --------------------------------------------------
# 9. INTERVAL EVALUATION
# --------------------------------------------------

coverage = np.mean(
    (actuals >= lower) & (actuals <= upper)
) * 100

average_width = np.mean(upper - lower)

metrics = pd.DataFrame([{
    "Model": "TFT",
    "Store": STORE,
    "Products": SAMPLE_SIZE,
    "Forecast_Days": FORECAST_HORIZON,
    "Lower_Quantile": available_quantiles[p10_idx],
    "Upper_Quantile": available_quantiles[p90_idx],
    "Interval_Coverage_Percent": coverage,
    "Average_Interval_Width": average_width
}])

metrics.to_csv(
    f"{OUTPUT_DIR}/tft_interval_metrics_CA_1_100.csv",
    index=False
)

print("\n========== TFT PREDICTION INTERVAL RESULTS ==========")
print(f"Lower quantile: {available_quantiles[p10_idx]}")
print(f"Median quantile: {available_quantiles[p50_idx]}")
print(f"Upper quantile: {available_quantiles[p90_idx]}")
print(f"Interval coverage: {coverage:.2f}%")
print(f"Average interval width: {average_width:.4f}")

print("\nSaved files:")
print(output_path)
print(f"{OUTPUT_DIR}/tft_interval_metrics_CA_1_100.csv")

print("\nPrediction preview:")
print(result.head())
