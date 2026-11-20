
import os
import numpy as np
import pandas as pd
import torch

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

STORE = "CA_1"
SAMPLE_SIZE = 100

TRAIN_END = 1885
FORECAST_HORIZON = 28
ENCODER_LENGTH = 56

BATCH_SIZE = 128
MAX_EPOCHS = 5

INPUT_FILE = "data/processed/modeling_CA_1.csv"

OUTPUT_DIR = "reports/baseline"
MODEL_DIR = "models/tft"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

seed_everything(42, workers=True)


# --------------------------------------------------
# 2. LOAD A SMALL DATASET
# --------------------------------------------------

print("Loading CA_1 data...")

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

print("Loaded shape:", df.shape)

# Convert day labels into numeric time index
df["time_idx"] = (
    df["d"].str.replace("d_", "", regex=False).astype(int)
)

# Keep the same 100 product series used in earlier comparisons
df = df.sort_values(["id", "time_idx"]).reset_index(drop=True)

# Product and calendar categories
categorical_columns = [
    "id", "item_id", "dept_id", "cat_id",
    "weekday", "event_type_1"
]

for col in categorical_columns:
    df[col] = df[col].fillna("Unknown").astype(str)

# Price availability indicator
df["price_available"] = df["sell_price"].notna().astype(int)

# Fill missing prices within each product series
df["sell_price"] = (
    df.groupby("id")["sell_price"]
    .transform(lambda x: x.ffill().bfill())
)

df["sell_price"] = df["sell_price"].fillna(
    df["sell_price"].median()
)

# Numeric columns
numeric_columns = [
    "sales", "month", "day_of_week",
    "is_weekend", "is_event", "snap_CA",
    "sell_price", "price_available"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[numeric_columns] = df[numeric_columns].fillna(0)

df["sales"] = df["sales"].clip(lower=0).astype("float32")

print("Unique product series:", df["id"].nunique())
print("Date range:", df["time_idx"].min(), "-", df["time_idx"].max())


# --------------------------------------------------
# 3. TRAINING DATASET
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

train_loader = training_dataset.to_dataloader(
    train=True,
    batch_size=BATCH_SIZE,
    num_workers=0
)

print("Training sequences:", len(training_dataset))


# --------------------------------------------------
# 4. VALIDATION DATASET
# --------------------------------------------------

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
# 5. BUILD TFT MODEL
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

print("TFT parameters:", tft.size())


# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=2,
    mode="min"
)

trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="cpu",
    devices=1,

    gradient_clip_val=0.1,
    callbacks=[early_stop],

    limit_train_batches=200,
    enable_model_summary=True,
    logger=False,
    enable_checkpointing=False
)

print("\nStarting TFT training...")

trainer.fit(
    tft,
    train_dataloaders=train_loader,
    val_dataloaders=validation_loader
)



# --------------------------------------------------
# 7. PREDICTIONS
# --------------------------------------------------

print("\nSaving trained TFT model...")

torch.save(
    tft.state_dict(),
    f"{MODEL_DIR}/tft_CA_1_100.pth"
)

print("Generating validation predictions...")

predictions = tft.predict(
    validation_loader,
    mode="prediction",
    trainer_kwargs={
        "accelerator": "cpu",
        "devices": 1
    }
)

predictions = predictions.detach().cpu().numpy().reshape(-1)
predictions = np.maximum(predictions, 0)

# Actual validation sales in matching product/date order
actuals = (
    df[df["time_idx"] > TRAIN_END]
    .sort_values(["id", "time_idx"])["sales"]
    .to_numpy()
)

# Evaluation metrics
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))

wape = (
    np.sum(np.abs(actuals - predictions))
    / np.sum(actuals)
) * 100

print("\n========== TFT RESULTS ==========")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%")

metrics = pd.DataFrame([{
    "Model": "TFT",
    "Store": STORE,
    "Products": SAMPLE_SIZE,
    "MAE": mae,
    "RMSE": rmse,
    "WAPE": wape
}])

metrics.to_csv(
    f"{OUTPUT_DIR}/tft_metrics_CA_1_100.csv",
    index=False
)

print("\nTFT model and metrics saved successfully.")


