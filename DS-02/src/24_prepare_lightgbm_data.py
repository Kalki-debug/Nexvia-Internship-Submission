
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

INPUT_FILE = "data/processed/modeling_CA_1.csv"
OUTPUT_DIR = Path("data/processed/lightgbm")

TRAIN_END = 1885
VALIDATION_END = 1913

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("Loading CA_1 modeling dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# --------------------------------------------------
# 3. Convert day column into numeric format
# --------------------------------------------------

df["day_number"] = df["d"].str.replace("d_", "", regex=False).astype(int)

# --------------------------------------------------
# 4. Select categorical features
# --------------------------------------------------

categorical_features = [
    "item_id",
    "dept_id",
    "cat_id"
]

for col in categorical_features:
    df[col] = df[col].astype("category")

# --------------------------------------------------
# 5. Select numerical features
# --------------------------------------------------

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

target_column = "sales"

# --------------------------------------------------
# 6. Remove rows without sufficient lag history
# --------------------------------------------------

df = df[df["day_number"] >= 29].copy()

# --------------------------------------------------
# 7. Training and validation split
# --------------------------------------------------

train_df = df[df["day_number"] <= TRAIN_END].copy()

validation_df = df[
    (df["day_number"] > TRAIN_END) &
    (df["day_number"] <= VALIDATION_END)
].copy()

X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_validation = validation_df[feature_columns]
y_validation = validation_df[target_column]

# --------------------------------------------------
# 8. Print results
# --------------------------------------------------

print("\nTRAINING DATA")
print("Rows:", len(X_train))
print("Features:", len(feature_columns))

print("\nVALIDATION DATA")
print("Rows:", len(X_validation))

print("\nTraining day range:",
      train_df["day_number"].min(),
      "to",
      train_df["day_number"].max())

print("\nValidation day range:",
      validation_df["day_number"].min(),
      "to",
      validation_df["day_number"].max())

print("\nMissing values in training features:")
print(X_train.isnull().sum())

# --------------------------------------------------
# 9. Save prepared datasets
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_pickle(OUTPUT_DIR / "train_CA_1.pkl")
validation_df.to_pickle(OUTPUT_DIR / "validation_CA_1.pkl")

print("\nPrepared datasets saved successfully!")
print("Output directory:", OUTPUT_DIR)
