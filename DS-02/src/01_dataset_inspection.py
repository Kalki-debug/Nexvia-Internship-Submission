
import pandas as pd
import os

# Dataset directory
DATA_PATH = "data/raw"

# M5 dataset files
files = [
    "calendar.csv",
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "sell_prices.csv",
    "sample_submission.csv"
]

print("=" * 60)
print("M5 RETAIL DEMAND FORECASTING - DATASET INSPECTION")
print("=" * 60)

for file in files:
    file_path = os.path.join(DATA_PATH, file)

    print(f"\nDataset: {file}")

    df = pd.read_csv(file_path)

    print("Shape:", df.shape)
    print("Columns:", len(df.columns))
    print("First 5 rows:")
    print(df.head())

    print("-" * 60)
