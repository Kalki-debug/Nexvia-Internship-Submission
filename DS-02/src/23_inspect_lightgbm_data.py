
import pandas as pd

# Load a small sample
file_path = "data/processed/modeling_CA_1.csv"

df = pd.read_csv(file_path, nrows=10)

print("\nDataset columns:")
print(df.columns.tolist())

print("\nDataset shape (sample):")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())
