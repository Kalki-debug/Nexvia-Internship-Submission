
import pandas as pd

# Load the prepared dataset
df = pd.read_csv("data/telco_ml_prepared.csv")

# Create a numeric target variable
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Separate target from input features
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts().sort_index())

print("\nFeature columns:")
print(X.columns.tolist())

print("\nRemaining missing values in features:", X.isnull().sum().sum())
print("Missing values in target:", y.isnull().sum())
