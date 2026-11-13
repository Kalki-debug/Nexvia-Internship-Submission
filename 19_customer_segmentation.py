
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================
# 1. LOAD DATA
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data" / "telco_clv_results.csv"
)

print("Dataset shape:", df.shape)

# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "tenure",
    "MonthlyCharges",
    "average_monthly_revenue"
]

X = df[features].copy()

X = X.fillna(X.median())

# ==========================================
# 3. STANDARDIZE FEATURES
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature standardization completed.")

# ==========================================
# 4. ELBOW METHOD
# ==========================================

inertia_values = []

for k in range(2, 8):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia_values.append(model.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 8),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    BASE_DIR / "models" / "elbow_method.png",
    dpi=300
)

plt.close()

# ==========================================
# 5. TRAIN K-MEANS
# ==========================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["customer_cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means clustering completed.")

# ==========================================
# 6. EVALUATE CLUSTERS
# ==========================================

silhouette = silhouette_score(
    X_scaled,
    df["customer_cluster"]
)

print("\nSilhouette Score:", round(silhouette, 4))

print("\nCluster Distribution:")
print(df["customer_cluster"].value_counts().sort_index())

# ==========================================
# 7. CLUSTER PROFILING
# ==========================================

profile = df.groupby("customer_cluster").agg(
    customer_count=("customer_cluster", "count"),
    average_tenure=("tenure", "mean"),
    average_monthly_charges=("MonthlyCharges", "mean"),
    average_monthly_revenue=("average_monthly_revenue", "mean"),
    average_historical_clv=("historical_clv", "mean")
).round(2)

print("\nCUSTOMER CLUSTER PROFILE")
print("--------------------------------")
print(profile)

# Churn rate by cluster
if "Churn" in df.columns:

    churn_rate = df.groupby(
        "customer_cluster"
    )["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    )

    profile["churn_rate_percent"] = churn_rate.round(2)

    print("\nChurn Rate by Cluster:")
    print(churn_rate.round(2))

# ==========================================
# 8. SAVE RESULTS
# ==========================================

output_path = (
    BASE_DIR / "data" / "customer_segmentation_results.csv"
)

profile_path = (
    BASE_DIR / "data" / "customer_cluster_profiles.csv"
)

df.to_csv(output_path, index=False)
profile.to_csv(profile_path)

print("\nSaved:", output_path)
print("Saved:", profile_path)

# ==========================================
# 9. VISUALIZATION
# ==========================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="tenure",
    y="MonthlyCharges",
    hue="customer_cluster",
    palette="viridis",
    alpha=0.6
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Tenure (Months)")
plt.ylabel("Monthly Charges")
plt.tight_layout()

plt.savefig(
    BASE_DIR / "models" / "customer_clusters.png",
    dpi=300
)

plt.close()

print("Cluster chart saved.")

print("\nCustomer segmentation completed.")
