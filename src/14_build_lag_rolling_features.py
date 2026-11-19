
import pandas as pd
import numpy as np
import os

print("=" * 65)
print("FULL-SCALE LAG AND ROLLING FEATURE ENGINEERING")
print("=" * 65)

os.makedirs("data/processed", exist_ok=True)

stores = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3"
]

metadata_columns = [
    "id", "item_id", "dept_id",
    "cat_id", "store_id", "state_id"
]

day_columns = [f"d_{i}" for i in range(1, 1914)]

for store in stores:

    print(f"\nProcessing store: {store}")

    # Load original sales data
    sales = pd.read_csv("data/raw/sales_train_validation.csv")

    # Select one store
    store_sales = sales[
        sales["store_id"] == store
    ].reset_index(drop=True)

    # Extract daily sales as a NumPy array
    sales_matrix = store_sales[day_columns].to_numpy(
        dtype=np.float32
    )

    output_path = (
        f"data/processed/lag_features_{store}.csv"
    )

    # Start a fresh output file for this store
    if os.path.exists(output_path):
        os.remove(output_path)

    batch_size = 50
    total_rows = 0

    for start in range(0, len(store_sales), batch_size):

        end = min(start + batch_size, len(store_sales))

        batch_features = []

        for index in range(start, end):

            values = sales_matrix[index]

            series = pd.Series(values)

            features = pd.DataFrame({
                "d": day_columns,
                "lag_1": series.shift(1),
                "lag_7": series.shift(7),
                "lag_28": series.shift(28),
                "rolling_mean_7": (
                    series.shift(1)
                    .rolling(window=7)
                    .mean()
                ),
                "rolling_mean_28": (
                    series.shift(1)
                    .rolling(window=28)
                    .mean()
                )
            })

            # Add product and store information
            for column in metadata_columns:
                features[column] = store_sales.iloc[index][column]

            batch_features.append(features)

        # Combine this batch and save it
        batch_df = pd.concat(
            batch_features,
            ignore_index=True
        )

        batch_df.to_csv(
            output_path,
            mode="a",
            header=(start == 0),
            index=False
        )

        total_rows += len(batch_df)

        print(
            f"Processed products: {end}/{len(store_sales)}"
        )

        del batch_features, batch_df

    print("Saved:", output_path)
    print("Feature rows:", f"{total_rows:,}")

    del sales, store_sales, sales_matrix

print("\n" + "=" * 65)
print("FULL-SCALE FEATURE ENGINEERING COMPLETED!")
print("=" * 65)
