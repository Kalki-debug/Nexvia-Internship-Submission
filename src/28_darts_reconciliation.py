
import pandas as pd
import numpy as np

from darts import TimeSeries
from darts.dataprocessing.transformers import BottomUpReconciliator

# 1. Load forecasts
df = pd.read_csv(
    "reports/baseline/lightgbm_recursive_predictions_CA_1_100.csv"
)

# 2. Convert forecasts into a wide format
wide = df.pivot(
    index="day_number",
    columns="item_id",
    values="predicted_sales"
).sort_index()

product_columns = wide.columns.tolist()

# 3. Create CA_1 aggregate component
wide["CA_1"] = wide[product_columns].sum(axis=1)

wide = wide.reset_index()

# 4. Create Darts TimeSeries
series = TimeSeries.from_dataframe(
    wide,
    time_col="day_number",
    value_cols=product_columns + ["CA_1"],
    freq=1
)

# 5. Define hierarchy: each product belongs to CA_1
hierarchy = {
    item: ["CA_1"] for item in product_columns
}

series = series.with_hierarchy(hierarchy)

print("Hierarchy attached:", series.has_hierarchy)
print("Number of components:", len(series.components))

# 6. Apply Bottom-Up reconciliation
reconciler = BottomUpReconciliator()
reconciled = reconciler.transform(series)

# 7. Export results
result = reconciled.to_dataframe()

result.to_csv(
    "reports/baseline/darts_reconciled_forecasts.csv"
)

# 8. Validate aggregate
if "CA_1" not in result.columns:
    raise ValueError("CA_1 aggregate missing after reconciliation.")

calculated_total = result[product_columns].sum(axis=1)

is_valid = np.allclose(
    calculated_total,
    result["CA_1"],
    atol=1e-6
)

print("\nBottom-Up reconciliation completed!")
print("Output shape:", result.shape)
print("Aggregate validation:", is_valid)
print("\nForecast preview:")
print(result.head())

if not is_valid:
    raise ValueError("Aggregate validation failed.")
