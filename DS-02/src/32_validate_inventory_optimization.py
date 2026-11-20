
import os
import math
import pandas as pd

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DEMAND_FILE = (
    "data/processed/inventory/"
    "inventory_demand_CA_1_100.csv"
)

OPTIMIZED_FILE = (
    "reports/inventory/"
    "optimized_inventory_CA_1_100.csv"
)

OUTPUT_DIR = "reports/inventory"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "inventory_policy_comparison.csv"
)

HOLDING_COST = 1.0
STOCKOUT_COST = 5.0

SCENARIOS = {
    "low": (0.80, 0.20),
    "expected": (1.00, 0.60),
    "high": (1.20, 0.20)
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading inventory and optimization results...")

demand_df = pd.read_csv(DEMAND_FILE)
optimized_df = pd.read_csv(OPTIMIZED_FILE)

df = demand_df.merge(
    optimized_df[ 
    [
        "item_id",
        "optimized_order_quantity",
        "total_expected_cost"
    ]
    ],
    on="item_id",
    how="inner",
    validate="one_to_one"
)

if len(df) != len(demand_df):
    raise ValueError("Some products are missing from optimization results.")

# --------------------------------------------------
# 3. COST CALCULATION FUNCTION
# --------------------------------------------------

def calculate_expected_cost(order_quantity, forecast):

    holding_cost = 0
    stockout_cost = 0

    for _, (multiplier, probability) in SCENARIOS.items():

        scenario_demand = int(round(forecast * multiplier))

        leftover = max(order_quantity - scenario_demand, 0)
        shortage = max(scenario_demand - order_quantity, 0)

        holding_cost += (
            probability * leftover * HOLDING_COST
        )

        stockout_cost += (
            probability * shortage * STOCKOUT_COST
        )

    return holding_cost, stockout_cost


# --------------------------------------------------
# 4. COMPARE THREE INVENTORY POLICIES
# --------------------------------------------------

results = []

print("Comparing inventory policies...")

for _, row in df.iterrows():

    item_id = row["item_id"]
    forecast = float(row["total_forecast_demand"])

    forecast_only_qty = math.ceil(forecast)

    safety_stock_qty = int(row["target_inventory"])

    optimized_qty = int(row["optimized_order_quantity"])

    policies = {
        "forecast_only": forecast_only_qty,
        "safety_stock_20_percent": safety_stock_qty,
        "ortools_optimized": optimized_qty
    }

    result = {"item_id": item_id}

    for policy, quantity in policies.items():

        holding, stockout = calculate_expected_cost(
            quantity, forecast
        )

        result[f"{policy}_quantity"] = quantity
        result[f"{policy}_holding_cost"] = round(holding, 2)
        result[f"{policy}_stockout_cost"] = round(stockout, 2)
        result[f"{policy}_total_cost"] = round(
            holding + stockout, 2
        )

    # Verify optimized cost against Step 31 output
    previous_cost = float(row["total_expected_cost"])
    current_cost = result["ortools_optimized_total_cost"]

    if abs(previous_cost - current_cost) > 0.02:
        raise ValueError(
            f"Cost mismatch for product {item_id}"
        )

    results.append(result)


# --------------------------------------------------
# 5. SAVE COMPARISON
# --------------------------------------------------

comparison_df = pd.DataFrame(results)

comparison_df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# 6. SUMMARY
# --------------------------------------------------

policies = [
    "forecast_only",
    "safety_stock_20_percent",
    "ortools_optimized"
]

print("\n========== INVENTORY POLICY COMPARISON ==========")

summary = []

for policy in policies:

    total_quantity = comparison_df[
        f"{policy}_quantity"
    ].sum()

    total_holding = comparison_df[
        f"{policy}_holding_cost"
    ].sum()

    total_stockout = comparison_df[
        f"{policy}_stockout_cost"
    ].sum()

    total_cost = comparison_df[
        f"{policy}_total_cost"
    ].sum()

    summary.append({
        "policy": policy,
        "total_order_quantity": total_quantity,
        "expected_holding_cost": round(total_holding, 2),
        "expected_stockout_cost": round(total_stockout, 2),
        "total_expected_cost": round(total_cost, 2)
    })

summary_df = pd.DataFrame(summary)

print(summary_df.to_string(index=False))

# --------------------------------------------------
# 7. VALIDATION
# --------------------------------------------------

optimized_cost = summary_df.loc[
    summary_df["policy"] == "ortools_optimized",
    "total_expected_cost"
].iloc[0]

best_baseline_cost = summary_df.loc[
    summary_df["policy"] != "ortools_optimized",
    "total_expected_cost"
].min()

if optimized_cost <= best_baseline_cost + 0.02:
    print("\nPASS: OR-Tools cost is no higher than the best baseline.")
else:
    raise ValueError("Optimization validation failed.")

print("\nProducts validated:", len(comparison_df))
print("Saved comparison:", OUTPUT_FILE)
print("\nInventory policy validation completed!")
