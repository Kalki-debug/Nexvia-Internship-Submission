
import os
import pandas as pd
from ortools.linear_solver import pywraplp

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

INPUT_FILE = (
    "data/processed/inventory/"
    "inventory_demand_CA_1_100.csv"
)

OUTPUT_DIR = "reports/inventory"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "optimized_inventory_CA_1_100.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Assumed costs per unit over the 28-day horizon
HOLDING_COST = 1.0
STOCKOUT_COST = 5.0

# Demand scenarios and their probabilities
SCENARIOS = {
    "low": (0.80, 0.20),
    "expected": (1.00, 0.60),
    "high": (1.20, 0.20)
}

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading inventory demand data...")

df = pd.read_csv(INPUT_FILE)

results = []

# --------------------------------------------------
# 3. OPTIMIZE EACH PRODUCT
# --------------------------------------------------

print("Starting OR-Tools inventory optimization...")

for _, row in df.iterrows():

    item_id = row["item_id"]
    forecast = float(row["total_forecast_demand"])

    solver = pywraplp.Solver.CreateSolver("SCIP")

    if solver is None:
        solver = pywraplp.Solver.CreateSolver("CBC")

    if solver is None:
        raise RuntimeError("No suitable OR-Tools solver available.")

    # Integer order quantity
    order_qty = solver.IntVar(
        0, solver.infinity(), "order_quantity"
    )

    scenario_vars = {}

    objective = solver.Objective()

    for scenario, (multiplier, probability) in SCENARIOS.items():

        demand = int(round(forecast * multiplier))

        leftover = solver.IntVar(
            0, solver.infinity(), f"leftover_{scenario}"
        )

        shortage = solver.IntVar(
            0, solver.infinity(), f"shortage_{scenario}"
        )

        # Order quantity = demand + leftover - shortage
        solver.Add(
            order_qty - leftover + shortage == demand
        )

        scenario_vars[scenario] = {
            "demand": demand,
            "leftover": leftover,
            "shortage": shortage
        }

        objective.SetCoefficient(
            leftover,
            probability * HOLDING_COST
        )

        objective.SetCoefficient(
            shortage,
            probability * STOCKOUT_COST
        )

    objective.SetMinimization()

    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(
            f"Optimization failed for product: {item_id}"
        )

    # --------------------------------------------------
    # 4. COLLECT RESULTS
    # --------------------------------------------------

    optimized_qty = int(round(order_qty.solution_value()))

    expected_holding_cost = 0
    expected_stockout_cost = 0
    expected_leftover = 0
    expected_shortage = 0

    for scenario, (_, probability) in SCENARIOS.items():

        variables = scenario_vars[scenario]

        leftover_value = variables["leftover"].solution_value()
        shortage_value = variables["shortage"].solution_value()

        expected_leftover += probability * leftover_value
        expected_shortage += probability * shortage_value

        expected_holding_cost += (
            probability * HOLDING_COST * leftover_value
        )

        expected_stockout_cost += (
            probability * STOCKOUT_COST * shortage_value
        )

    results.append({
        "item_id": item_id,
        "forecast_demand": round(forecast, 2),
        "optimized_order_quantity": optimized_qty,
        "expected_leftover": round(expected_leftover, 2),
        "expected_shortage": round(expected_shortage, 2),
        "expected_holding_cost": round(expected_holding_cost, 2),
        "expected_stockout_cost": round(expected_stockout_cost, 2),
        "total_expected_cost": round(
            expected_holding_cost + expected_stockout_cost, 2
        )
    })

# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\n========== INVENTORY OPTIMIZATION RESULTS ==========")

print("Products optimized:", len(results_df))

print(
    "Total optimized order quantity:",
    results_df["optimized_order_quantity"].sum()
)

print(
    "Total expected inventory cost:",
    round(results_df["total_expected_cost"].sum(), 2)
)

print("\nSample results:")
print(results_df.head())

print("\nSaved:", OUTPUT_FILE)
print("\nInventory optimization completed successfully!")
