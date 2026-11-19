# Retail Demand Forecasting & Inventory Optimization System

## 1. Project Overview

The **Retail Demand Forecasting & Inventory Optimization System** is a data science project developed as part of the Nexvia Technologies Data Science Internship.

The project uses the Walmart M5 Forecasting dataset to analyze historical retail sales, forecast future product demand, and optimize inventory order quantities.

It combines time-series forecasting, machine learning, deep learning, hierarchical forecast reconciliation, and mathematical optimization in an interactive dashboard.

## 2. Problem Statement

Retail businesses need to maintain sufficient inventory to meet customer demand while avoiding excess stock and unnecessary holding costs.

This project aims to:

- Analyze historical retail sales patterns.
- Identify trends, seasonality, and demand-related factors.
- Build and compare multiple forecasting models.
- Generate prediction intervals to represent forecast uncertainty.
- Optimize inventory order quantities using OR-Tools.
- Present insights and recommendations through a Plotly Dash dashboard.

## 3. Dataset

**Dataset:** Walmart M5 Forecasting Dataset

The dataset contains historical Walmart sales information, calendar events, and product pricing.

| File | Description |
|---|---|
| `sales_train_validation.csv` | Historical daily product sales |
| `sales_train_evaluation.csv` | Extended sales history |
| `calendar.csv` | Dates, weekdays, events, and SNAP information |
| `sell_prices.csv` | Weekly product selling prices by store |
| `sample_submission.csv` | M5 forecasting submission format |

Dataset dimensions:

- Products: 3,049
- Stores: 10
- States: 3 (CA, TX, WI)
- Categories: 3
- Departments: 7
- Historical validation sales days: 1,913

## 4. Technologies Used

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Forecasting | Prophet, LightGBM |
| Deep Learning | PyTorch, PyTorch Forecasting |
| Forecast Reconciliation | Darts |
| Inventory Optimization | Google OR-Tools |
| Dashboard | Plotly Dash |
| Development | VS Code, Git, GitHub |

## 5. Project Workflow

### Stage 1 – Dataset Understanding
- Loaded and inspected the M5 dataset.
- Examined dataset dimensions, columns, and hierarchy.
- Identified products, stores, departments, and categories.

### Stage 2 – Exploratory Data Analysis
- Analyzed daily, monthly, and weekday sales patterns.
- Compared sales across categories, stores, and departments.
- Investigated unusually low-sales days, including Christmas.

### Stage 3 – Feature Engineering
Created features including:

- Lag features: `lag_1`, `lag_7`, `lag_28`
- Rolling averages: `rolling_mean_7`, `rolling_mean_28`
- Calendar features: weekday, weekend, month, year, and events.
- Price features: previous price, price change, and percentage change.

Generated store-wise combined datasets and modeling datasets.

### Stage 4 – Baseline Forecasting

Implemented and compared five forecasting approaches:

1. Naive Forecasting
2. Seasonal Naive Forecasting
3. Prophet
4. LightGBM
5. Temporal Fusion Transformer (TFT)

### Stage 5 – Forecast Reconciliation and Prediction Intervals

- Demonstrated bottom-up hierarchical forecast reconciliation using Darts.
- Generated TFT prediction intervals using quantile forecasts.
- Evaluated interval coverage and average interval width.

### Stage 6 – Inventory Optimization

Used Google OR-Tools to determine inventory order quantities.

The optimization considers:

- Forecast demand
- Low-demand scenario
- Expected-demand scenario
- High-demand scenario
- Holding costs
- Stockout costs

Compared three inventory policies:

- Forecast-only ordering
- 20% safety-stock ordering
- OR-Tools optimized ordering

### Stage 7 – Interactive Dashboard

Developed a Plotly Dash dashboard containing:

- Key project metrics
- Historical demand trends
- Forecasting model comparison
- TFT prediction intervals
- Inventory policy comparison
- Product-level optimized order quantities

## 6. Model Performance

**Evaluation setup:** First 100 products from CA_1, using a 28-day historical holdout.

Training period: `d_1` to `d_1885`

Validation period: `d_1886` to `d_1913`

| Model | MAE ↓ | RMSE ↓ | WAPE ↓ |
|---|---:|---:|---:|
| Naive | 1.3296 | 3.4989 | 107.38% |
| Seasonal Naive | 1.4032 | 3.5559 | 113.33% |
| Prophet | 1.1640 | 2.6638 | 94.01% |
| LightGBM | 1.2150 | **2.6036** | 98.13% |
| TFT | **1.0774** | 2.8912 | **87.01%** |

**Observations:**

- TFT achieved the lowest MAE and WAPE.
- LightGBM achieved the lowest RMSE.
- Both advanced models performed better than the naive baselines across all three metrics.

## 7. Inventory Optimization Results

The inventory optimization was demonstrated on 100 CA_1 products.

Assumptions:
- Holding cost: 1.0 per unit.
- Stockout cost: 5.0 per unit.
- Low-demand scenario: 80% of forecast demand (20% probability).
- Expected-demand scenario: 100% of forecast demand (60% probability).
- High-demand scenario: 120% of forecast demand (20% probability).

### Inventory Policy Comparison

| Policy | Order Quantity | Expected Cost |
|---|---:|---:|
| Forecast Only | 3,947 | 930.0 |
| 20% Safety Stock | 4,679 | 799.2 |
| OR-Tools Optimized | 4,683 | **782.8** |

The OR-Tools optimization achieved:

- 15.83% lower expected cost compared with forecast-only ordering.
- 2.05% lower expected cost compared with the 20% safety-stock policy.

*Note: These are illustrative cost units based on assumed cost parameters, not actual Walmart financial data or INR savings.*

## 8. Project Architecture

The system architecture diagram is available here:

[`docs/architecture.md`](docs/architecture.md)

## 9. Project Structure

```text
ds-02/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│
├── models/
│
├── reports/
│   ├── baseline/
│   └── inventory/
│
├── docs/
│   └── architecture.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 10. Installation and Execution

### Prerequisites

- Python 3.12
- Git
- Visual Studio Code

### Step 1 – Clone the repository

```bash
git clone <https://github.com/Kalki-debug?tab=repositories>
cd ds-02
```

### Step 2 – Create a virtual environment

```bash
python -m venv .venv
```

### Step 3 – Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 4 – Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5 – Run the dashboard

After preparing the required datasets, models, and report files:

```bash
python src/33_dashboard.py
```

## 📸 Dashboard Screenshots

### Dashboard Overview
![Dashboard Overview](c:\Users\HP\OneDrive\Pictures\Screenshots\dashboard_overview.png)

### Forecasting Analysis
![Forecasting Analysis](c:\Users\HP\OneDrive\Pictures\Screenshots\forecasting_analysis.png)

### Inventory Optimization
![Inventory Optimization](c:\Users\HP\OneDrive\Pictures\Screenshots\inventory_optimization.png)

Open the following address in your browser:

```text
http://127.0.0.1:8050
```

## 11. Important Limitations

- Model comparisons were conducted on a 100-product CA_1 subset rather than the entire M5 dataset.
- TFT training used a limited CPU training configuration of five epochs and 200 batches per epoch.
- TFT prediction interval coverage was 30.29%, below the intended 80% coverage for the P10–P90 interval.
- Darts reconciliation was demonstrated on 100 CA_1 product forecasts, not the complete M5 hierarchy.
- Inventory optimization used illustrative holding and stockout cost assumptions.
- The evaluation uses an internal historical holdout and should not be interpreted as an official M5 competition forecast evaluation.

## 12. Future Improvements

- Train forecasting models on larger portions of the M5 dataset.
- Improve TFT training and prediction interval calibration.
- Extend hierarchical reconciliation to store, department, category, and state levels.
- Incorporate more realistic inventory costs and operational constraints.
- Improve dashboard interactivity and deployment.
- Evaluate forecasts across multiple stores and longer validation periods.

## 13. Acknowledgements

- Walmart M5 Forecasting Dataset.
- Nexvia Technologies – Data Science Internship.

---

**Project:** Retail Demand Forecasting & Inventory Optimization System  
**Domain:** Data Science, Time-Series Forecasting, Machine Learning, and Operations Research  
**Language:** Python