
# Data Dictionary
## Retail Demand Forecasting and Inventory Optimization

## 1. Dataset Overview

This project uses the Walmart M5 Forecasting dataset.

The dataset contains historical daily product sales, calendar events, store information and weekly selling prices.

The dataset is used to build demand forecasting models and demonstrate inventory optimization.

### Dataset dimensions

| Dataset | Rows | Columns |
|---|---:|---:|
| calendar.csv | 1,969 | 14 |
| sales_train_validation.csv | 30,490 | 1,919 |
| sales_train_evaluation.csv | 30,490 | 1,947 |
| sell_prices.csv | 6,841,121 | 4 |
| sample_submission.csv | 60,980 | 29 |

## 2. Source Datasets

### 2.1 calendar.csv

Contains date-level information, including weekdays, months, years, events and SNAP indicators.

| Column | Description |
|---|---|
| date | Calendar date |
| wm_yr_wk | Walmart year-week identifier |
| weekday | Name of the weekday |
| wday | Numeric weekday identifier |
| month | Month number |
| year | Calendar year |
| event_name_1 | First event name |
| event_type_1 | First event category |
| event_name_2 | Second event name |
| event_type_2 | Second event category |
| snap_CA | SNAP indicator for California |
| snap_TX | SNAP indicator for Texas |
| snap_WI | SNAP indicator for Wisconsin |
| d | Dataset day identifier |

### 2.2 sales_train_validation.csv

Contains historical daily unit sales for products across Walmart stores.

| Column | Description |
|---|---|
| id | Unique product-store series identifier |
| item_id | Product identifier |
| dept_id | Department identifier |
| cat_id | Product category identifier |
| store_id | Store identifier |
| state_id | State identifier |
| d_1 to d_1913 | Daily unit sales observations |

**Dataset hierarchy:**

State → Store → Category → Department → Product → Daily Sales

### 2.3 sales_train_evaluation.csv

Contains the extended sales history used in the M5 evaluation dataset.

It has the same six metadata columns as the validation dataset, followed by daily sales columns from `d_1` to `d_1941`.

### 2.4 sell_prices.csv

Contains weekly selling prices for products at individual stores.

| Column | Description |
|---|---|
| store_id | Store identifier |
| item_id | Product identifier |
| wm_yr_wk | Walmart year-week identifier |
| sell_price | Product selling price |

**Relationship:** Product and store identifiers, together with the year-week identifier, connect prices to sales and calendar data.

### 2.5 sample_submission.csv

Contains the required submission format for the M5 forecasting competition.

It contains product-store identifiers and forecast columns for the evaluation horizon.

## 3. Engineered Features

The project creates additional features to improve demand forecasting.

### 3.1 Calendar Features

| Feature | Description |
|---|---|
| day_of_week | Day of the week |
| is_weekend | Indicates whether the day is a weekend |
| month | Month number |
| year | Calendar year |
| day_of_month | Day within the month |
| quarter | Calendar quarter |
| is_event | Indicates whether an event is present |

Missing event names and types are represented as `No Event`.

### 3.2 Price Features

| Feature | Description |
|---|---|
| sell_price | Current weekly selling price |
| previous_price | Previous available price for the product-store series |
| price_change | Difference between current and previous price |
| price_change_pct | Percentage change in price |

Missing historical prices are retained as missing values during the initial feature engineering process.

### 3.3 Lag Features

Lag features capture previous sales observations.

| Feature | Description |
|---|---|
| lag_1 | Sales from the previous day |
| lag_7 | Sales from seven days earlier |
| lag_28 | Sales from 28 days earlier |

### 3.4 Rolling Features

Rolling features capture recent demand patterns.

| Feature | Description |
|---|---|
| rolling_mean_7 | Average sales over the previous seven days |
| rolling_mean_28 | Average sales over the previous 28 days |

**Leakage prevention:** Rolling features are calculated using shifted sales values so that the current day's sales are not included in its own features.

## 4. Data Quality and Processing

The following checks were performed:

- Checked duplicate product-store identifiers.
- Checked duplicate item-store-week price records.
- Examined missing values in source datasets.
- Investigated unusually low daily sales.
- Examined missing selling prices across stores.
- Verified calendar and price feature alignment.
- Validated lag and rolling calculations.
- Checked integrated dataset row counts.

### Important observations

- The sales dataset contains 30,490 product-store series.
- Historical sales contain a high proportion of zero-sales observations.
- Extremely low sales were observed on Christmas dates.
- Missing selling prices were associated with zero-sales observations in the investigated dataset.
- Missing prices were retained rather than automatically replaced with zero.

## 5. Processed Dataset Structure

The project generates the following processed datasets:

| Output | Purpose |
|---|---|
| calendar_features.csv | Engineered calendar information |
| price_features.csv | Historical price features |
| combined_{store}.csv | Store-level sales, calendar and price data |
| lag_features_{store}.csv | Lag and rolling demand features |
| modeling_{store}.csv | Integrated modeling dataset |

The processing pipeline creates separate datasets for the ten Walmart stores.

Each store-level integrated modeling dataset contains 5,832,737 records.

## 6. Modeling Dataset

The integrated datasets are used for demand forecasting.

The forecasting workflow includes:

- Naive forecasting
- Seasonal Naive forecasting
- Prophet
- LightGBM
- Temporal Fusion Transformer (TFT)

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Weighted Absolute Percentage Error (WAPE)

## 7. Inventory Optimization Dataset

Forecast outputs are aggregated by product to prepare inventory demand estimates.

| Field | Description |
|---|---|
| item_id | Product identifier |
| total_forecast_demand | Total predicted demand over the forecast horizon |
| average_daily_demand | Average predicted daily demand |
| total_actual_demand | Actual demand during the validation period |
| forecast_days | Number of forecast days |
| safety_stock | Additional inventory allowance |
| target_inventory | Forecast demand plus safety stock |

The OR-Tools optimization stage calculates recommended order quantities using illustrative holding costs, stockout costs and demand scenarios.

## 8. Important Limitations

- Forecasting model comparisons use the first 100 product series from the CA_1 store.
- The validation period consists of 28 days.
- TFT training uses a limited CPU training configuration.
- Prediction intervals require further calibration.
- Darts reconciliation demonstrates bottom-up aggregation for the selected 100 products, not the complete M5 hierarchy.
- Inventory cost assumptions are illustrative and do not represent actual Walmart financial costs.
