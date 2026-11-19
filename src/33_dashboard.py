
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def project_path(*parts):
    return os.path.join(BASE_DIR, *parts)


INTERVAL_FILE = project_path(
    "reports", "baseline",
    "tft_prediction_intervals_CA_1_100.csv"
)

INVENTORY_FILE = project_path(
    "reports", "inventory",
    "optimized_inventory_CA_1_100.csv"
)

POLICY_FILE = project_path(
    "reports", "inventory",
    "inventory_policy_comparison.csv"
)

SALES_FILE = project_path(
    "data", "raw", "sales_train_validation.csv"
)

CALENDAR_FILE = project_path(
    "data", "raw", "calendar.csv"
)


# ==================================================
# 2. LOAD DASHBOARD DATA
# ==================================================

print("Loading dashboard data...")

interval_df = pd.read_csv(INTERVAL_FILE)
inventory_df = pd.read_csv(INVENTORY_FILE)
policy_raw = pd.read_csv(POLICY_FILE)

policy_df = pd.DataFrame({
    "policy": [
        "Forecast Only",
        "20% Safety Stock",
        "OR-Tools Optimized"
    ],
    "total_expected_cost": [
        policy_raw["forecast_only_total_cost"].sum(),
        policy_raw["safety_stock_20_percent_total_cost"].sum(),
        policy_raw["ortools_optimized_total_cost"].sum()
    ]
})

# Forecast model benchmark metrics:
# Same 100-product CA_1 validation subset.

model_metrics = pd.DataFrame([
    {"Model": "Naive", "MAE": 1.3296, "RMSE": 3.4989, "WAPE": 107.38},
    {"Model": "Seasonal Naive", "MAE": 1.4032, "RMSE": 3.5559, "WAPE": 113.33},
    {"Model": "Prophet", "MAE": 1.1640, "RMSE": 2.6638, "WAPE": 94.01},
    {"Model": "LightGBM", "MAE": 1.2150, "RMSE": 2.6036, "WAPE": 98.13},
    {"Model": "TFT", "MAE": 1.0774, "RMSE": 2.8912, "WAPE": 87.01}
])


# ==================================================
# 3. BUILD HISTORICAL DAILY SALES TREND
# ==================================================

print("Calculating historical daily sales...")

# Read sales in chunks to reduce memory usage.
daily_sales = None

for chunk in pd.read_csv(
    SALES_FILE,
    usecols=lambda col: col.startswith("d_"),
    chunksize=1000
):
    chunk_total = chunk.sum(axis=0)

    if daily_sales is None:
        daily_sales = chunk_total
    else:
        daily_sales = daily_sales.add(
            chunk_total, fill_value=0
        )

sales_trend = pd.DataFrame({
    "d": daily_sales.index,
    "total_sales": daily_sales.values
})

calendar = pd.read_csv(
    CALENDAR_FILE,
    usecols=["d", "date"]
)

sales_trend = sales_trend.merge(
    calendar,
    on="d",
    how="left"
)

sales_trend["date"] = pd.to_datetime(sales_trend["date"])

sales_trend = sales_trend.sort_values("date")


# ==================================================
# 4. CALCULATE INTERVAL METRICS
# ==================================================

interval_coverage = (
    (interval_df["actual_sales"] >= interval_df["P10_lower"])
    & (interval_df["actual_sales"] <= interval_df["P90_upper"])
).mean() * 100

average_interval_width = (
    interval_df["P90_upper"] - interval_df["P10_lower"]
).mean()


# ==================================================
# 5. PREPARE INVENTORY SUMMARY
# ==================================================

total_products = len(inventory_df)

total_order_quantity = int(
    inventory_df["optimized_order_quantity"].sum()
)

total_expected_cost = float(
    inventory_df["total_expected_cost"].sum()
)

total_forecast = float(
    inventory_df["forecast_demand"].sum()
)


# ==================================================
# 6. DASH APPLICATION
# ==================================================

app = Dash(__name__)

app.title = "Retail Demand Intelligence"


# ==================================================
# 7. DASHBOARD LAYOUT
# ==================================================

card_style = {
    "backgroundColor": "#ffffff",
    "padding": "20px",
    "borderRadius": "10px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
    "flex": "1",
    "minWidth": "180px"
}

section_style = {
    "backgroundColor": "#ffffff",
    "padding": "22px",
    "marginTop": "22px",
    "borderRadius": "10px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.06)"
}


app.layout = html.Div(
    style={
        "backgroundColor": "#f3f5f9",
        "minHeight": "100vh",
        "padding": "25px",
        "fontFamily": "Arial, sans-serif"
    },
    children=[

        # HEADER
        html.Div([
            html.H1(
                "Retail Demand Forecasting & Inventory Optimization",
                style={"color": "#172554", "marginBottom": "8px"}
            ),
            html.P(
                "Walmart M5 Dataset | CA_1 | 100 Products",
                style={"color": "#64748b", "fontSize": "15px"}
            )
        ]),

        # KPI CARDS
        html.Div(
            style={
                "display": "flex",
                "gap": "18px",
                "flexWrap": "wrap",
                "marginTop": "25px"
            },
            children=[

                html.Div([
                    html.H4("Products Analyzed"),
                    html.H2(
                        f"{total_products:,}",
                        style={"color": "#2563eb"}
                    )
                ], style=card_style),

                html.Div([
                    html.H4("Forecast Demand"),
                    html.H2(
                        f"{total_forecast:,.0f}",
                        style={"color": "#0891b2"}
                    )
                ], style=card_style),

                html.Div([
                    html.H4("Optimized Order Quantity"),
                    html.H2(
                        f"{total_order_quantity:,}",
                        style={"color": "#059669"}
                    )
                ], style=card_style),

                html.Div([
                    html.H4("Expected Inventory Cost"),
                    html.H2(
                        f"{total_expected_cost:,.2f}",
                        style={"color": "#7c3aed"}
                    )
                ], style=card_style)
            ]
        ),

        # SECTION 1: DEMAND ANALYSIS
        html.Div([
            html.H2("1. Historical Demand Analysis"),

            dcc.Graph(
                figure=px.line(
                    sales_trend,
                    x="date",
                    y="total_sales",
                    title="Daily Total Sales – All M5 Series",
                    labels={
                        "date": "Date",
                        "total_sales": "Units Sold"
                    }
                ).update_layout(
                    template="plotly_white"
                )
            )
        ], style=section_style),

        # SECTION 2: MODEL COMPARISON
        html.Div([
            html.H2("2. Forecast Model Performance"),

            html.Label("Select evaluation metric:"),

            dcc.Dropdown(
                id="metric-selector",
                options=[
                    {"label": "MAE", "value": "MAE"},
                    {"label": "RMSE", "value": "RMSE"},
                    {"label": "WAPE (%)", "value": "WAPE"}
                ],
                value="MAE",
                clearable=False,
                style={"maxWidth": "300px"}
            ),

            dcc.Graph(id="model-comparison-chart"),

            html.P(
                "Evaluation: 100 CA_1 products, "
                "28-day holdout (d_1886–d_1913). "
                "Lower values indicate better performance.",
                style={"color": "#64748b"}
            )
        ], style=section_style),

        # SECTION 3: PREDICTION INTERVALS
        html.Div([
            html.H2("3. TFT Prediction Intervals"),

            html.Div([
                html.Div([
                    html.H4("P10–P90 Coverage"),
                    html.H2(
                        f"{interval_coverage:.2f}%",
                        style={"color": "#2563eb"}
                    )
                ], style=card_style),

                html.Div([
                    html.H4("Average Interval Width"),
                    html.H2(
                        f"{average_interval_width:.2f}",
                        style={"color": "#7c3aed"}
                    )
                ], style=card_style)
            ], style={
                "display": "flex",
                "gap": "18px",
                "flexWrap": "wrap"
            }),

            html.Br(),

            html.Label("Select product:"),

            dcc.Dropdown(
                id="product-selector",
                options=[
                    {"label": item, "value": item}
                    for item in sorted(interval_df["item_id"].unique())
                ],
                value=sorted(interval_df["item_id"].unique())[0],
                clearable=False,
                style={"maxWidth": "400px"}
            ),

            dcc.Graph(id="prediction-interval-chart"),

            html.P(
                "P10 and P90 represent the lower and upper "
                "forecast quantiles. Coverage is the observed "
                "percentage of actual values inside this interval.",
                style={"color": "#64748b"}
            )
        ], style=section_style),

        # SECTION 4: INVENTORY OPTIMIZATION
        html.Div([
            html.H2("4. Inventory Optimization"),

            dcc.Graph(
                figure=px.bar(
                    policy_df,
                    x="policy",
                    y="total_expected_cost",
                    color="policy",
                    title="Expected Cost by Inventory Policy",
                    labels={
                        "policy": "Inventory Policy",
                        "total_expected_cost": "Expected Cost"
                    }
                ).update_layout(
                    template="plotly_white",
                    showlegend=False
                )
            ),

            dcc.Graph(
                figure=px.bar(
                    inventory_df.sort_values(
                        "optimized_order_quantity",
                        ascending=False
                    ).head(15),
                    x="item_id",
                    y="optimized_order_quantity",
                    title="Top 15 Products by Optimized Order Quantity",
                    labels={
                        "item_id": "Product",
                        "optimized_order_quantity": "Order Quantity"
                    }
                ).update_layout(
                    template="plotly_white",
                    xaxis_tickangle=-45
                )
            ),

            html.H3("Product-level Inventory Recommendations"),

            html.Div(
                id="inventory-table",
                style={"overflowX": "auto"}
            )

        ], style=section_style),

        # FOOTER
        html.Div([
            html.P(
                "Nexvia Technologies | Retail Demand Forecasting "
                "& Inventory Optimization System",
                style={
                    "textAlign": "center",
                    "color": "#64748b",
                    "marginTop": "30px"
                }
            )
        ])
    ]
)


# ==================================================
# 8. MODEL COMPARISON CALLBACK
# ==================================================

@app.callback(
    Output("model-comparison-chart", "figure"),
    Input("metric-selector", "value")
)
def update_model_chart(metric):

    fig = px.bar(
        model_metrics,
        x="Model",
        y=metric,
        color="Model",
        title=f"Model Comparison – {metric}",
        text=metric
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        yaxis_title=metric
    )

    return fig


# ==================================================
# 9. TFT INTERVAL CALLBACK
# ==================================================

@app.callback(
    Output("prediction-interval-chart", "figure"),
    Input("product-selector", "value")
)
def update_interval_chart(selected_product):

    product_data = interval_df[
        interval_df["item_id"] == selected_product
    ].sort_values("day_number")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=product_data["day_number"],
        y=product_data["P90_upper"],
        mode="lines",
        line={"width": 0},
        showlegend=False,
        name="P90"
    ))

    fig.add_trace(go.Scatter(
        x=product_data["day_number"],
        y=product_data["P10_lower"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(59,130,246,0.18)",
        line={"width": 0},
        name="P10–P90 interval"
    ))

    fig.add_trace(go.Scatter(
        x=product_data["day_number"],
        y=product_data["P50_median"],
        mode="lines+markers",
        name="P50 Median"
    ))

    fig.add_trace(go.Scatter(
        x=product_data["day_number"],
        y=product_data["actual_sales"],
        mode="lines+markers",
        name="Actual Sales"
    ))

    fig.update_layout(
        title=f"TFT Forecast vs Actual – {selected_product}",
        xaxis_title="Day Number",
        yaxis_title="Sales Units",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


# ==================================================
# 10. INVENTORY TABLE CALLBACK
# ==================================================

@app.callback(
    Output("inventory-table", "children"),
    Input("product-selector", "value")
)
def update_inventory_table(selected_product):

    display_df = inventory_df[
        [
            "item_id",
            "forecast_demand",
            "optimized_order_quantity",
            "expected_leftover",
            "expected_shortage",
            "total_expected_cost"
        ]
    ].sort_values(
        "optimized_order_quantity",
        ascending=False
    )

    header = html.Tr([
        html.Th(column, style={"padding": "10px"})
        for column in display_df.columns
    ])

    rows = []

    for _, row in display_df.iterrows():

        rows.append(
            html.Tr([
                html.Td(
                    str(row[column]),
                    style={
                        "padding": "9px",
                        "borderBottom": "1px solid #e2e8f0"
                    }
                )
                for column in display_df.columns
            ])
        )

    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "backgroundColor": "white"
        }
    )


# ==================================================
# 11. RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    print("\nDashboard is starting...")
    print("Open: http://127.0.0.1:8050")

    app.run(
        debug=False,
        host="127.0.0.1",
        port=8050
    )
