
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Lifetime Value & Churn Intelligence")
st.caption("Nexvia Technologies | DS-01 Data Science Internship")


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

@st.cache_data
def load_data():
    clv = pd.read_csv(
        BASE_DIR / "data" / "telco_clv_results.csv"
    )

    segments = pd.read_csv(
        BASE_DIR / "data" / "customer_segmentation_results.csv"
    )

    profiles = pd.read_csv(
        BASE_DIR / "data" / "customer_cluster_profiles.csv"
    )

    return clv, segments, profiles


def show_bar_chart(data, title, xlabel="", ylabel="Count"):
    """Display a Matplotlib bar chart without Streamlit's bar_chart."""

    fig, ax = plt.subplots(figsize=(8, 4))

    data.plot(kind="bar", ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


def show_table(data):
    """Display a pandas table as HTML to avoid dataframe Arrow rendering."""

    html = data.to_html(
        index=False,
        border=0,
        classes="project-table"
    )

    st.markdown(
        """
        <style>
        .project-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        .project-table th {
            background-color: #262730;
            color: white;
            padding: 10px;
            text-align: left;
        }
        .project-table td {
            padding: 9px;
            border-bottom: 1px solid #555;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(html, unsafe_allow_html=True)


# ==================================================
# 3. LOAD DATA
# ==================================================

try:
    df, segments, profiles = load_data()

except FileNotFoundError as error:
    st.error(f"Required project data file is missing: {error}")
    st.stop()

except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()


# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Overview",
        "Churn Analytics",
        "Customer Segmentation",
        "CLV Analysis",
        "ROI Simulator",
        "Model Performance"
    ]
)


# ==================================================
# 5. OVERVIEW
# ==================================================

if page == "Overview":

    st.header("Business Overview")

    total_customers = len(df)

    churn_count = (df["Churn"] == "Yes").sum()

    churn_rate = (
        churn_count / total_customers * 100
        if total_customers > 0 else 0
    )

    average_clv = df["historical_clv"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    col3.metric(
        "Churned Customers",
        f"{churn_count:,}"
    )

    col4.metric(
        "Average Historical CLV",
        f"{average_clv:,.2f}"
    )

    st.subheader("Customer Churn Distribution")

    churn_data = df["Churn"].value_counts()

    show_bar_chart(
        churn_data,
        "Customer Churn Distribution",
        "Churn Status",
        "Number of Customers"
    )

    st.subheader("Customer Segmentation")

    cluster_counts = (
        segments["customer_cluster"]
        .value_counts()
        .sort_index()
    )

    show_bar_chart(
        cluster_counts,
        "Customer Distribution by Cluster",
        "Cluster",
        "Number of Customers"
    )

    st.info(
        "This dashboard combines historical customer data, "
        "churn analysis and customer segmentation."
    )


# ==================================================
# 6. CHURN ANALYTICS
# ==================================================

elif page == "Churn Analytics":

    st.header("Customer Churn Analytics")

    churn_summary = df.groupby("Churn").agg(
        customers=("Churn", "count"),
        average_tenure=("tenure", "mean"),
        average_monthly_charges=("MonthlyCharges", "mean"),
        average_historical_clv=("historical_clv", "mean")
    ).round(2).reset_index()

    st.subheader("Churn Summary")

    show_table(churn_summary)

    st.subheader("Churn by Contract")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100

    show_bar_chart(
        contract_churn,
        "Churn Rate by Contract",
        "Contract Type",
        "Churn Percentage"
    )

    st.subheader("Monthly Charges by Churn")

    monthly_charges = (
        df.groupby("Churn")["MonthlyCharges"].mean()
    )

    show_bar_chart(
        monthly_charges,
        "Average Monthly Charges by Churn",
        "Churn Status",
        "Average Monthly Charges"
    )


# ==================================================
# 7. CUSTOMER SEGMENTATION
# ==================================================

elif page == "Customer Segmentation":

    st.header("Customer Segmentation")

    st.subheader("Cluster Profiles")

    show_table(profiles)

    st.subheader("Customer Distribution")

    cluster_counts = (
        segments["customer_cluster"]
        .value_counts()
        .sort_index()
    )

    show_bar_chart(
        cluster_counts,
        "Customer Distribution by Cluster",
        "Cluster",
        "Number of Customers"
    )

    st.subheader("Explore Individual Clusters")

    available_clusters = sorted(
        segments["customer_cluster"].unique()
    )

    default_cluster = (
        2 if 2 in available_clusters
        else available_clusters[0]
    )

    selected_cluster = st.selectbox(
        "Choose Customer Cluster",
        available_clusters,
        index=available_clusters.index(default_cluster)
    )

    selected_data = segments[
        segments["customer_cluster"] == selected_cluster
    ]

    st.write(
        f"Customers in selected cluster: {len(selected_data):,}"
    )

    show_table(selected_data.head(20))


# ==================================================
# 8. CLV ANALYSIS
# ==================================================

elif page == "CLV Analysis":

    st.header("Customer Lifetime Value Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Historical CLV",
        f"{df['historical_clv'].mean():,.2f}"
    )

    col2.metric(
        "Median Historical CLV",
        f"{df['historical_clv'].median():,.2f}"
    )

    col3.metric(
        "Average Monthly Revenue",
        f"{df['average_monthly_revenue'].mean():,.2f}"
    )

    st.subheader("CLV Distribution")

    clv_by_segment = (
        df.groupby("clv_segment")["historical_clv"].mean()
    )

    show_bar_chart(
        clv_by_segment,
        "Average Historical CLV by Segment",
        "CLV Segment",
        "Historical CLV"
    )

    st.subheader("Historical CLV by Churn Status")

    clv_by_churn = (
        df.groupby("Churn")["historical_clv"].mean()
    )

    show_bar_chart(
        clv_by_churn,
        "Historical CLV by Churn Status",
        "Churn Status",
        "Historical CLV"
    )

    st.caption(
        "Historical CLV represents accumulated charges in the "
        "available dataset. It is not a prediction of future CLV."
    )


# ==================================================
# 9. ROI SIMULATOR
# ==================================================

elif page == "ROI Simulator":

    st.header("Retention Campaign ROI Simulator")

    st.write(
        "Estimate the potential financial impact of a "
        "customer retention campaign."
    )

    st.warning(
        "This is a scenario simulator. Retention success rate, "
        "offer cost and additional retention duration are "
        "user assumptions, not validated causal predictions."
    )

    cluster_options = sorted(
        segments["customer_cluster"].unique()
    )

    selected_clusters = st.multiselect(
        "Select customer segments to target",
        cluster_options,
        default=[2] if 2 in cluster_options else cluster_options[:1]
    )

    offer_cost = st.number_input(
        "Retention offer cost per customer",
        min_value=0.0,
        value=200.0,
        step=50.0
    )

    retention_rate = st.slider(
        "Assumed incremental retention success rate (%)",
        min_value=1,
        max_value=100,
        value=10
    )

    additional_months = st.number_input(
        "Assumed additional months retained",
        min_value=1,
        max_value=24,
        value=6
    )

    if st.button("Calculate Campaign ROI"):

        if not selected_clusters:

            st.error(
                "Please select at least one customer segment."
            )

        else:

            targeted = segments[
                segments["customer_cluster"].isin(selected_clusters)
            ]

            targeted_count = len(targeted)

            customer_ids = targeted["customerID"]

            target_customers = df[
                df["customerID"].isin(customer_ids)
            ]

            average_monthly_revenue = (
                target_customers["MonthlyCharges"].mean()
            )

            expected_saved = (
                targeted_count * retention_rate / 100
            )

            campaign_cost = targeted_count * offer_cost

            expected_revenue = (
                expected_saved
                * average_monthly_revenue
                * additional_months
            )

            net_benefit = expected_revenue - campaign_cost

            roi = (
                (net_benefit / campaign_cost) * 100
                if campaign_cost > 0 else 0
            )

            st.subheader("Campaign Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Targeted Customers",
                f"{targeted_count:,}"
            )

            col2.metric(
                "Expected Retained Customers",
                f"{expected_saved:,.0f}"
            )

            col3.metric(
                "Campaign Cost",
                f"{campaign_cost:,.2f}"
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Expected Revenue",
                f"{expected_revenue:,.2f}"
            )

            col5.metric(
                "Net Benefit",
                f"{net_benefit:,.2f}"
            )

            col6.metric(
                "Estimated ROI",
                f"{roi:.2f}%"
            )

            st.subheader("Financial Summary")

            financial_summary = pd.Series({
                "Campaign Cost": campaign_cost,
                "Expected Revenue": expected_revenue
            })

            show_bar_chart(
                financial_summary,
                "Campaign Financial Summary",
                "Category",
                "Amount"
            )

            st.caption(
                "Revenue estimate assumes retained customers generate "
                "the selected average monthly revenue for the assumed "
                "additional months. It excludes service costs, discounts, "
                "and other operating expenses."
            )


# ==================================================
# 10. MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.header("Machine Learning Model Performance")

    st.subheader("Churn Prediction")

    churn_metrics = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "LightGBM Default",
            "LightGBM Balanced"
        ],
        "Accuracy": [
            0.8055,
            0.7892,
            0.7566
        ],
        "ROC-AUC": [
            0.8421,
            0.8291,
            0.8341
        ]
    })

    show_table(churn_metrics)

    st.subheader("Survival Analysis")

    survival_metrics = pd.DataFrame({
        "Model": [
            "Baseline Cox",
            "Stratified Cox"
        ],
        "Test C-index": [
            0.8318,
            0.8019
        ]
    })

    show_table(survival_metrics)

    st.subheader("Uplift Model")

    uplift_metrics = pd.DataFrame({
        "Metric": [
            "Control ROC-AUC",
            "Treatment ROC-AUC",
            "Qini Coefficient"
        ],
        "Value": [
            0.6577,
            0.5347,
            0.00076305
        ]
    })

    show_table(uplift_metrics)

    st.caption(
        "These are recorded experiment results. The uplift model "
        "has weak treatment outcome discrimination and should not "
        "yet be used to automate real retention decisions."
    )


# ==================================================
# 11. FOOTER
# ==================================================

st.divider()

st.caption(
    "Nexvia Technologies | DS-01 | Customer Lifetime Value "
    "and Churn Intelligence System"
)
