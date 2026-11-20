# Customer Lifetime Value and Churn Intelligence System

**Nexvia Technologies | Data Science Internship | Project DS-01**

## 1. Project Overview

The Customer Lifetime Value and Churn Intelligence System is a data science project designed to analyze customer behavior, identify customers at risk of churn, estimate historical customer value, and explore customer retention strategies.

The project combines data analytics, machine learning, explainable AI, survival analysis, uplift modeling, customer segmentation, and an interactive Streamlit dashboard.

## 2. Project Objectives

- Analyze customer churn patterns and behavior.
- Predict customer churn using machine learning.
- Explain model predictions using SHAP.
- Estimate customer survival patterns using Kaplan-Meier and Cox regression.
- Explore treatment effects using uplift modeling.
- Calculate historical Customer Lifetime Value (CLV).
- Segment customers using K-Means clustering.
- Provide business insights through an interactive dashboard.

## 3. Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib |
| Database | PostgreSQL |
| Machine Learning | Scikit-learn, LightGBM |
| Explainable AI | SHAP |
| Survival Analysis | Lifelines |
| Statistical Analysis | SciPy |
| Dashboard | Streamlit |
| Model Storage | Joblib |
| Development | VS Code |

## 4. Datasets

### IBM Telco Customer Churn Dataset
- 7,043 customer records.
- 21 original columns.
- Used for EDA, churn prediction, survival analysis, CLV calculation, and customer segmentation.

### Hillstrom Email Marketing Dataset
- 64,000 original records.
- Used for exploratory uplift modeling.
- Women's Email and No Email groups were selected for the treatment-control comparison.
- Men's Email group was excluded from this comparison.

## 5. Project Workflow

1. Dataset collection and PostgreSQL integration.
2. Exploratory Data Analysis and data cleaning.
3. Feature engineering and preprocessing.
4. Churn prediction using Logistic Regression and LightGBM.
5. Model interpretation using SHAP.
6. Customer survival analysis using Kaplan-Meier and Cox regression.
7. Uplift modeling using a T-Learner approach.
8. Historical CLV calculation.
9. Customer segmentation using K-Means.
10. Interactive dashboard development using Streamlit.

## 6. Key Results

### Churn Prediction

Best recorded ROC-AUC: **0.8421 - Logistic Regression**

| Metric | Result |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| F1-score | 60.40% |
| ROC-AUC | 0.8421 |

### Survival Analysis

| Model | Test C-index |
|---|---:|
| Baseline Cox | 0.8318 |
| Stratified Cox | 0.8019 |

### Uplift Modeling

| Metric | Result |
|---|---:|
| Control ROC-AUC | 0.6577 |
| Treatment ROC-AUC | 0.5347 |
| Qini Coefficient | 0.00076305 |

### Historical CLV

| Metric | Result |
|---|---:|
| Average Historical CLV | 2,279.73 |
| Median Historical CLV | 1,394.55 |
| Average Monthly Revenue | 64.70 |

### Customer Segmentation

- K-Means clustering with four clusters.
- Silhouette score: 0.4549.
- Customer groups analyzed using tenure, monthly charges, and average monthly revenue.

## 7. Streamlit Dashboard

The dashboard contains six sections:

1. Business Overview
2. Churn Analytics
3. Customer Segmentation
4. CLV Analysis
5. ROI Simulator
6. Model Performance

The ROI simulator allows users to explore hypothetical retention campaign scenarios using configurable assumptions.

## 8. Important Limitations

- Historical CLV represents accumulated charges and is not a prediction of future customer value.
- Survival analysis uses customer tenure and churn status as an approximation because precise event and observation dates are unavailable.
- The uplift model is exploratory and has limited predictive performance. It should not be used for automated retention decisions.
- The ROI simulator uses user-defined assumptions and does not represent validated causal campaign outcomes.
- Cox proportional hazards diagnostics identified remaining assumption violations.

## 9. Running the Project

### Step 1: Clone the repository

```bash
git clone <https://github.com/Kalki-debug/customer-lifetime-value-churn-intelligence.git>
cd ds-01
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the environment (Windows)

```powershell
.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run the dashboard

```bash
python -m streamlit run 20_streamlit_app.py
```

Open the local URL displayed in the terminal.

**Note:** The required data files and model artifacts must be available in their expected project directories. PostgreSQL configuration and dataset import may require separate setup.

## 10. Project Outcome

This project demonstrates an integrated data science workflow combining customer analytics, machine learning, model explainability, survival analysis, uplift modeling, customer value analysis, segmentation, and business-oriented dashboard development.

**Developed as part of the Nexvia Technologies Data Science Internship — DS-01.**