# =========================================
# Enhanced Electricity Theft Detection Dashboard
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# -----------------------------------------
# Page Configuration
# -----------------------------------------
st.set_page_config(
    page_title="Electricity Theft Detection Dashboard",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("⚡ Electricity Theft Detection Dashboard⚡")
st.markdown(
    "Interactive, explainable AI dashboard for detecting electricity theft with actionable insights."
)

# -----------------------------------------
# Cached Loaders
# -----------------------------------------
@st.cache_data
def load_features():
    return pd.read_csv("data/processed/final_features.csv")

@st.cache_data
def load_risk_scores():
    return pd.read_csv("data/processed/risk_scores.csv")

@st.cache_resource
def load_model():
    with open("models/ensemble_model.pkl", "rb") as f:
        return pickle.load(f)

# -----------------------------------------
# Load Data
# -----------------------------------------
features_df = load_features()
risk_df = load_risk_scores()
model = load_model()

dashboard_df = features_df.merge(
    risk_df[["meter_id", "risk_score", "risk_category"]],
    on="meter_id",
    how="left"
)

X = dashboard_df.drop(columns=["meter_id", "is_theft", "risk_score", "risk_category"], errors="ignore")

# -----------------------------------------
# Tabs for clean layout
# -----------------------------------------
tab1, tab2, tab3 = st.tabs(["Portfolio Overview", "Customer Insights", "Explainable AI"])

# =========================================
# Tab 1: Portfolio Overview
# =========================================
with tab1:
    st.subheader("Portfolio Risk Overview")

    col1, col2, col3 = st.columns(3)
    total_customers = len(dashboard_df)
    high_risk_count = (dashboard_df['risk_category'] == 'High Risk').sum()
    avg_risk_score = dashboard_df['risk_score'].mean()

    col1.metric("Total Customers", f"{total_customers:,}", "⚡")
    col2.metric("High-Risk Customers", f"{high_risk_count:,}", "⚡")
    col3.metric("Average Risk Score", f"{avg_risk_score:.2f}", "⚡")

    st.markdown("### Risk Score Distribution")
    fig = px.histogram(
        dashboard_df,
        x="risk_score",
        color="risk_category",
        nbins=30,
        color_discrete_map={"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"},
        title="Distribution of Electricity Theft Risk Scores"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Global Feature Importance")

    # -----------------------------------------
    # Cached Global SHAP Importance (sampled)
    # -----------------------------------------
    @st.cache_data
    def compute_global_shap(X_sample, _explainer):
        shap_vals = _explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            shap_matrix = shap_vals[1]  # class 1
        elif shap_vals.ndim == 3:
            shap_matrix = shap_vals[:, :, 1]
        else:
            shap_matrix = shap_vals

        mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": X_sample.columns,
            "importance": mean_abs_shap
        }).sort_values("importance", ascending=False)
        return importance_df

    X_sample = X.sample(min(100, len(X)), random_state=42)
    # Using only the first estimator for SHAP (TreeExplainer)
    rf_model = model.named_estimators_["rf"].named_steps["model"]
    explainer_global = shap.TreeExplainer(rf_model)
    importance = compute_global_shap(X_sample, explainer_global)

    fig2 = px.bar(
        importance.head(15),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Viridis",
        hover_data={"feature": True, "importance": ":.4f"},
        title="Top 15 Features Driving Theft Predictions"
    )
    st.plotly_chart(fig2, use_container_width=True)

# =========================================
# Tab 2: Customer Insights
# =========================================
with tab2:
    st.subheader("Customer Risk Table & Inspection Simulator")

    risk_filter = st.multiselect(
        "Filter by Risk Category",
        options=dashboard_df["risk_category"].unique(),
        default=list(dashboard_df["risk_category"].unique())
    )

    filtered_df = dashboard_df[dashboard_df["risk_category"].isin(risk_filter)]

    # Conditional coloring
    def color_risk(val):
        color = "green" if val == "Low Risk" else "orange" if val == "Medium Risk" else "red"
        return f"color: {color}"

    st.dataframe(
        filtered_df[["meter_id", "risk_score", "risk_category"]].style.applymap(color_risk, subset=["risk_category"]),
        height=350
    )

    st.markdown("### 🛠️ Inspection Strategy Simulator")

    inspection_capacity = st.slider(
        "Number of Inspections Available",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )

    strategy = st.radio(
        "Select Inspection Strategy",
        ["Top Risk", "Random", "Mixed"]
    )

    if strategy == "Top Risk":
        priority_df = filtered_df.sort_values("risk_score", ascending=False).head(inspection_capacity)
    elif strategy == "Random":
        priority_df = filtered_df.sample(inspection_capacity, random_state=42)
    else:  # Mixed
        top_count = int(inspection_capacity * 0.7)
        random_count = inspection_capacity - top_count
        priority_df = pd.concat([
            filtered_df.sort_values("risk_score", ascending=False).head(top_count),
            filtered_df.sample(random_count, random_state=42)
        ])

    expected_hits = (priority_df["risk_score"] > 0.7).sum()

    st.success(
        f"Recommended inspections: **{inspection_capacity} customers**\n"
        f"Expected high-risk detections: **{expected_hits} customers**"
    )

    st.dataframe(priority_df[["meter_id", "risk_score", "risk_category"]], height=300)

# =========================================
# Tab 3: Explainable AI (Individual Customer)
# =========================================
with tab3:
    st.subheader("Explainable AI – Individual Customer Analysis")

    selected_meter = st.selectbox("Select a Customer (Meter ID)", dashboard_df["meter_id"])

    customer_row = dashboard_df[dashboard_df["meter_id"] == selected_meter]
    X_customer = X.loc[customer_row.index]

    rf_model = model.named_estimators_["rf"].named_steps["model"]
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_customer)

    # Normalize SHAP values
    if isinstance(shap_values, list):
        shap_vector = shap_values[1][0]  # class 1
        base_value = explainer.expected_value[1]
    elif shap_values.ndim == 3:
        shap_vector = shap_values[0, :, 1]
        base_value = explainer.expected_value[1]
    else:
        shap_vector = shap_values[0]
        base_value = explainer.expected_value

    explanation = shap.Explanation(
        values=shap_vector,
        base_values=base_value,
        data=X_customer.iloc[0],
        feature_names=X_customer.columns
    )

    st.markdown(
        f"**Meter ID:** `{selected_meter}`  \n"
        f"**Predicted Risk Score:** `{customer_row['risk_score'].values[0]:.2f}`  \n"
        f"**Risk Category:** `{customer_row['risk_category'].values[0]}`"
    )

    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig, bbox_inches="tight")

# -----------------------------------------
# Footer
# -----------------------------------------
st.markdown("---")
st.caption(
    "Interactive dashboard for electricity theft detection. All metrics are read-only and derived from precomputed models and SHAP explainability."
)
