import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling cards and metrics
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E1E2F 0%, #2D2D44 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .info-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Description
st.markdown("<div class='main-header'>💳 Credit Card Fraud Detection System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Machine Learning Web Application powered by XGBoost & Imbalanced-Learn Pipeline</div>",
    unsafe_allow_html=True
)

st.markdown("""
### 📌 Project Overview
This application provides an interactive interface for detecting fraudulent credit card transactions in real time. 
Built on a dataset of **284,807 transactions**, the underlying machine learning model uses a specialized **scikit-learn & imbalanced-learn pipeline** 
incorporating `RobustScaler`, `SMOTE` oversampling, and an `XGBoost Classifier`. The app is designed for risk analysts to explore data distributions 
and simulate live transaction fraud scoring.
""")

st.markdown("---")

# Key Metrics Row
st.subheader("📊 Key System & Model Performance Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>Total Dataset Rows</div>
        <div class='metric-value'>284,807</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>Fraud Ratio</div>
        <div class='metric-value'>0.17%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>Model Accuracy</div>
        <div class='metric-value'>99.9%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>Fraud Precision</div>
        <div class='metric-value'>62.9%</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>Fraud Recall / F1</div>
        <div class='metric-value'>76.8% / 69.2%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Instructions Cards
st.subheader("🗺️ App Navigation")

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    st.info("""
    #### 📊 1. Analysis (EDA Dashboard)
    Explore interactive visualizations of credit card transactions:
    - **Class Balance & Fraud Proportion**: Visualizing the extreme class imbalance.
    - **Transaction Amount Distributions**: Log-scale comparison between normal and fraudulent purchases.
    - **Key PCA Feature Distributions**: Deep dive into top discriminating features (`V14`, `V12`, `V10`, `V17`).
    - Dynamic sidebar filters for Transaction Class and Amount range.
    """)

with nav_col2:
    st.success("""
    #### 🎯 2. Prediction (Live Scoring)
    Test the trained ML pipeline on live or preset transaction data:
    - **Preset Sample Selector**: Load real legitimate or fraudulent transaction test cases.
    - **Interactive Operational Controls**: Adjust transaction `Amount` and fine-tune feature parameters.
    - **Real-time Inference**: View model prediction (Legitimate vs. Fraudulent) along with estimated Fraud Risk Probability.
    """)

st.markdown("---")
st.caption("Developed with Streamlit • Model: XGBoost Pipeline with SMOTE & RobustScaler")
