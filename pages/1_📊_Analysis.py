import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
import os

st.set_page_config(page_title="EDA Analysis - CC Fraud Detection", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis Dashboard")
st.markdown("Analyze transaction patterns, feature distributions, and differences between legitimate and fraudulent activities.")

# Cache dataset loading
@st.cache_data
def load_data():
    csv_path = "creditcard.csv"
    if not os.path.exists(csv_path):
        st.error(f"Dataset '{csv_path}' not found in project directory.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df

df_raw = load_data()

if df_raw.empty:
    st.stop()

# Sidebar Filters
st.sidebar.header("🔍 Filter Transactions")

class_filter = st.sidebar.radio(
    "Select Transaction Class:",
    options=["All", "Legitimate Only", "Fraudulent Only"],
    index=0
)

min_amt = float(df_raw["Amount"].min())
max_amt = float(df_raw["Amount"].max())

amt_range = st.sidebar.slider(
    "Transaction Amount ($):",
    min_value=min_amt,
    max_value=max_amt,
    value=(min_amt, max_amt)
)

# Apply filters
df_filtered = df_raw[(df_raw["Amount"] >= amt_range[0]) & (df_raw["Amount"] <= amt_range[1])]

if class_filter == "Legitimate Only":
    df_filtered = df_filtered[df_filtered["Class"] == 0]
elif class_filter == "Fraudulent Only":
    df_filtered = df_filtered[df_filtered["Class"] == 1]

# Summary KPI Header
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    st.metric("Filtered Rows", f"{len(df_filtered):,}")
with col_kpi2:
    legit_cnt = (df_filtered['Class'] == 0).sum()
    st.metric("Legitimate Count", f"{legit_cnt:,}")
with col_kpi3:
    fraud_cnt = (df_filtered['Class'] == 1).sum()
    st.metric("Fraudulent Count", f"{fraud_cnt:,}")
with col_kpi4:
    avg_amt = df_filtered['Amount'].mean() if not df_filtered.empty else 0
    st.metric("Avg Amount ($)", f"${avg_amt:,.2f}")

st.markdown("---")

# Layout Charts in 2 rows of 2 columns
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("1. Target Class Distribution")
    class_counts = df_filtered['Class'].value_counts().reset_index()
    class_counts.columns = ['Class_Code', 'Count']
    class_counts['Class_Name'] = class_counts['Class_Code'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    fig_pie = px.pie(
        class_counts,
        names='Class_Name',
        values='Count',
        color='Class_Name',
        color_discrete_map={'Legitimate': '#22C55E', 'Fraudulent': '#EF4444'},
        hole=0.4,
        title="Ratio of Legitimate vs Fraudulent Transactions"
    )
    fig_pie.update_traces(textinfo='percent+label')
    fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

with row1_col2:
    st.subheader("2. Transaction Amount Distribution (Log Scale)")
    df_plot_amt = df_filtered.copy()
    df_plot_amt['Class_Name'] = df_plot_amt['Class'].map({0: 'Legitimate', 1: 'Fraudulent'})
    df_plot_amt['Amount_Log'] = np.log1p(df_plot_amt['Amount'])
    
    fig_amt = px.histogram(
        df_plot_amt,
        x='Amount_Log',
        color='Class_Name',
        barmode='overlay',
        nbins=40,
        color_discrete_map={'Legitimate': '#3B82F6', 'Fraudulent': '#EF4444'},
        title="Distribution of log(Amount + 1) by Class",
        labels={'Amount_Log': 'Log(Amount + $1)'}
    )
    fig_amt.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_amt, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("3. Top Discriminating Features (Box Plots)")
    # V14, V12, V10, V17 are known strong signals in creditcard dataset
    top_features = ['V14', 'V12', 'V10', 'V17']
    selected_feat = st.selectbox("Select PCA Feature to Compare:", top_features, index=0)
    
    df_plot_feat = df_filtered.copy()
    df_plot_feat['Class_Name'] = df_plot_feat['Class'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    fig_box = px.box(
        df_plot_feat,
        x='Class_Name',
        y=selected_feat,
        color='Class_Name',
        color_discrete_map={'Legitimate': '#10B981', 'Fraudulent': '#F43F5E'},
        points=False,
        title=f"Distribution of {selected_feat} across Classes"
    )
    fig_box.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_box, use_container_width=True)

with row2_col2:
    st.subheader("4. Feature Separation Scatter Plot")
    # Plot V14 vs V12 sample
    sample_n = min(5000, len(df_filtered))
    df_sample = df_filtered.sample(n=sample_n, random_state=42) if len(df_filtered) > 5000 else df_filtered
    df_sample['Class_Name'] = df_sample['Class'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    fig_scatter = px.scatter(
        df_sample,
        x='V14',
        y='V12',
        color='Class_Name',
        color_discrete_map={'Legitimate': '#94A3B8', 'Fraudulent': '#DC2626'},
        opacity=0.7,
        title="V14 vs. V12 Feature Scatter Plot",
        labels={'V14': 'Feature V14', 'V12': 'Feature V12'}
    )
    fig_scatter.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_scatter, use_container_width=True)
