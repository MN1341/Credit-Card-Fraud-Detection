import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Live Fraud Prediction - CC Fraud Detection", page_icon="🎯", layout="wide")

st.title("🎯 Live Credit Card Fraud Risk Scoring")
st.markdown("Pass real transaction parameters into the trained **XGBoost & SMOTE Pipeline** to obtain instant fraud predictions and probability scores.")

# Cache model loading using st.cache_resource
@st.cache_resource
def load_model():
    model_path = "Fraud_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in root directory.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Cache sample dataset loading using st.cache_data
@st.cache_data
def load_sample_transactions():
    sample_path = "sample_transactions.csv"
    if os.path.exists(sample_path):
        df_sample = pd.read_csv(sample_path)
        return df_sample
    elif os.path.exists("creditcard.csv"):
        df_raw = pd.read_csv("creditcard.csv")
        fraud_samples = df_raw[df_raw["Class"] == 1].head(100)
        legit_samples = df_raw[df_raw["Class"] == 0].head(100)
        return pd.concat([legit_samples, fraud_samples]).reset_index(drop=True)
    else:
        return pd.DataFrame()

model = load_model()
samples_df = load_sample_transactions()

if model is None:
    st.stop()

# Explanation Banner regarding PCA features
st.info("""
💡 **Why Preset Samples & Controls?**  
The dataset features `V1` through `V28` represent **anonymized numerical PCA features** (e.g. `V14 = -4.32`, `V12 = -2.18`) generated from confidentiality transformations. 
Because typing 28 abstract PCA floats manually is error-prone, you can **pick a representative transaction template below**, adjust operational metrics like **Transaction Amount**, and fine-tune key features to evaluate model behavior.
""")

st.markdown("---")

# Feature Input Section
st.subheader("1. Select & Configure Transaction Input")

col_sel1, col_sel2 = st.columns([2, 1])

with col_sel1:
    if not samples_df.empty:
        # Create user-friendly selectbox labels
        sample_labels = []
        for idx, row in samples_df.iterrows():
            actual_type = "Fraudulent (Class 1)" if row.get("Class", 0) == 1 else "Legitimate (Class 0)"
            sample_labels.append(f"Sample #{idx + 1} — Actual: {actual_type} | Amount: ${row.get('Amount', 0.0):.2f}")
        
        selected_index = st.selectbox(
            "Select Baseline Transaction Record:",
            options=range(len(sample_labels)),
            format_func=lambda i: sample_labels[i],
            index=0
        )
        base_record = samples_df.iloc[selected_index].copy()
    else:
        st.warning("No sample dataset available. Initializing zero baseline.")
        # Default feature columns
        feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        base_record = pd.Series({col: 0.0 for col in feature_cols})

with col_sel2:
    actual_label = base_record.get("Class", 0)
    badge_color = "red" if actual_label == 1 else "green"
    badge_text = "🚨 Fraudulent Sample" if actual_label == 1 else "✅ Legitimate Sample"
    st.markdown(f"**Selected Sample Ground Truth:**")
    st.markdown(f"### :{badge_color}[{badge_text}]")

st.markdown("### 2. Operational Parameters & Fine-Tuning")

param_col1, param_col2 = st.columns(2)

with param_col1:
    current_amount = float(base_record.get("Amount", 10.0))
    input_amount = st.number_input(
        "Transaction Amount ($ USD):",
        min_value=0.0,
        max_value=25000.0,
        value=current_amount,
        step=10.0,
        help="Modify the purchase value to test risk sensitivity."
    )

with param_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Modifying key PCA features below allows testing how extreme anomaly signals alter the predicted risk score.")

# Drawer for Fine-tuning PCA Features
with st.expander("🛠️ Advanced: Inspect & Fine-Tune PCA Features (V1 - V28)", expanded=False):
    st.caption("You can adjust individual PCA principal components if desired.")
    pca_cols = st.columns(4)
    modified_pca = {}
    
    for i in range(1, 29):
        col_name = f"V{i}"
        col_target = pca_cols[(i - 1) % 4]
        default_val = float(base_record.get(col_name, 0.0))
        modified_pca[col_name] = col_target.number_input(
            f"{col_name}:",
            value=default_val,
            format="%.4f",
            key=f"pca_{col_name}"
        )

# Construct final feature DataFrame for prediction
expected_features = [f"V{i}" for i in range(1, 29)] + ["Amount"]

input_data = {}
for col in expected_features:
    if col == "Amount":
        input_data[col] = input_amount
    else:
        # Use fine-tuned value if adjusted in expander, else baseline record value
        input_data[col] = modified_pca.get(col, float(base_record.get(col, 0.0)))

input_df = pd.DataFrame([input_data])

st.markdown("---")

# Predict Button & Result Display
st.subheader("3. Model Prediction Execution")

if st.button("🚀 Run Fraud Risk Prediction", type="primary", use_container_width=True):
    try:
        # Execute prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        fraud_prob = prediction_proba[1] * 100.0
        legit_prob = prediction_proba[0] * 100.0
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("#### Prediction Outcome:")
            if prediction == 1:
                st.error("### 🚨 HIGH RISK: Fraudulent Transaction Detected")
                st.markdown(f"The model flags this transaction as **Fraudulent** with **{fraud_prob:.2f}% risk probability**.")
            else:
                st.success("### ✅ LOW RISK: Legitimate Transaction")
                st.markdown(f"The model classifies this transaction as **Legitimate** with **{legit_prob:.2f}% confidence**.")
        
        with res_col2:
            st.markdown("#### Estimated Fraud Probability Gauge:")
            st.progress(float(fraud_prob / 100.0))
            
            pcol1, pcol2 = st.columns(2)
            pcol1.metric("Legitimate Probability", f"{legit_prob:.1f}%")
            pcol2.metric("Fraud Probability", f"{fraud_prob:.1f}%")
            
        with st.expander("🔍 View Prepared Feature Vector (Input Data)", expanded=False):
            st.dataframe(input_df)
            
    except Exception as err:
        st.error(f"Error during model prediction: {err}")
