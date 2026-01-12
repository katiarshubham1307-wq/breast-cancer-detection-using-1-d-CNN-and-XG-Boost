import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# ---------------- Load Models ----------------
cnn = load_model("model_cnn.h5")
xgb = pickle.load(open("model_xgb.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- Title Section ----------------
st.markdown(
    "<h1 style='text-align: center; color: #d63384;'>🩺 Breast Cancer Detection System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Using <b>1D CNN + XGBoost</b> on Wisconsin Dataset</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("🔢 Enter Tumor Features")
st.sidebar.info("Please enter the 30 tumor feature values")

inputs = []
for i in range(30):
    val = st.sidebar.number_input(
        f"Feature {i+1}",
        min_value=0.0,
        value=0.0,
        step=0.01
    )
    inputs.append(val)

# ---------------- Main Area ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Model Information")
    st.write("""
    - **Dataset**: Wisconsin Breast Cancer Dataset  
    - **Deep Learning Model**: 1D Convolutional Neural Network  
    - **Classifier**: XGBoost  
    - **Output**: Benign or Malignant
    """)

    st.subheader("🧪 Prediction")
    predict_btn = st.button("🔍 Predict Cancer Type")

with col2:
    st.subheader("📊 Result")

    if predict_btn:
        X = np.array(inputs).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_cnn = X_scaled.reshape(1, 30, 1)

        features = cnn.predict(X_cnn)
        result = xgb.predict(features)

        if result[0] == 1:
            st.success("🟢 **Benign Tumor**")
            st.balloons()
        else:
            st.error("🔴 **Malignant Tumor**")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>Final Year Project | Breast Cancer Detection</p>",
    unsafe_allow_html=True
)
