import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# ---------------- Load Models ----------------
cnn = load_model("model_cnn_classifier.h5")
xgb = pickle.load(open("model_xgb.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- Feature Names ----------------
feature_names = [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius Error", "Texture Error", "Perimeter Error", "Area Error", "Smoothness Error",
    "Compactness Error", "Concavity Error", "Concave Points Error", "Symmetry Error", "Fractal Dimension Error",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
    "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

# ---------------- Title ----------------
st.markdown("<h1 style='text-align:center;color:#d63384;'>🩺 Breast Cancer Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>1D CNN + XGBoost | Wisconsin Dataset</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Sidebar ----------------
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)
st.sidebar.header("📥 Input Options")
option = st.sidebar.radio("Choose input method:", ["Manual Input", "Upload CSV"])

inputs = []

if option == "Manual Input":
    for name in feature_names:
        val = st.sidebar.number_input(name, value=0.0, step=0.01)
        inputs.append(val)

else:  # CSV Upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)

        if df.empty:
            st.error("❌ CSV file is empty.")
        elif df.shape[1] != 30:
            st.error("❌ CSV must contain exactly 30 features.")
        else:
            inputs = df.iloc[0].values.tolist()
            st.success("✅ CSV loaded successfully")



# ---------------- Main Section ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("ℹ️ Project Information")
    st.write("""
    - **Dataset**: Wisconsin Breast Cancer Dataset  
    - **Deep Learning Model**: 1D CNN  
    - **Classifier**: XGBoost  
    - **Purpose**: Early breast cancer detection
    """)

    predict_btn = st.button("🔍 Predict")

with col2:
    st.subheader("📊 Prediction Result")
if predict_btn and len(inputs) == 30:
    X = np.array(inputs, dtype=np.float32).reshape(1, 30)
    X_scaled = scaler.transform(X)
    X_cnn = X_scaled.reshape(1, 30, 1)

    cnn_features = cnn.predict(X_cnn)
    cnn_features = np.array(cnn_features).reshape(1, -1)

    if cnn_features.shape[1] != 32:
        st.error(f"Feature mismatch: expected 32, got {cnn_features.shape[1]}")
    else:
        prob = xgb.predict_proba(cnn_features)[0]
        result = np.argmax(prob)

        if result == 1:
            st.success(f"🟢 Benign (Confidence: {prob[1]*100:.2f}%)")
            st.balloons()
        else:
            st.error(f"🔴 Malignant (Confidence: {prob[0]*100:.2f}%)")

    
# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:14px;'>Final Year Project | Breast Cancer Detection using AI</p>",
    unsafe_allow_html=True
)
