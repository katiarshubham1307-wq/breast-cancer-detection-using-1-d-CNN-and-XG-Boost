import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.title("Breast Cancer Detection System")

cnn = load_model("model_cnn.h5")
xgb = pickle.load(open("model_xgb.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

inputs = []
for i in range(30):
    inputs.append(st.number_input(f"Feature {i+1}", value=0.0))

if st.button("Predict"):
    X = np.array(inputs).reshape(1, -1)
    X = scaler.transform(X)
    X = X.reshape(1, 30, 1)

    features = cnn.predict(X)
    result = xgb.predict(features)

    if result[0] == 1:
        st.success("🟢 Benign")
    else:
        st.error("🔴 Malignant")
