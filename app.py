import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Water Potability Predictor", page_icon="💧")

st.title("💧 Water Potability Prediction System")
st.markdown("### Machine Learning based Water Quality Assessment")

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.sidebar.header("Enter Water Quality Parameters")

ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0)
Hardness = st.sidebar.slider("Hardness", 0.0, 500.0, 200.0)
Solids = st.sidebar.slider("Solids", 0.0, 50000.0, 10000.0)
Chloramines = st.sidebar.slider("Chloramines", 0.0, 15.0, 7.0)
Sulfate = st.sidebar.slider("Sulfate", 0.0, 500.0, 300.0)
Conductivity = st.sidebar.slider("Conductivity", 0.0, 800.0, 400.0)
Organic_carbon = st.sidebar.slider("Organic Carbon", 0.0, 30.0, 12.0)
Trihalomethanes = st.sidebar.slider("Trihalomethanes", 0.0, 120.0, 75.0)
Turbidity = st.sidebar.slider("Turbidity", 0.0, 10.0, 3.5)

input_data = pd.DataFrame([[ph, Hardness, Solids, Chloramines, Sulfate,
                            Conductivity, Organic_carbon, Trihalomethanes, Turbidity]],
                          columns=[
                              "ph","Hardness","Solids","Chloramines","Sulfate",
                              "Conductivity","Organic_carbon","Trihalomethanes","Turbidity"
                          ])

if st.button("🔍 Predict Water Quality"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Water is POTABLE (Safe to drink)")
    else:
        st.error("❌ Water is NOT POTABLE (Unsafe to drink)")

st.markdown("---")
st.caption("Developed by Siddharth Purohit | Data Science & ML Project")
