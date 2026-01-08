import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.title("💧 Water Potability Prediction App")

# ---------------------
# Load and clean data
# ---------------------
@st.cache_data
def load_data():
    data = pd.read_csv("water_potability.csv")
    return data.dropna()

data = load_data()

st.subheader("Missing values after dropping NaNs")
st.write(data.isnull().sum())

# -------------------------
# Plotly Histograms (EDA)
# -------------------------
st.subheader("Exploratory Data Analysis")

for col in ["ph","Hardness","Solids","Chloramines","Sulfate","Conductivity",
            "Organic_carbon","Trihalomethanes","Turbidity"]:
    fig = px.histogram(data, x=col, color="Potability",
                       title=f"Factors Affecting Water Quality: {col}")
    fig.update_layout(barmode='group')
    st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Correlation analysis
# ----------------------
st.subheader("Correlation with pH")
correlation = data.corr()
st.write(correlation["ph"].sort_values(ascending=False))

# ----------------------
# Model training
# ----------------------
X = data.drop("Potability", axis=1)
y = data["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=786)

model = RandomForestClassifier(random_state=786)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Feature Importance
st.subheader("Feature Importances")
importances = model.feature_importances_
feature_names = X.columns
feat_importance = pd.Series(importances, index=feature_names)
fig, ax = plt.subplots()
feat_importance.sort_values(ascending=True).plot(kind='barh', color="Skyblue", ax=ax)
st.pyplot(fig)

# ------------------------------
# Predict on new water samples
# ------------------------------
st.subheader("Predict Potability for New Samples")

def predict_sample(sample_dict):
    sample_df = pd.DataFrame([sample_dict])
    prediction = model.predict(sample_df)
    return "POTABLE ✅" if prediction[0] == 1 else "NOT POTABLE ❌"

new_sample = {
    "ph":7.0,"Hardness":200,"Solids":10000,"Chloramines":7.0,
    "Sulfate":300,"Conductivity":400,"Organic_carbon":12.0,
    "Trihalomethanes":75,"Turbidity":3.5
}
st.write("Sample 1:", predict_sample(new_sample))

new_sample_2 = {
    "ph":7.2,"Hardness":180.0,"Solids":15000.0,"Chloramines":6.5,
    "Sulfate":250.0,"Conductivity":420.0,"Organic_carbon":10.0,
    "Trihalomethanes":60.0,"Turbidity":2.5
}
st.write("Sample 2:", predict_sample(new_sample_2))
