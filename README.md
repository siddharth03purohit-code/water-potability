# 💧 Water Potability Prediction using Machine Learning

## 📌 Project Overview
Access to safe drinking water is a critical public health concern.  
This project focuses on predicting whether water is **potable (safe to drink)** or **not potable**, based on various physicochemical properties of water.

A **Random Forest Classifier** is used to build a robust machine learning model that analyzes water quality features and predicts potability with reliable performance.

---

## 📊 Dataset Information
The dataset contains multiple water quality parameters and a target variable indicating potability.

**Target Variable:**
- `Potability`
  - `1` → Potable (Safe to drink)
  - `0` → Not Potable (Unsafe)

**Features:**
- `ph`
- `Hardness`
- `Solids`
- `Chloramines`
- `Sulfate`
- `Conductivity`
- `Organic_carbon`
- `Trihalomethanes`
- `Turbidity`

Missing values were handled by removing rows containing NaNs to ensure clean and reliable model training.

---

## 🔍 Exploratory Data Analysis (EDA)
To understand the impact of different features on water quality, interactive visualizations were created using **Plotly**:

- Histograms comparing potable vs non-potable water
- Distribution analysis of key chemical parameters
- Correlation analysis to study feature relationships

These visual insights help in identifying the most influential factors affecting water potability.

---

## 🧠 Machine Learning Approach

### Model Used
- **Random Forest Classifier**

### Steps Involved
1. Data Cleaning & Preprocessing
2. Feature–Target Separation
3. Train-Test Split (80% Train, 20% Test)
4. Model Training
5. Model Evaluation
6. Feature Importance Analysis
7. Prediction on New Water Samples

---

## 📈 Model Evaluation
The model performance is evaluated using:

- **Classification Report**
  - Precision
  - Recall
  - F1-score
- **Confusion Matrix**
- **Feature Importance Visualization**

These metrics help assess how accurately the model predicts water potability.

---

## 🔬 Feature Importance
Random Forest provides insights into which water quality parameters most influence potability predictions, helping in better interpretability of the model.

---

## 🧪 Prediction on New Samples
The trained model can predict potability for new water samples by providing chemical parameter values as input.

Example:
```python
prediction = model.predict(sample_df)
