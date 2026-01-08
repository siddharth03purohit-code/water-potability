import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

#---------------------
# Load and clean data
#---------------------

data = pd.read_csv("water_potability.csv")
data = data.dropna() # you could use imputation here instead of dropping 

#------------------------
# check for missing values
#------------------------
print("Missing values after dropping NaNs:")
print(data.isnull().sum())

#-------------------------
# plotly Histogram (EDA)
#-------------------------
figure = px.histogram(
    data,
    x="ph",
    color="Potability",
    title="Factors Affecting Water Quality: PH"
)
figure.update_layout(barmode='group')
figure.show()

figure = px.histogram(
    data,
    x="Hardness",
    color="Potability",
    title="Factors Affecting Water Quality: Hardness"
)
figure.update_layout(barmode='group')
figure.show()

figure = px.histogram(
        data,
        x="Solids",
        color="Potability",
        title=f"Factors Affecting Water Quality:Solids "
    )
figure.update_layout(barmode='group')
figure.show()


figure = px.histogram(
        data,
        x= "Chloramines",
        color="Potability",
        title=f"Factors Affecting Water Quality: Chloramines "
    )
figure.update_layout(barmode='group')
figure.show()


figure = px.histogram(
        data,
        x= "Sulfate",
        color="Potability",
        title=f"Factors Affecting Water Quality: Sulfate"
    )
figure.update_layout(barmode='group')
figure.show()

figure = px.histogram(
     data,
     x="Conductivity",
     color="Potability",
     title=f"Factors Affecting Water Quality: Conductivity"
)
figure.update_layout(barmode='group')
figure.show()

figure = px.histogram(
     data,
     x="Organic_carbon",
     color="Potability",
     title=f"Factors Affecting Water Quality: Organic_carbon"
)
figure.update_layout(barmode='group')
figure.show()

figure = px.histogram(
        data,
        x= "Trihalomethanes",
        color="Potability",
        title=f"Factors Affecting Water Quality: Trihalomethanes"
    )
figure.update_layout(barmode='group')
figure.show()

figure = px.histogram(
        data,
        x= "Turbidity",
        color="Potability",
        title=f"Factors Affecting Water Quality: Turbidity"
)
figure.update_layout(barmode='group')
figure.show()


# ----------------------
# Correlation analysis
# ----------------------

correlation = data.corr()
print("\nCorrelation with PH")
print(correlation["ph"].sort_values(ascending=False))

# Step 1: Split features and target
X = data.drop("Potability", axis=1)
y = data["Potability"]

# Step 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=786)

# Step 3: Train a Random Forest model

model = RandomForestClassifier(random_state=786)
model.fit(X_train,y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# step 5: Evaluate the model 

print("\nClassification Report:")
print(classification_report(y_test,y_pred))

# confusion Matrix

conf_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues")
plt.title(" CONFUSION MATRIX ")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 6: Feature Importance (like PyCaret's feature rank)

importances = model.feature_importances_
feature_names = X.columns
feat_importance = pd.Series(importances, index=feature_names)
feat_importance.sort_values(ascending=True).plot(kind='barh',color="Skyblue")
plt.title("Feature importances (Random Forest)")
plt.xlabel("importance score")
plt.show()

# ------------------------------
# Predict on a new water sample
# ------------------------------

new_sample = {
    "ph":7.0,
    "Hardness":200,
    "Solids": 10000,
    "Chloramines":7.0,
    "Sulfate":300,
    "Conductivity": 400,
    "Organic_carbon": 12.0,
    "Trihalomethanes": 75,
    "Turbidity": 3.5
}

# Convert to DataFrame
sample_df_1 = pd.DataFrame([new_sample])

# Predict potability
prediction = model.predict(sample_df_1)

# Show result
print("\nNew Sample Prediction:")
if prediction[0] == 1:
    print(" The water is predicted to be POTABLE (safe to drink).")
else:
    print(" The water is predicted to be NOT POTABLE (unsafe to drink).")
    
new_sample_1 = {
    
    "ph": 7.2,                  # Neutral/slightly basic (normal drinking water pH)
    "Hardness": 180.0,          # Within acceptable range
    "Solids": 15000.0,          # Not too high
    "Chloramines": 6.5,         # Within EPA standards (4–10 ppm)
    "Sulfate": 250.0,           # Within WHO acceptable limits (< 500 mg/L)
    "Conductivity": 420.0,      # Typical for drinking water
    "Organic_carbon": 10.0,     # Normal range
    "Trihalomethanes": 60.0,    # Below limit (80 μg/L by EPA)
    "Turbidity": 2.5            # Low turbidity = clear water
}

# convert to DataFrame
sample_df_2 = pd.DataFrame([new_sample_1])

# prediction potability
prediction = model.predict(sample_df_2)

# Show result
print("\nNew Sample Prediction:")
if prediction[0] == 1:
    print("The water is predicted to be POTABLE (safe to drink).")
else:
    print("The water is predicted to be NOT POTABLE (unsafe to drink).")