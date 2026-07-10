import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

# Load Dataset
df = pd.read_csv("IBM_HR_Analytics.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Remove unnecessary column
if "EmployeeNumber" in df.columns:
    df.drop("EmployeeNumber", axis=1, inplace=True)
    
columns_to_drop = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]

df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

# Label Encoding
print("\nObject Columns:")
print(df.select_dtypes(include=['object']).columns)
le = LabelEncoder()

le = LabelEncoder()

categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

print("\nData Types:")
print(df.dtypes)

# Features and Target
X = df.drop("Attrition", axis=1)
print("\nTraining Features:")
print(X.columns.tolist())
y = df["Attrition"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save Model
joblib.dump(model, "employee_attrition_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel saved successfully as employee_attrition_model.pkl")

#Accuracy Graph

plt.figure(figsize=(5,5))
plt.bar(["Random Forest"], [accuracy], color="green")
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Employee Attrition Model Accuracy")
plt.savefig("accuracy.png")
plt.close()

#Confusion Matrix

import seaborn as sns

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True,
            fmt="d",
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

#Attrition Count Graph

plt.figure(figsize=(5,5))
df["Attrition"].value_counts().plot(
    kind="bar",
    color=["skyblue","orange"]
)

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.savefig("attrition_distribution.png")
plt.close()

#Feature Importance Graph

importance = model.feature_importances_

feature_importance = pd.Series(
    importance,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10,6))
feature_importance.head(10).plot(kind="barh", color="teal")

plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.savefig("feature_importance.png")
plt.close()