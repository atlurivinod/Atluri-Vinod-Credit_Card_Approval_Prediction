# ===========================
# Import Required Libraries
# ===========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# Read the Dataset
# ===========================

application = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

print("Application Dataset")
print(application.head())

print("\nCredit Dataset")
print(credit.head())

# ===========================
# Descriptive Analysis
# ===========================

print("\nApplication Dataset Information")
print(application.info())

print("\nMissing Values")
print(application.isnull().sum())

print("\nDescriptive Statistics")
print(application.describe())

# ===========================
# Data Cleaning and Merging
# ===========================

data = application.merge(credit, on="ID", how="inner")

print("\nMerged Dataset")
print(data.head())

print("\nDataset Shape Before Removing Duplicates")
print(data.shape)

# ===========================
# Drop Duplicate Records
# ===========================

data = data.drop_duplicates()

print("\nDataset Shape After Removing Duplicates")
print(data.shape)

# ===========================
# Univariate Analysis
# ===========================

plt.figure(figsize=(8,5))
plt.hist(data["AMT_INCOME_TOTAL"], bins=20, edgecolor="black")
plt.title("Distribution of Annual Income")
plt.xlabel("Annual Income")
plt.ylabel("Frequency")
plt.show()

# ===========================
# Multivariate Analysis
# ===========================

plt.figure(figsize=(8,5))
plt.scatter(data["AMT_INCOME_TOTAL"], data["CNT_CHILDREN"])
plt.title("Annual Income vs Number of Children")
plt.xlabel("Annual Income")
plt.ylabel("Number of Children")
plt.show()

# ===========================
# Feature Engineering
# ===========================

data["STATUS"] = data["STATUS"].replace({
    "C": 0,
    "X": 0,
    "0": 0,
    "1": 1,
    "2": 1,
    "3": 1,
    "4": 1,
    "5": 1
})

print("\nTarget Variable Distribution")
print(data["STATUS"].value_counts())

# ===========================
# Handling Missing Values
# ===========================

data["OCCUPATION_TYPE"] = data["OCCUPATION_TYPE"].fillna("Unknown")

print("\nMissing Values After Cleaning")
print(data.isnull().sum())

# ===========================
# Handling Categorical Values
# ===========================

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

text_columns = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
]

for column in text_columns:
    data[column] = le.fit_transform(data[column])

print("\nEncoded Dataset")
print(data.head())
# ===========================
# Train Test Split
# ===========================

from sklearn.model_selection import train_test_split

X = data.drop("STATUS", axis=1)
y = data["STATUS"].astype(int)

print("\nFeature Columns")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ===========================
# Import Machine Learning Models
# ===========================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===========================
# Logistic Regression Model
# ===========================

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\n==============================")
print("Logistic Regression Accuracy")
print("==============================")
print("Accuracy :", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# ===========================
# Decision Tree Model
# ===========================

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\n==============================")
print("Decision Tree Accuracy")
print("==============================")
print("Accuracy :", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# ===========================
# Random Forest Model
# ===========================

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n==============================")
print("Random Forest Accuracy")
print("==============================")
print("Accuracy :", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# ===========================
# XGBoost Model
# ===========================

xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

print("\n==============================")
print("XGBoost Accuracy")
print("==============================")
print("Accuracy :", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))
# ===========================
# Model Accuracy Comparison
# ===========================

accuracy_scores = {
    "Logistic Regression": accuracy_score(y_test, lr_pred),
    "Decision Tree": accuracy_score(y_test, dt_pred),
    "Random Forest": accuracy_score(y_test, rf_pred),
    "XGBoost": accuracy_score(y_test, xgb_pred)
}

print("\n==============================")
print("Model Accuracy Comparison")
print("==============================")

for model, score in accuracy_scores.items():
    print(f"{model}: {score:.4f}")

# ===========================
# Best Model Selection
# ===========================

best_model_name = max(accuracy_scores, key=accuracy_scores.get)

print("\nBest Model:", best_model_name)

if best_model_name == "Logistic Regression":
    best_model = lr
elif best_model_name == "Decision Tree":
    best_model = dt
elif best_model_name == "Random Forest":
    best_model = rf
else:
    best_model = xgb

# ===========================
# Save Best Model
# ===========================

import pickle

with open("model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nModel Saved Successfully as model.pkl")

# ===========================
# Project Completed
# ===========================

print("\n==========================================")
print("Credit Card Approval Prediction Completed")
print("==========================================")