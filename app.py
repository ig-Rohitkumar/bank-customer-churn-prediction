import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv("European_Bank.csv")

# -------------------------------
# Data Cleaning
# -------------------------------

df.drop(["Year", "CustomerId", "Surname"], axis=1, inplace=True)

# -------------------------------
# Feature Engineering
# -------------------------------

df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

df["ProductPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

df["EngagementScore"] = (
    df["IsActiveMember"] *
    df["NumOfProducts"] *
    df["HasCrCard"]
)

df["AgeTenureRatio"] = df["Age"] / (df["Tenure"] + 1)

# -------------------------------
# Encoding
# -------------------------------

df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True
)

# -------------------------------
# Features and Target
# -------------------------------

X = df.drop("Exited", axis=1)

y = df["Exited"]

# -------------------------------
# Train Model
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

rf_model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, roc_auc_score

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

auc = roc_auc_score(
    y_test,
    rf_model.predict_proba(X_test)[:,1]
)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("🏦 Bank Customer Churn Prediction System")

st.write("Enter customer details to predict churn risk.")

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy:.2%}")

with col2:
    st.metric("ROC-AUC", f"{auc:.3f}")

st.subheader("What-If Scenario Simulator")

st.info(
    "Adjust customer attributes below and observe how churn probability changes in real time."
)
# -------------------------------
# User Inputs
# -------------------------------

credit_score = st.slider("Credit Score", 300, 900, 650)

age = st.slider("Age", 18, 92, 35)

tenure = st.slider("Tenure", 0, 10, 5)

balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)

num_products = st.slider("Number of Products", 1, 4, 1)

has_card = st.selectbox("Has Credit Card", [0, 1])

active_member = st.selectbox("Is Active Member", [0, 1])

salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

# -------------------------------
# Feature Engineering on Inputs
# -------------------------------

balance_salary_ratio = balance / (salary + 1)

product_per_tenure = num_products / (tenure + 1)

engagement_score = (
    active_member *
    num_products *
    has_card
)

age_tenure_ratio = age / (tenure + 1)

# -------------------------------
# Geography Encoding
# -------------------------------

geo_germany = 1 if geography == "Germany" else 0

geo_spain = 1 if geography == "Spain" else 0

# -------------------------------
# Gender Encoding
# -------------------------------

gender_male = 1 if gender == "Male" else 0

# -------------------------------
# Prediction Input DataFrame
# -------------------------------

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_card],
    'IsActiveMember': [active_member],
    'EstimatedSalary': [salary],
    'BalanceSalaryRatio': [balance_salary_ratio],
    'ProductPerTenure': [product_per_tenure],
    'EngagementScore': [engagement_score],
    'AgeTenureRatio': [age_tenure_ratio],
    'Geography_Germany': [geo_germany],
    'Geography_Spain': [geo_spain],
    'Gender_Male': [gender_male]
})

# -------------------------------
# Predict Button
# -------------------------------

if st.button("Predict Churn Risk"):

    prediction = rf_model.predict(input_data)[0]

    probability = rf_model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Results")

    st.write(f"Churn Probability: {probability:.2%}")

    if probability < 0.3:
        st.success("Low Risk Customer")

    elif probability < 0.7:
        st.warning("Medium Risk Customer")

    else:
        st.error("High Risk Customer")

# -------------------------------
# Feature Importance Dashboard
# -------------------------------

st.subheader("Top 10 Important Features")

fig, ax = plt.subplots(figsize=(8,5))

ax.barh(
    importance_df.head(10)["Feature"],
    importance_df.head(10)["Importance"]
)

ax.invert_yaxis()

st.pyplot(fig)

st.subheader("Churn Probability Distribution")

probabilities = rf_model.predict_proba(X_test)[:,1]

fig2, ax2 = plt.subplots(figsize=(8,4))

ax2.hist(
    probabilities,
    bins=20
)

ax2.set_xlabel("Churn Probability")
ax2.set_ylabel("Number of Customers")
ax2.set_title("Distribution of Churn Risk Scores")

st.pyplot(fig2)
