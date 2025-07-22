import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Load model and training metadata
# =========================
model = joblib.load("final_xgb_model.pkl")

# Get feature names from the pipeline
numeric_features = list(model.named_steps['preprocessor'].transformers_[0][2])
categorical_features = list(model.named_steps['preprocessor'].transformers_[1][2])
all_features = numeric_features + categorical_features

# Load the original training dataset to derive defaults (IMPORTANT)
# Make sure you keep your original train CSV in the same folder
train_df = pd.read_csv("train.csv")  # or adjust the path

# Build default values: mode for categorical, median for numeric
default_values = {}
for col in all_features:
    if col in train_df.columns:
        if col in numeric_features:
            default_values[col] = train_df[col].median()
        else:
            default_values[col] = train_df[col].mode()[0]
    else:
        # Fallback if column wasn't in train.csv (rare case)
        default_values[col] = 0 if col in numeric_features else "None"

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Prediction")
st.write("Enter a few key details to estimate the house price. Other features will use typical defaults from the dataset.")

# Main user-controlled inputs
overall_qual = st.slider("Overall Quality (1=Poor, 10=Excellent)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sqft)", 500, 5000, 1500, step=50)
garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 1)
total_bsmt_sf = st.number_input("Total Basement Area (sqft)", 0, 3000, 800, step=50)

# Build input row using defaults
input_dict = default_values.copy()
input_dict["OverallQual"] = overall_qual
input_dict["GrLivArea"] = gr_liv_area
input_dict["GarageCars"] = garage_cars
input_dict["TotalBsmtSF"] = total_bsmt_sf

input_data = pd.DataFrame([input_dict])

# =========================
# Prediction
# =========================
if st.button("Predict Price"):
    log_price = model.predict(input_data)[0]
    price = np.expm1(log_price)  # undo log1p transform (if applied)
    st.success(f"Estimated House Price: **${price:,.0f}**")

