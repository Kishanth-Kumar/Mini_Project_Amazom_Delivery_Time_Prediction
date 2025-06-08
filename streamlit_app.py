import streamlit as st
import pandas as pd
import joblib

# Define available models
model_files = {
    "Linear Regression": "saved_models/Linear_Regression.joblib",
    "Random Forest": "saved_models/Random_Forest.joblib",
    "Gradient Boosting": "saved_models/Gradient_Boosting.joblib",
    "SVR": "saved_models/SVR.joblib",
    "XGBoost": "saved_models/XGBoost.joblib"
}

# Load feature list used in training
model_features = joblib.load("saved_models/model_features.pkl")

# Streamlit UI
st.title("📦 Amazon Delivery Time Predictor")
st.markdown("Select a model and provide delivery/order details to estimate the delivery time.")

# Model selection
model_choice = st.selectbox("Choose a model", list(model_files.keys()))

# Load selected model
model_path = model_files[model_choice]
model = joblib.load(model_path)

# Input fields
st.subheader("📝 Input Order Details")
agent_age = st.number_input("Agent Age", min_value=18, max_value=70, value=30)
agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5, 0.1)
distance = st.number_input("Distance (km)", min_value=1.0, value=5.0)

# Categorical inputs
weather = st.selectbox("Weather", ["Sunny", "Sandstorms", "Windy", "Fog","Stormy"])
traffic = st.selectbox("Traffic", ["Low", "Medium", "Jam"])
vehicle = st.selectbox("Vehicle Type", ["scooter", "van"])
area = st.selectbox("Area Type", ["Other","Semi-Urban", "Urban"])
category = st.selectbox("Product Category", ["Grocery", "Electronics", "Clothing", 
                                             "Books","Cosmetics","Home","Jewelry","Kitchen","Outdoors","Pet Supplies","Shoes","Skincare"
                                             "Snacks","Sports","Toys"])

# Create input DataFrame
input_df = pd.DataFrame({
    'Agent_Age': [agent_age],
    'Agent_Rating': [agent_rating],
    'Distance_km': [distance],
    'Weather': [weather],
    'Traffic': [traffic],
    'Vehicle': [vehicle],
    'Area': [area],
    'Category': [category],
})

# One-hot encode categorical variables
input_encoded = pd.get_dummies(input_df)
print(input_encoded)

# Align input with model features
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

# Prediction
if st.button("Predict Delivery Time"):
    prediction = model.predict(input_encoded)[0]
    print(prediction)
    st.success(f"⏱️ Estimated Delivery Time using **{model_choice}**: **{round(prediction, 2)} hours**")