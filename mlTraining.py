import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from xgboost import XGBRegressor
import joblib
import os

# Load the processed data
df = pd.read_csv("processed_delivery_data.csv")

# Split into features and target
X = df.drop(columns=['Delivery_Time'])
y = df['Delivery_Time']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0)
}

# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    })

# Display results
results_df = pd.DataFrame(results)
print(results_df)

# Create directory to store models
os.makedirs('saved_models', exist_ok=True)

# ✅ Save all trained models for Streamlit model selection
for name, model in models.items():
    filename = f"saved_models/{name.replace(' ', '_')}.joblib"
    joblib.dump(model, filename)

# Save feature columns used during training
features_path = "saved_models/model_features.pkl"
joblib.dump(X.columns.tolist(), features_path)

# Identify and report best model (by R² score here)
best_model_name = results_df.sort_values(by='R2 Score', ascending=False).iloc[0]['Model']
print(f"\nBest model '{best_model_name}' saved at: saved_models/{best_model_name.replace(' ', '_')}.joblib")
print(f"Model features saved at: {features_path}")