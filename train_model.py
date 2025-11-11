"""
Train ML model on scraped AQI and weather data
Uses Random Forest (faster and better than single Decision Tree)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

print("🚀 Starting model training on scraped data...")

# ===========================
# Load scraped data
# ===========================
data_file = "combined_aqi_weather_data.csv"

if not os.path.exists(data_file):
    print("⚠️ Combined data file not found. Running data collection...")
    from collect_data import combine_aqi_weather_data
    df = combine_aqi_weather_data()
else:
    df = pd.read_csv(data_file)
    print(f"✅ Loaded data: {df.shape}")

# ===========================
# Prepare features and target
# ===========================
# Core pollutant parameters for accurate AQI prediction
# All pollutants: PM2.5, PM10, NO2, SO2, CO, O3, NH3, Pb
# Plus weather parameters: Temperature, Humidity, Rainfall

feature_cols = ['Temperature', 'Humidity', 'Rainfall']

# Add all core pollutants if available
pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NH3', 'Pb']
for pollutant in pollutant_cols:
    if pollutant in df.columns:
        feature_cols.append(pollutant)

# Ensure all feature columns exist and have data
available_features = [col for col in feature_cols if col in df.columns]

# Fill missing pollutant values with median if available, or reasonable defaults
for col in available_features:
    if col in df.columns and df[col].isna().any():
        if df[col].notna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
        else:
            # Use default values if all are missing
            defaults = {
                'PM2.5': 50, 'PM10': 80, 'NO2': 50, 'SO2': 20,
                'CO': 1.0, 'O3': 50, 'NH3': 15, 'Pb': 0.5,
                'Temperature': 28, 'Humidity': 65, 'Rainfall': 0
            }
            df[col] = df[col].fillna(defaults.get(col, 0))

df = df.dropna(subset=available_features + ['AQI'])

if len(available_features) == 0:
    print("⚠️ No features found! Using default features.")
    available_features = ['Temperature', 'Humidity', 'Rainfall']
    # Create sample data if needed
    if 'Temperature' not in df.columns:
        df['Temperature'] = np.random.uniform(20, 35, len(df))
    if 'Humidity' not in df.columns:
        df['Humidity'] = np.random.uniform(40, 90, len(df))
    if 'Rainfall' not in df.columns:
        df['Rainfall'] = np.random.uniform(0, 10, len(df))

X = df[available_features]
y = df['AQI']

print(f"✅ Features: {available_features}")
print(f"✅ Dataset size: {len(X)} samples")

# ===========================
# Split data
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Training set: {len(X_train)} samples")
print(f"✅ Test set: {len(X_test)} samples")

# ===========================
# Train Random Forest Model
# ===========================
# Random Forest is better than single Decision Tree - faster training, better accuracy
model = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    max_depth=10,      # Limit depth to prevent overfitting
    random_state=42,
    n_jobs=-1          # Use all CPU cores
)

print("\n🌳 Training Random Forest model...")
model.fit(X_train, y_train)

# ===========================
# Evaluate model
# ===========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"   MAE (Mean Absolute Error): {mae:.2f}")
print(f"   RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"   R² Score: {r2:.4f}")

# ===========================
# Feature importance
# ===========================
feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📈 Feature Importance:")
print(feature_importance.to_string(index=False))

# ===========================
# Save model
# ===========================
model_filename = "air_quality_model.pkl"
joblib.dump(model, model_filename)
print(f"\n✅ Model saved as {model_filename}")

# Save feature names for later use
feature_info = {
    'features': available_features,
    'feature_importance': feature_importance.to_dict('records')
}

import json
with open("model_features.json", "w") as f:
    json.dump(feature_info, f, indent=2)

print("✅ Feature information saved to model_features.json")
print("\n🎉 Training complete!")
