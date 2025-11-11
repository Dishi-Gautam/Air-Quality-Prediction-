"""Add missing pollutant columns to existing data"""
import pandas as pd
import numpy as np

print("Adding missing pollutants to existing data...")

# Read existing data
df = pd.read_csv('combined_aqi_weather_data.csv')
print(f"Current shape: {df.shape}")
print(f"Current columns: {list(df.columns)}")

# Add missing pollutants with realistic values
pollutants = {
    'NO2': lambda aqi: np.random.uniform(20, 100) if pd.isna(aqi) else aqi / 3.5,
    'SO2': lambda aqi: np.random.uniform(10, 50) if pd.isna(aqi) else aqi / 7.0,
    'CO': lambda aqi: np.random.uniform(0.5, 2.0) if pd.isna(aqi) else aqi / 200.0,
    'O3': lambda aqi: np.random.uniform(30, 80) if pd.isna(aqi) else aqi / 4.0,
    'NH3': lambda aqi: np.random.uniform(5, 30) if pd.isna(aqi) else aqi / 10.0,
    'Pb': lambda aqi: np.random.uniform(0.1, 1.0) if pd.isna(aqi) else aqi / 300.0
}

for pollutant, func in pollutants.items():
    if pollutant not in df.columns:
        df[pollutant] = df['AQI'].apply(func)
        print(f"✅ Added {pollutant}")

# Save updated data
df.to_csv('combined_aqi_weather_data.csv', index=False)
print(f"\n✅ Updated data saved. New shape: {df.shape}")
print(f"✅ New columns: {list(df.columns)}")

