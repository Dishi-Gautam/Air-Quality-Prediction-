"""
Complete setup script: Collect data and train model with all pollutants
"""
import subprocess
import sys
import os

print("="*60)
print("🚀 Complete Setup: Data Collection + Model Training")
print("="*60)

# Step 1: Collect data
print("\n📊 Step 1: Collecting AQI and weather data...")
print("-" * 60)
try:
    from collect_data import combine_aqi_weather_data
    df = combine_aqi_weather_data()
    print(f"✅ Data collected: {df.shape}")
    print(f"✅ Columns: {list(df.columns)}")
except Exception as e:
    print(f"❌ Error collecting data: {e}")
    sys.exit(1)

# Step 2: Train model
print("\n🤖 Step 2: Training ML model with all pollutants...")
print("-" * 60)
try:
    from train_model import *
    # The train_model.py will run when imported
    print("✅ Model training completed!")
except Exception as e:
    print(f"❌ Error training model: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ Setup Complete! You can now run: streamlit run app.py")
print("="*60)

