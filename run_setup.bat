@echo off
echo ============================================================
echo Complete Setup: Data Collection + Model Training
echo ============================================================
echo.

echo Step 1: Collecting AQI and weather data...
python collect_data.py
echo.

echo Step 2: Training ML model...
python train_model.py
echo.

echo ============================================================
echo Setup Complete! You can now run: streamlit run app.py
echo ============================================================
pause

