# 🌍 India Air Quality Prediction Dashboard

A comprehensive air quality monitoring and prediction system for major Indian cities using web scraping, machine learning, and interactive visualizations.

## ✨ Features

- 📍 **Location-based AQI Display**: View current air quality based on your location
- 🏙️ **Major Cities AQI**: Real-time air quality data for 15+ major Indian cities
- 🔮 **Future Predictions**: ML-powered AQI predictions based on weather parameters
- 📊 **Interactive Graphs**: Beautiful visualizations using Plotly
- 🌡️ **Weather Integration**: Temperature, Humidity, and Rainfall parameters
- 🤖 **ML Model**: Random Forest Regressor for fast and accurate predictions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Data

Run the data collection script to scrape AQI and weather data:

```bash
python collect_data.py
```

This will:
- Scrape AQI data for major Indian cities
- Fetch weather data (temperature, humidity, rainfall)
- Combine and save the data to `combined_aqi_weather_data.csv`

### 3. Train the Model

Train the ML model on the scraped data:

```bash
python train_model.py
```

This will:
- Load the combined AQI and weather data
- Train a Random Forest Regressor model
- Save the model to `air_quality_model.pkl`
- Display model performance metrics

### 4. Run the Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
air_quality_project/
├── app.py                      # Main Streamlit dashboard
├── train_model.py              # ML model training script
├── collect_data.py             # Data collection orchestrator
├── scrape_aqi_data.py          # AQI data web scraper
├── scrape_weather_data.py      # Weather data scraper
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── air_quality_model.pkl       # Trained ML model (generated)
├── model_features.json         # Model feature info (generated)
├── combined_aqi_weather_data.csv  # Combined dataset (generated)
└── data/                       # Historical data folder
```

## 🎯 Usage

### Dashboard Pages

1. **🏠 Dashboard**: View current AQI for selected city with health recommendations
2. **🏙️ Major Cities**: Compare AQI across major Indian cities
3. **🔮 Predictions**: Predict future AQI based on weather conditions
4. **📊 Trends & Analysis**: Correlation analysis and feature importance
5. **ℹ️ About**: Project information and model details

### Making Predictions

1. Navigate to the **🔮 Predictions** page
2. Enter current weather conditions:
   - Temperature (°C)
   - Humidity (%)
   - Rainfall (mm)
3. Select number of days to predict
4. Click "Predict Future AQI" to see forecasts

## 🤖 ML Model Details

- **Algorithm**: Random Forest Regressor
- **Features**: Temperature, Humidity, Rainfall (and optionally PM2.5, PM10)
- **Target**: Air Quality Index (AQI)
- **Advantages**: 
  - Fast training time
  - Good accuracy
  - Handles non-linear relationships
  - Feature importance analysis

## 📊 Data Sources

- **AQI Data**: Scraped from public air quality sources
- **Weather Data**: Open-Meteo API (free, no API key required)

## 🔧 Customization

### Adding More Cities

Edit `MAJOR_CITIES` in `scrape_aqi_data.py` and `MAJOR_CITIES_COORDS` in `scrape_weather_data.py`

### Model Parameters

Modify `train_model.py` to adjust:
- Number of trees (`n_estimators`)
- Max depth (`max_depth`)
- Test/train split ratio

## 📝 Notes

- The web scrapers may need updates if source websites change
- For production use, consider using official APIs with API keys
- Model accuracy depends on data quality and quantity

## 🛠️ Troubleshooting

**Model not found error:**
- Run `python train_model.py` first

**No data available:**
- Run `python collect_data.py` to collect fresh data

**Import errors:**
- Install all dependencies: `pip install -r requirements.txt`

## 📄 License

This project is for educational purposes.

## 👤 Author

Developed for air quality awareness and prediction in India.

---

**Made with ❤️ for better air quality awareness**

