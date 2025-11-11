# 🚀 Quick Start Guide

## Step-by-Step Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect Data (Web Scraping)
```bash
python collect_data.py
```
This will:
- Scrape AQI data for 15 major Indian cities
- Fetch weather data (temperature, humidity, rainfall)
- Combine and save to `combined_aqi_weather_data.csv`

### 3. Train the ML Model
```bash
python train_model.py
```
This will:
- Load the scraped data
- Train a Random Forest model
- Save model to `air_quality_model.pkl`
- Show performance metrics

### 4. Run the Dashboard
```bash
streamlit run app.py
```

Or use the automated script:
```bash
python run_project.py
```

## 🎯 What You'll See

1. **Dashboard**: Current AQI for your selected city
2. **Major Cities**: AQI comparison across 15+ cities
3. **Predictions**: Future AQI forecasts with graphs
4. **Trends**: Correlation analysis and feature importance

## 📝 Notes

- The web scrapers use BeautifulSoup and may need updates if source sites change
- Weather data uses Open-Meteo API (free, no key needed)
- Model uses Random Forest (fast training, good accuracy)
- All data is scraped in real-time when you run `collect_data.py`

## ⚠️ Troubleshooting

**Import errors?**
```bash
pip install -r requirements.txt
```

**No data?**
```bash
python collect_data.py
```

**Model not found?**
```bash
python train_model.py
```

Enjoy! 🌍

