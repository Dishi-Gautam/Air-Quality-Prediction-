"""
Web scraper for weather data (temperature, humidity, rainfall)
Uses OpenWeatherMap API or web scraping as fallback
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import json

# Major Indian cities with coordinates
MAJOR_CITIES_COORDS = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "Kanpur": {"lat": 26.4499, "lon": 80.3319},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Indore": {"lat": 22.7196, "lon": 75.8577},
    "Thane": {"lat": 19.2183, "lon": 72.9781},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126}
}

def scrape_weather_wunderground(city_name):
    """Scrape weather data from wunderground or similar"""
    try:
        # Using a free weather API alternative
        # For demonstration, we'll use Open-Meteo (free, no API key needed)
        coords = MAJOR_CITIES_COORDS.get(city_name)
        if not coords:
            return None
        
        # Open-Meteo API (free, no key required)
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': coords['lat'],
            'longitude': coords['lon'],
            'current': 'temperature_2m,relative_humidity_2m,precipitation',
            'timezone': 'Asia/Kolkata'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data.get('current', {})
            
            return {
                'City': city_name,
                'Temperature': current.get('temperature_2m'),
                'Humidity': current.get('relative_humidity_2m'),
                'Rainfall': current.get('precipitation', 0),
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Time': datetime.now().strftime('%H:%M:%S')
            }
    except Exception as e:
        print(f"Error getting weather for {city_name}: {e}")
    
    return None

def get_weather_data():
    """Main function to get weather data for all cities"""
    print("🌤️ Fetching weather data for major Indian cities...")
    weather_data = []
    
    for city_name in MAJOR_CITIES_COORDS.keys():
        try:
            data = scrape_weather_wunderground(city_name)
            if data:
                weather_data.append(data)
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"Error for {city_name}: {e}")
    
    df = pd.DataFrame(weather_data)
    return df

if __name__ == "__main__":
    df = get_weather_data()
    print(f"\n✅ Fetched weather data for {len(df)} cities")
    print(df.head())
    df.to_csv("scraped_weather_data.csv", index=False)
    print("\n💾 Data saved to scraped_weather_data.csv")

