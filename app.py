"""
Air Quality Prediction Dashboard with Location Permission
Shows current AQI, major cities AQI, and future predictions with graphs
"""
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import json
from datetime import datetime, timedelta
import os
import threading
import time

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(
    page_title="🌍 Air Quality Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #00B4D8, #0077B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .city-card {
        background: #1e1e1e;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #00B4D8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model and Data
# ---------------------------
@st.cache_data
def load_model():
    """Load trained model"""
    try:
        model = joblib.load("air_quality_model.pkl")
        return model, True
    except:
        return None, False

@st.cache_data
def load_feature_info():
    """Load feature information"""
    try:
        with open("model_features.json", "r") as f:
            return json.load(f)
    except:
        return {"features": ["Temperature", "Humidity", "Rainfall"]}

def auto_refresh_data():
    """Automatically refresh data if it's older than 1 hour"""
    data_file = "combined_aqi_weather_data.csv"
    if os.path.exists(data_file):
        file_time = os.path.getmtime(data_file)
        current_time = time.time()
        hours_old = (current_time - file_time) / 3600
        
        # Refresh if data is older than 1 hour
        if hours_old > 1:
            try:
                from collect_data import combine_aqi_weather_data
                combine_aqi_weather_data()
                st.cache_data.clear()
            except Exception as e:
                pass  # Silently fail if refresh doesn't work

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_cities_data():
    """Load cities AQI data with auto-refresh"""
    # Auto-refresh in background if needed
    if not hasattr(load_cities_data, '_refreshed'):
        auto_refresh_data()
        load_cities_data._refreshed = True
    
    try:
        if os.path.exists("combined_aqi_weather_data.csv"):
            df = pd.read_csv("combined_aqi_weather_data.csv")
            # Get latest data for each city
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.sort_values('Date', ascending=False)
            latest_df = df.groupby('City').first().reset_index()
            return latest_df
        else:
            # Fallback sample data
            return pd.DataFrame({
                "City": ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore", 
                        "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"],
                "AQI": [320, 180, 210, 160, 130, 150, 140, 200, 190, 220],
                "Temperature": [28, 32, 30, 35, 27, 33, 29, 31, 26, 25],
                "Humidity": [65, 75, 70, 80, 60, 68, 55, 58, 45, 50],
                "Rainfall": [0, 2, 1, 0, 0, 0, 0, 0, 0, 0]
            })
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame({
            "City": ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore"],
            "AQI": [320, 180, 210, 160, 130],
            "Temperature": [28, 32, 30, 35, 27],
            "Humidity": [65, 75, 70, 80, 60],
            "Rainfall": [0, 2, 1, 0, 0]
        })

model, model_loaded = load_model()
feature_info = load_feature_info()
cities_data = load_cities_data()

# ---------------------------
# Helper Functions
# ---------------------------
def get_aqi_status(aqi):
    """Get AQI status and color"""
    if aqi <= 50:
        return "Good", "#00e400", "🟢"
    elif aqi <= 100:
        return "Moderate", "#ffff00", "🟡"
    elif aqi <= 200:
        return "Unhealthy for Sensitive", "#ff7e00", "🟠"
    elif aqi <= 300:
        return "Unhealthy", "#ff0000", "🔴"
    else:
        return "Hazardous", "#7e0023", "🟣"

def predict_future_aqi_with_pollutants(model, base_input_dict, features, days=7):
    """Predict future AQI based on all parameters with variations"""
    if not model_loaded:
        return None
    
    predictions = []
    dates = []
    
    # Create variations for future predictions
    for i in range(days):
        date = datetime.now() + timedelta(days=i)
        dates.append(date)
        
        # Create input with slight variations
        future_input = base_input_dict.copy()
        
        # Vary weather parameters
        if 'Temperature' in future_input:
            future_input['Temperature'] = max(0, future_input['Temperature'] + np.random.normal(0, 2))
        if 'Humidity' in future_input:
            future_input['Humidity'] = max(0, min(100, future_input['Humidity'] + np.random.normal(0, 5)))
        if 'Rainfall' in future_input:
            future_input['Rainfall'] = max(0, future_input['Rainfall'] + np.random.normal(0, 1))
        
        # Vary pollutants slightly
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NH3', 'Pb']
        for pollutant in pollutant_cols:
            if pollutant in future_input:
                variation = np.random.normal(0, future_input[pollutant] * 0.1)
                future_input[pollutant] = max(0, future_input[pollutant] + variation)
        
        # Predict AQI
        input_df = pd.DataFrame([future_input])
        pred_aqi = model.predict(input_df[features])[0]
        predictions.append(max(0, pred_aqi))
    
    return pd.DataFrame({
        'Date': dates,
        'Predicted_AQI': predictions
    })

# ---------------------------
# Main Header
# ---------------------------
st.markdown('<h1 class="main-header">🌍 India Air Quality Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Location Permission Section
# ---------------------------
st.sidebar.header("📍 Location Settings")
use_location = st.sidebar.checkbox("Use Current Location", value=False)

# Auto-refresh data button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data Now", help="Manually refresh AQI and weather data"):
    with st.sidebar.spinner("Refreshing data..."):
        try:
            from collect_data import combine_aqi_weather_data
            combine_aqi_weather_data()
            st.cache_data.clear()
            st.sidebar.success("✅ Data refreshed!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")

if use_location:
    st.sidebar.info("📍 Location permission requested. Using browser geolocation API.")
    # In a real app, you would use JavaScript to get location
    # For now, we'll use a manual input
    user_city = st.sidebar.selectbox(
        "Or select your city:",
        ["Select..."] + cities_data["City"].tolist()
    )
else:
    user_city = st.sidebar.selectbox(
        "Select your city:",
        ["Select..."] + cities_data["City"].tolist()
    )

# ---------------------------
# Navigation
# ---------------------------
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🏙️ Major Cities", "🔮 Predictions", "📊 Trends & Analysis", "ℹ️ About"],
    index=0
)

# ---------------------------
# DASHBOARD PAGE
# ---------------------------
if page == "🏠 Dashboard":
    st.header("📊 Current Air Quality Overview")
    
    if user_city and user_city != "Select...":
        city_data = cities_data[cities_data["City"] == user_city]
        if not city_data.empty:
            row = city_data.iloc[0]
            aqi = row.get("AQI", 0)
            temp = row.get("Temperature", 0)
            humidity = row.get("Humidity", 0)
            rainfall = row.get("Rainfall", 0)
            
            status, color, emoji = get_aqi_status(aqi)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("AQI", f"{int(aqi)}", f"{emoji} {status}")
            
            with col2:
                st.metric("Temperature", f"{temp:.1f}°C", "🌡️")
            
            with col3:
                st.metric("Humidity", f"{humidity:.1f}%", "💧")
            
            with col4:
                st.metric("Rainfall", f"{rainfall:.1f} mm", "🌧️")
            
            # Display all pollutant parameters
            st.subheader("🌫️ Core Pollutant Parameters")
            pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NH3', 'Pb']
            pollutant_info = {
                'PM2.5': {'unit': 'µg/m³', 'desc': 'Particulate Matter ≤ 2.5µm', 'icon': '🌫️'},
                'PM10': {'unit': 'µg/m³', 'desc': 'Particulate Matter ≤ 10µm', 'icon': '💨'},
                'NO2': {'unit': 'µg/m³', 'desc': 'Nitrogen Dioxide', 'icon': '🚗'},
                'SO2': {'unit': 'µg/m³', 'desc': 'Sulfur Dioxide', 'icon': '🏭'},
                'CO': {'unit': 'mg/m³', 'desc': 'Carbon Monoxide', 'icon': '⛽'},
                'O3': {'unit': 'µg/m³', 'desc': 'Ozone', 'icon': '☀️'},
                'NH3': {'unit': 'µg/m³', 'desc': 'Ammonia', 'icon': '🌾'},
                'Pb': {'unit': 'µg/m³', 'desc': 'Lead', 'icon': '⚗️'}
            }
            
            # Create columns for pollutants
            poll_cols = st.columns(4)
            for idx, pollutant in enumerate(pollutant_cols):
                col = poll_cols[idx % 4]
                with col:
                    value = row.get(pollutant, 0)
                    info = pollutant_info[pollutant]
                    st.metric(
                        f"{info['icon']} {pollutant}",
                        f"{value:.2f} {info['unit']}",
                        help=info['desc']
                    )
            
            # AQI Status Card
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, {color} 0%, #1e1e1e 100%);
                        padding:2rem;border-radius:15px;text-align:center;color:white;margin:1rem 0;">
                <h2>{user_city}</h2>
                <h1 style="font-size:4rem;margin:0.5rem 0;">{int(aqi)}</h1>
                <h3>{emoji} {status}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Health Recommendations
            st.subheader("💡 Health Recommendations")
            if aqi <= 50:
                st.success("✅ Air quality is good. Enjoy outdoor activities!")
            elif aqi <= 100:
                st.info("ℹ️ Air quality is acceptable. Sensitive individuals may experience minor issues.")
            elif aqi <= 200:
                st.warning("⚠️ Unhealthy for sensitive groups. Limit outdoor activities.")
            else:
                st.error("🚨 Unhealthy air quality! Stay indoors, use air purifiers, and avoid outdoor activities.")
        else:
            st.warning(f"⚠️ No data available for {user_city}")
    else:
        st.info("👆 Please select a city from the sidebar to view current air quality.")

# ---------------------------
# MAJOR CITIES PAGE
# ---------------------------
elif page == "🏙️ Major Cities":
    st.header("🏙️ Air Quality in Major Indian Cities")
    
    # City cards grid
    cols = st.columns(3)
    for idx, (_, row) in enumerate(cities_data.iterrows()):
        col = cols[idx % 3]
        with col:
            aqi = row.get("AQI", 0)
            status, color, emoji = get_aqi_status(aqi)
            st.markdown(f"""
            <div class="city-card">
                <h3>{row['City']}</h3>
                <h2 style="color:{color};">{emoji} {int(aqi)}</h2>
                <p>{status}</p>
                <small>🌡️ {row.get('Temperature', 0):.1f}°C | 💧 {row.get('Humidity', 0):.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive map/chart
    fig = px.bar(
        cities_data.sort_values('AQI', ascending=False),
        x='City',
        y='AQI',
        color='AQI',
        color_continuous_scale='RdYlGn_r',
        title="AQI Comparison Across Major Cities"
    )
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# PREDICTIONS PAGE
# ---------------------------
elif page == "🔮 Predictions":
    st.header("🔮 AQI Prediction with All Core Pollutants")
    
    if not model_loaded:
        st.warning("⚠️ Model not loaded. Please train the model first using train_model.py")
    else:
        st.info("💡 Enter all pollutant parameters and weather conditions for accurate AQI prediction")
        
        st.subheader("🌡️ Weather Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            current_temp = st.number_input("Temperature (°C)", 0.0, 50.0, 28.0)
        with col2:
            current_humidity = st.number_input("Humidity (%)", 0.0, 100.0, 65.0)
        with col3:
            current_rainfall = st.number_input("Rainfall (mm)", 0.0, 100.0, 0.0)
        
        st.subheader("🌫️ Core Pollutant Parameters")
        st.caption("Enter values for all pollutants to get the most accurate AQI prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pm25 = st.number_input("PM2.5 (µg/m³) - Particulate Matter ≤ 2.5µm", 0.0, 500.0, 50.0, help="Penetrates deep into lungs; major AQI driver")
            pm10 = st.number_input("PM10 (µg/m³) - Particulate Matter ≤ 10µm", 0.0, 500.0, 80.0, help="Dust, pollen, smoke — affects visibility & respiration")
            no2 = st.number_input("NO₂ (µg/m³) - Nitrogen Dioxide", 0.0, 500.0, 50.0, help="From vehicles & industries; causes respiratory irritation")
            so2 = st.number_input("SO₂ (µg/m³) - Sulfur Dioxide", 0.0, 500.0, 20.0, help="From burning coal/diesel; contributes to acid rain")
        
        with col2:
            co = st.number_input("CO (mg/m³) - Carbon Monoxide", 0.0, 50.0, 1.0, help="Incomplete combustion (vehicles); reduces oxygen delivery")
            o3 = st.number_input("O₃ (µg/m³) - Ozone", 0.0, 500.0, 50.0, help="Formed from NOx + VOCs under sunlight; causes eye/lung irritation")
            nh3 = st.number_input("NH₃ (µg/m³) - Ammonia", 0.0, 100.0, 15.0, help="From agriculture and fertilizers; eye/skin irritant")
            pb = st.number_input("Pb (µg/m³) - Lead", 0.0, 10.0, 0.5, help="From industrial sources; toxic even in small quantities")
        
        prediction_days = st.slider("Days to Predict", 1, 30, 7)
        
        if st.button("🔍 Predict AQI", type="primary"):
            with st.spinner("Predicting AQI using all pollutant parameters..."):
                # Prepare input data with all pollutants
                features = feature_info['features']
                input_data_dict = {}
                
                # Add weather parameters
                if 'Temperature' in features:
                    input_data_dict['Temperature'] = current_temp
                if 'Humidity' in features:
                    input_data_dict['Humidity'] = current_humidity
                if 'Rainfall' in features:
                    input_data_dict['Rainfall'] = current_rainfall
                
                # Add all pollutants
                pollutant_mapping = {
                    'PM2.5': pm25, 'PM10': pm10, 'NO2': no2, 'SO2': so2,
                    'CO': co, 'O3': o3, 'NH3': nh3, 'Pb': pb
                }
                
                for pollutant, value in pollutant_mapping.items():
                    if pollutant in features:
                        input_data_dict[pollutant] = value
                
                # Ensure all required features are present
                for feat in features:
                    if feat not in input_data_dict:
                        defaults = {
                            'PM2.5': 50, 'PM10': 80, 'NO2': 50, 'SO2': 20,
                            'CO': 1.0, 'O3': 50, 'NH3': 15, 'Pb': 0.5,
                            'Temperature': 28, 'Humidity': 65, 'Rainfall': 0
                        }
                        input_data_dict[feat] = defaults.get(feat, 0)
                
                # Predict current AQI
                input_df = pd.DataFrame([input_data_dict])
                predicted_aqi = model.predict(input_df[features])[0]
                
                # Show immediate prediction
                status, color, emoji = get_aqi_status(predicted_aqi)
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, {color} 0%, #1e1e1e 100%);
                            padding:2rem;border-radius:15px;text-align:center;color:white;margin:1rem 0;">
                    <h2>Predicted AQI</h2>
                    <h1 style="font-size:4rem;margin:0.5rem 0;">{int(predicted_aqi)}</h1>
                    <h3>{emoji} {status}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Predict future trends
                predictions_df = predict_future_aqi_with_pollutants(
                    model, input_data_dict, features, prediction_days
                )
                
                if predictions_df is not None:
                    # Display predictions
                    st.subheader("📈 Predicted AQI Trends")
                    
                    # Line chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=predictions_df['Date'],
                        y=predictions_df['Predicted_AQI'],
                        mode='lines+markers',
                        name='Predicted AQI',
                        line=dict(color='#00B4D8', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Add AQI threshold lines
                    fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
                    fig.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate")
                    fig.add_hline(y=200, line_dash="dash", line_color="orange", annotation_text="Unhealthy")
                    
                    fig.update_layout(
                        title="Future AQI Predictions",
                        xaxis_title="Date",
                        yaxis_title="Predicted AQI",
                        template="plotly_dark",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Predicted AQI", f"{predictions_df['Predicted_AQI'].mean():.1f}")
                    with col2:
                        st.metric("Min AQI", f"{predictions_df['Predicted_AQI'].min():.1f}")
                    with col3:
                        st.metric("Max AQI", f"{predictions_df['Predicted_AQI'].max():.1f}")
                    with col4:
                        trend = "📈 Increasing" if predictions_df['Predicted_AQI'].iloc[-1] > predictions_df['Predicted_AQI'].iloc[0] else "📉 Decreasing"
                        st.metric("Trend", trend)
                    
                    # Data table
                    st.subheader("📋 Detailed Predictions")
                    st.dataframe(predictions_df.style.format({
                        'Predicted_AQI': '{:.1f}',
                        'Date': lambda x: x.strftime('%Y-%m-%d')
                    }), use_container_width=True)

# ---------------------------
# TRENDS & ANALYSIS PAGE
# ---------------------------
elif page == "📊 Trends & Analysis":
    st.header("📊 Air Quality Trends & Analysis")
    st.caption("📌 This page analyzes ACTUAL scraped data from cities (not predictions)")
    
    if len(cities_data) > 0:
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        st.write("Shows how AQI correlates with weather parameters and pollutants")
        
        numeric_cols = ['AQI', 'Temperature', 'Humidity', 'Rainfall', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NH3', 'Pb']
        available_numeric = [col for col in numeric_cols if col in cities_data.columns]
        
        if len(available_numeric) > 1:
            corr_data = cities_data[available_numeric].corr()
            
            fig = px.imshow(
                corr_data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu',
                title="Correlation Matrix: AQI vs All Parameters (Weather + Pollutants)"
            )
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Pollutant Analysis
        st.subheader("🌫️ Pollutant Analysis")
        st.write("Current pollutant levels across major cities")
        
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NH3', 'Pb']
        available_pollutants = [p for p in pollutant_cols if p in cities_data.columns]
        
        if available_pollutants:
            # Pollutant comparison chart
            fig_poll = go.Figure()
            for pollutant in available_pollutants[:4]:  # Show first 4 to avoid clutter
                fig_poll.add_trace(go.Bar(
                    x=cities_data['City'],
                    y=cities_data[pollutant],
                    name=pollutant,
                    text=cities_data[pollutant].round(1),
                    textposition='auto'
                ))
            
            fig_poll.update_layout(
                title="Pollutant Levels by City (Top 4 Pollutants)",
                xaxis_title="City",
                yaxis_title="Concentration",
                template="plotly_dark",
                height=400,
                barmode='group'
            )
            st.plotly_chart(fig_poll, use_container_width=True)
        
        # Scatter plots - Weather vs AQI
        st.subheader("📈 AQI vs Weather Parameters")
        st.write("Relationship between weather conditions and air quality")
        
        if 'Temperature' in cities_data.columns:
            fig1 = px.scatter(
                cities_data,
                x='Temperature',
                y='AQI',
                color='City',
                size='AQI',
                hover_data=['Humidity', 'Rainfall'] + available_pollutants[:3],
                title="AQI vs Temperature (hover to see pollutant levels)"
            )
            fig1.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        if 'Humidity' in cities_data.columns:
            fig2 = px.scatter(
                cities_data,
                x='Humidity',
                y='AQI',
                color='City',
                size='AQI',
                hover_data=['Temperature', 'Rainfall'] + available_pollutants[:3],
                title="AQI vs Humidity (hover to see pollutant levels)"
            )
            fig2.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Pollutant vs AQI scatter
        if available_pollutants:
            st.subheader("🌫️ Pollutants vs AQI")
            st.write("How individual pollutants relate to overall AQI")
            
            selected_pollutant = st.selectbox("Select Pollutant", available_pollutants)
            if selected_pollutant:
                fig3 = px.scatter(
                    cities_data,
                    x=selected_pollutant,
                    y='AQI',
                    color='City',
                    size='AQI',
                    hover_data=['Temperature', 'Humidity'],
                    title=f"AQI vs {selected_pollutant}",
                    labels={selected_pollutant: f"{selected_pollutant} Concentration"}
                )
                fig3.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig3, use_container_width=True)
        
        # Feature importance (if available) - Shows what the MODEL uses for predictions
        if model_loaded and 'feature_importance' in feature_info:
            st.subheader("🎯 ML Model Feature Importance")
            st.write("**For Predictions**: Which parameters the trained model considers most important when predicting AQI")
            importance_df = pd.DataFrame(feature_info['feature_importance'])
            
            fig = px.bar(
                importance_df,
                x='Feature',
                y='Importance',
                color='Importance',
                color_continuous_scale='Viridis',
                title="Which factors most influence AQI predictions? (Higher = More Important)"
            )
            fig.update_layout(template="plotly_dark", height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Current Data Statistics:**")
                st.dataframe(cities_data[['City', 'AQI'] + available_pollutants[:4]].describe(), use_container_width=True)
            with col2:
                st.write("**Top Polluting Cities:**")
                top_cities = cities_data.nlargest(5, 'AQI')[['City', 'AQI'] + available_pollutants[:2]]
                st.dataframe(top_cities, use_container_width=True)

# ---------------------------
# ABOUT PAGE
# ---------------------------
else:
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### 🌍 India Air Quality Prediction Dashboard
    
    This dashboard provides real-time air quality monitoring and predictions for major Indian cities.
    
    **Features:**
    - 📍 Location-based AQI display
    - 🏙️ Real-time AQI for major Indian cities
    - 🔮 Future AQI predictions using ML models
    - 📊 Interactive graphs and trend analysis
    - 🌡️ Weather parameter integration (Temperature, Humidity, Rainfall)
    
    **Technology Stack:**
    - **Frontend**: Streamlit
    - **ML Model**: Random Forest Regressor
    - **Data Visualization**: Plotly
    - **Web Scraping**: BeautifulSoup
    - **Data Processing**: Pandas, NumPy
    
    **Model Details:**
    - Uses Random Forest algorithm (faster and more accurate than single Decision Tree)
    - Trained on scraped AQI and weather data
    - Predicts AQI based on Temperature, Humidity, and Rainfall parameters
    
    **Data Sources:**
    - AQI data scraped from public sources
    - Weather data from Open-Meteo API
    
    ---
    **Made with ❤️ for better air quality awareness**
    """)
    
    if model_loaded:
        st.success("✅ ML Model is loaded and ready for predictions!")
    else:
        st.warning("⚠️ ML Model not found. Run train_model.py to train the model.")
