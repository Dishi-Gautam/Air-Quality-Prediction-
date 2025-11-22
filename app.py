"""
Air Quality Prediction Dashboard
Professional UI without location prompts or emoji icons
"""
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
from datetime import datetime, timedelta
import os
import time


st.set_page_config(
    page_title="Air Quality Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.4rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }
        .metric-card {
            padding: 1.2rem;
            border-radius: 8px;
            color: white;
            text-align: center;
            box-shadow: 0 3px 6px rgba(0,0,0,0.08);
        }
        .city-card {
            background: #1e1e1e;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_model():
    try:
        model = joblib.load("air_quality_model.pkl")
        return model, True
    except Exception:
        return None, False


@st.cache_data
def load_feature_info():
    try:
        with open("model_features.json", "r") as f:
            return json.load(f)
    except Exception:
        return {"features": ["Temperature", "Humidity", "Rainfall"]}


def auto_refresh_data():
    data_file = "combined_aqi_weather_data.csv"
    if os.path.exists(data_file):
        file_time = os.path.getmtime(data_file)
        current_time = time.time()
        hours_old = (current_time - file_time) / 3600
        if hours_old > 1:
            try:
                from collect_data import combine_aqi_weather_data

                combine_aqi_weather_data()
                st.cache_data.clear()
            except Exception:
                pass


@st.cache_data(ttl=3600)
def load_cities_data():
    if not hasattr(load_cities_data, "_refreshed"):
        auto_refresh_data()
        load_cities_data._refreshed = True

    try:
        if os.path.exists("combined_aqi_weather_data.csv"):
            df = pd.read_csv("combined_aqi_weather_data.csv")
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.sort_values("Date", ascending=False)
            latest_df = df.groupby("City").first().reset_index()
            return latest_df
        else:
            return pd.DataFrame(
                {
                    "City": [
                        "Delhi",
                        "Mumbai",
                        "Kolkata",
                        "Chennai",
                        "Bangalore",
                        "Hyderabad",
                        "Pune",
                        "Ahmedabad",
                        "Jaipur",
                        "Lucknow",
                    ],
                    "AQI": [320, 180, 210, 160, 130, 150, 140, 200, 190, 220],
                    "Temperature": [28, 32, 30, 35, 27, 33, 29, 31, 26, 25],
                    "Humidity": [65, 75, 70, 80, 60, 68, 55, 58, 45, 50],
                    "Rainfall": [0, 2, 1, 0, 0, 0, 0, 0, 0, 0],
                }
            )
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(
            {
                "City": ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore"],
                "AQI": [320, 180, 210, 160, 130],
                "Temperature": [28, 32, 30, 35, 27],
                "Humidity": [65, 75, 70, 80, 60],
                "Rainfall": [0, 2, 1, 0, 0],
            }
        )


model, model_loaded = load_model()
feature_info = load_feature_info()
cities_data = load_cities_data()


def get_aqi_status(aqi):
    if aqi <= 50:
        return "Good", "#00e400", ""
    elif aqi <= 100:
        return "Moderate", "#ffff00", ""
    elif aqi <= 200:
        return "Unhealthy for Sensitive", "#ff7e00", ""
    elif aqi <= 300:
        return "Unhealthy", "#ff0000", ""
    else:
        return "Hazardous", "#7e0023", ""


def predict_future_aqi_with_pollutants(model, base_input_dict, features, days=7):
    if not model_loaded:
        return None

    predictions = []
    dates = []
    for i in range(days):
        date = datetime.now() + timedelta(days=i)
        dates.append(date)
        future_input = base_input_dict.copy()
        if "Temperature" in future_input:
            future_input["Temperature"] = max(0, future_input["Temperature"] + np.random.normal(0, 2))
        if "Humidity" in future_input:
            future_input["Humidity"] = max(0, min(100, future_input["Humidity"] + np.random.normal(0, 5)))
        if "Rainfall" in future_input:
            future_input["Rainfall"] = max(0, future_input["Rainfall"] + np.random.normal(0, 1))

        pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3", "Pb"]
        for pollutant in pollutant_cols:
            if pollutant in future_input:
                variation = np.random.normal(0, future_input[pollutant] * 0.1)
                future_input[pollutant] = max(0, future_input[pollutant] + variation)

        input_df = pd.DataFrame([future_input])
        pred_aqi = model.predict(input_df[features])[0]
        predictions.append(max(0, pred_aqi))

    return pd.DataFrame({"Date": dates, "Predicted_AQI": predictions})


st.markdown('<h1 class="main-header">India Air Quality Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")


st.sidebar.header("Location")
user_city = st.sidebar.selectbox(
    "Select your city:", ["Select..."] + cities_data["City"].tolist()
)

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Major Cities", "Predictions", "Trends & Analysis", "About"],
    index=0,
)


if page == "Dashboard":
    st.header("Current Air Quality Overview")
    if user_city and user_city != "Select...":
        city_data = cities_data[cities_data["City"] == user_city]
        if not city_data.empty:
            row = city_data.iloc[0]
            aqi = row.get("AQI", 0)
            temp = row.get("Temperature", 0)
            humidity = row.get("Humidity", 0)
            rainfall = row.get("Rainfall", 0)
            status, color, _ = get_aqi_status(aqi)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("AQI", f"{int(aqi)}", f"{status}")
            with col2:
                st.metric("Temperature (C)", f"{temp:.1f}")
            with col3:
                st.metric("Humidity (%)", f"{humidity:.1f}")
            with col4:
                st.metric("Rainfall (mm)", f"{rainfall:.1f}")

            st.subheader("Core Pollutant Parameters")
            pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3", "Pb"]
            pollutant_info = {
                "PM2.5": {"unit": "ug/m3", "desc": "Particulate Matter <= 2.5um"},
                "PM10": {"unit": "ug/m3", "desc": "Particulate Matter <= 10um"},
                "NO2": {"unit": "ug/m3", "desc": "Nitrogen Dioxide"},
                "SO2": {"unit": "ug/m3", "desc": "Sulfur Dioxide"},
                "CO": {"unit": "mg/m3", "desc": "Carbon Monoxide"},
                "O3": {"unit": "ug/m3", "desc": "Ozone"},
                "NH3": {"unit": "ug/m3", "desc": "Ammonia"},
                "Pb": {"unit": "ug/m3", "desc": "Lead"},
            }

            poll_cols = st.columns(4)
            for idx, pollutant in enumerate(pollutant_cols):
                col = poll_cols[idx % 4]
                with col:
                    value = row.get(pollutant, 0)
                    info = pollutant_info[pollutant]
                    st.metric(f"{pollutant}", f"{value:.2f} {info['unit']}", help=info["desc"])

            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg, {color} 0%, #1e1e1e 100%);
                            padding:1.2rem;border-radius:10px;text-align:center;color:white;margin:1rem 0;">
                    <h2 style="margin:0">{user_city}</h2>
                    <h3 style="margin:0">{int(aqi)} — {status}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.warning(f"No data available for {user_city}")
    else:
        st.info("Please select a city from the sidebar to view current air quality.")


elif page == "Major Cities":
    st.header("Air Quality in Major Indian Cities")
    cols = st.columns(3)
    for idx, (_, row) in enumerate(cities_data.iterrows()):
        col = cols[idx % 3]
        with col:
            aqi = row.get("AQI", 0)
            status, color, _ = get_aqi_status(aqi)
            st.markdown(
                f"""
                <div class="city-card">
                    <h3 style="margin:0">{row['City']}</h3>
                    <h4 style="margin:0;color:{color};">{int(aqi)} — {status}</h4>
                    <small>Temperature: {row.get('Temperature',0):.1f} C | Humidity: {row.get('Humidity',0):.1f}%</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    fig = px.bar(
        cities_data.sort_values("AQI", ascending=False),
        x="City",
        y="AQI",
        color="AQI",
        color_continuous_scale="RdYlGn_r",
        title="AQI Comparison Across Major Cities",
    )
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)


elif page == "Predictions":
    st.header("AQI Prediction")
    if not model_loaded:
        st.warning("Model not loaded. Please train the model first using train_model.py")
    else:
        st.info("Enter pollutant and weather parameters to predict AQI")
        st.subheader("Weather Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            current_temp = st.number_input("Temperature (C)", 0.0, 50.0, 28.0)
        with col2:
            current_humidity = st.number_input("Humidity (%)", 0.0, 100.0, 65.0)
        with col3:
            current_rainfall = st.number_input("Rainfall (mm)", 0.0, 100.0, 0.0)

        st.subheader("Pollutant Parameters")
        col1, col2 = st.columns(2)
        with col1:
            pm25 = st.number_input("PM2.5 (ug/m3)", 0.0, 500.0, 50.0)
            pm10 = st.number_input("PM10 (ug/m3)", 0.0, 500.0, 80.0)
            no2 = st.number_input("NO2 (ug/m3)", 0.0, 500.0, 50.0)
            so2 = st.number_input("SO2 (ug/m3)", 0.0, 500.0, 20.0)
        with col2:
            co = st.number_input("CO (mg/m3)", 0.0, 50.0, 1.0)
            o3 = st.number_input("O3 (ug/m3)", 0.0, 500.0, 50.0)
            nh3 = st.number_input("NH3 (ug/m3)", 0.0, 100.0, 15.0)
            pb = st.number_input("Pb (ug/m3)", 0.0, 10.0, 0.5)

        prediction_days = st.slider("Days to Predict", 1, 30, 7)
        if st.button("Predict AQI", type="primary"):
            with st.spinner("Predicting AQI..."):
                features = feature_info["features"]
                input_data_dict = {}
                if "Temperature" in features:
                    input_data_dict["Temperature"] = current_temp
                if "Humidity" in features:
                    input_data_dict["Humidity"] = current_humidity
                if "Rainfall" in features:
                    input_data_dict["Rainfall"] = current_rainfall

                pollutant_mapping = {
                    "PM2.5": pm25,
                    "PM10": pm10,
                    "NO2": no2,
                    "SO2": so2,
                    "CO": co,
                    "O3": o3,
                    "NH3": nh3,
                    "Pb": pb,
                }
                for pollutant, value in pollutant_mapping.items():
                    if pollutant in features:
                        input_data_dict[pollutant] = value

                for feat in features:
                    if feat not in input_data_dict:
                        defaults = {
                            "PM2.5": 50,
                            "PM10": 80,
                            "NO2": 50,
                            "SO2": 20,
                            "CO": 1.0,
                            "O3": 50,
                            "NH3": 15,
                            "Pb": 0.5,
                            "Temperature": 28,
                            "Humidity": 65,
                            "Rainfall": 0,
                        }
                        input_data_dict[feat] = defaults.get(feat, 0)

                input_df = pd.DataFrame([input_data_dict])
                predicted_aqi = model.predict(input_df[features])[0]
                status, color, _ = get_aqi_status(predicted_aqi)
                st.markdown(
                    f"""
                    <div style="background:linear-gradient(135deg, {color} 0%, #1e1e1e 100%);
                                padding:1rem;border-radius:10px;text-align:center;color:white;margin:1rem 0;">
                        <h3 style="margin:0">Predicted AQI</h3>
                        <h2 style="margin:0">{int(predicted_aqi)} — {status}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                predictions_df = predict_future_aqi_with_pollutants(model, input_data_dict, features, prediction_days)
                if predictions_df is not None:
                    st.subheader("Predicted AQI Trends")
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=predictions_df["Date"],
                            y=predictions_df["Predicted_AQI"],
                            mode="lines+markers",
                            name="Predicted AQI",
                        )
                    )
                    fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
                    fig.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate")
                    fig.add_hline(y=200, line_dash="dash", line_color="orange", annotation_text="Unhealthy")
                    fig.update_layout(title="Future AQI Predictions", xaxis_title="Date", yaxis_title="Predicted AQI", template="plotly_dark", height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Predicted AQI", f"{predictions_df['Predicted_AQI'].mean():.1f}")
                    with col2:
                        st.metric("Min AQI", f"{predictions_df['Predicted_AQI'].min():.1f}")
                    with col3:
                        st.metric("Max AQI", f"{predictions_df['Predicted_AQI'].max():.1f}")
                    with col4:
                        trend = (
                            "Increasing"
                            if predictions_df["Predicted_AQI"].iloc[-1] > predictions_df["Predicted_AQI"].iloc[0]
                            else "Decreasing"
                        )
                        st.metric("Trend", trend)


elif page == "Trends & Analysis":
    st.header("Air Quality Trends & Analysis")
    st.caption("This page analyzes actual scraped data from cities (not predictions)")
    if len(cities_data) > 0:
        st.subheader("Correlation Analysis")
        numeric_cols = [
            "AQI",
            "Temperature",
            "Humidity",
            "Rainfall",
            "PM2.5",
            "PM10",
            "NO2",
            "SO2",
            "CO",
            "O3",
            "NH3",
            "Pb",
        ]
        available_numeric = [col for col in numeric_cols if col in cities_data.columns]
        if len(available_numeric) > 1:
            corr_data = cities_data[available_numeric].corr()
            fig = px.imshow(corr_data, text_auto=True, aspect="auto", color_continuous_scale="RdBu", title="Correlation Matrix: AQI vs Parameters")
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Pollutant Analysis")
        pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3", "Pb"]
        available_pollutants = [p for p in pollutant_cols if p in cities_data.columns]
        if available_pollutants:
            fig_poll = go.Figure()
            for pollutant in available_pollutants[:4]:
                fig_poll.add_trace(go.Bar(x=cities_data["City"], y=cities_data[pollutant], name=pollutant, text=cities_data[pollutant].round(1), textposition="auto"))
            fig_poll.update_layout(title="Pollutant Levels by City (Top 4)", xaxis_title="City", yaxis_title="Concentration", template="plotly_dark", height=400, barmode="group")
            st.plotly_chart(fig_poll, use_container_width=True)

        st.subheader("AQI vs Weather Parameters")
        if "Temperature" in cities_data.columns:
            fig1 = px.scatter(cities_data, x="Temperature", y="AQI", color="City", size="AQI", hover_data=["Humidity", "Rainfall"] + available_pollutants[:3], title="AQI vs Temperature")
            fig1.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig1, use_container_width=True)
        if "Humidity" in cities_data.columns:
            fig2 = px.scatter(cities_data, x="Humidity", y="AQI", color="City", size="AQI", hover_data=["Temperature", "Rainfall"] + available_pollutants[:3], title="AQI vs Humidity")
            fig2.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig2, use_container_width=True)

        if model_loaded and "feature_importance" in feature_info:
            st.subheader("Model Feature Importance")
            importance_df = pd.DataFrame(feature_info["feature_importance"])
            fig = px.bar(importance_df, x="Feature", y="Importance", color="Importance", color_continuous_scale="Viridis", title="Feature Importance")
            fig.update_layout(template="plotly_dark", height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)


else:
    st.header("About This Project")
    st.markdown(
        """
        ### India Air Quality Prediction Dashboard

        This dashboard provides air quality monitoring and predictions for major Indian cities.

        Features:
        - Location-based AQI display
        - Real-time AQI for major cities
        - Future AQI predictions using ML models
        - Interactive graphs and trend analysis

        Technology Stack:
        - Frontend: Streamlit
        - ML Model: Random Forest Regressor
        - Visualization: Plotly
        - Web Scraping: BeautifulSoup
        - Data Processing: Pandas, NumPy
        """
    )
    if model_loaded:
        st.success("ML Model is loaded and ready for predictions")
    else:
        st.warning("ML Model not found. Run train_model.py to train the model.")
