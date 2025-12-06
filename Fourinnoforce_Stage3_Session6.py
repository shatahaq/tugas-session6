"""
IoT Machine Learning Dashboard
==============================

This module provides a Streamlit-based dashboard for real-time monitoring and
prediction of IoT sensor data (Temperature & Humidity). It integrates MQTT for
data ingestion and a pre-trained Machine Learning model for anomaly detection.

Author: Fourinnoforce Team
Date: 2025-12-06
"""

import json
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from paho.mqtt import client as mqtt
from plotly.subplots import make_subplots


# ============================================
# CONFIGURATION
# ============================================
@dataclass(frozen=True)
class AppConfig:
    """Application configuration constants."""
    MQTT_BROKER: str = "broker.hivemq.com"
    MQTT_PORT: int = 1883
    SENSOR_TOPIC: str = "fourinnoforce/class/session5/sensor"
    OUTPUT_TOPIC: str = "fourinnoforce/class/session5/output"
    LOG_FILE: str = "realtime_predictions.csv"
    MAX_DATA_POINTS: int = 50
    MODEL_FILE: str = "iot_temp_model.pkl"
    TEMP_THRESHOLD: float = 30.0


CONFIG = AppConfig()


# ============================================
# UTILITIES & STYLING
# ============================================
def inject_custom_css() -> None:
    """Injects custom CSS for modern UI styling."""
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* Global Styles */
        .main { font-family: 'Inter', sans-serif; }
        
        /* Animated Header */
        .animated-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            animation: fadeInDown 0.8s ease-out;
        }
        
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animated-header h1 {
            margin: 0; font-size: 2.5rem; font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .animated-header p {
            margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95;
        }
        
        /* Status Card */
        .status-card {
            background: white; padding: 1.5rem; border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: all 0.3s ease; animation: fadeIn 0.6s ease-out;
        }
        .status-card:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }
        
        /* Prediction Card */
        .prediction-card {
            padding: 2.5rem; border-radius: 20px; text-align: center;
            font-size: 2rem; font-weight: 700; color: white;
            box-shadow: 0 8px 30px rgba(0,0,0,0.2);
            animation: pulse 2s infinite; transition: all 0.3s ease;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        .prediction-card:hover { transform: scale(1.1) !important; }
        
        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem; border-radius: 15px; text-align: center;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1); transition: all 0.3s ease;
        }
        .metric-card:hover { transform: translateY(-8px) rotate(2deg); box-shadow: 0 10px 30px rgba(0,0,0,0.15); }
        
        .metric-value {
            font-size: 3rem; font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin: 0.5rem 0;
        }
        
        .metric-label { font-size: 1.1rem; color: #555; font-weight: 600; }
        
        /* Status Badge */
        .status-badge {
            display: inline-block; padding: 0.5rem 1.5rem; border-radius: 25px;
            font-weight: 600; font-size: 0.9rem; animation: slideInRight 0.5s ease-out;
        }
        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .status-connected { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; }
        .status-disconnected { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); color: white; }
        
        /* General */
        * { transition: all 0.3s ease; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #f5f7fa 0%, #c3cfe2 100%); }
        .stButton>button { border-radius: 10px; font-weight: 600; transition: all 0.3s ease; }
        .stButton>button:hover { transform: scale(1.05); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        .dataframe { border-radius: 10px; overflow: hidden; }
        </style>
    """, unsafe_allow_html=True)


# ============================================
# MODEL MANAGEMENT
# ============================================
@st.cache_resource
def load_model(model_path: str) -> Optional[Any]:
    """
    Loads the pre-trained Machine Learning model from a pickle file.

    Args:
        model_path (str): Path to the .pkl model file.

    Returns:
        Optional[Any]: The loaded model object or None if loading fails.
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ ML Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found! Please upload '{model_path}'")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


def predict_temperature(
    model: Optional[Any], 
    temperature: float, 
    humidity: Optional[float] = None
) -> Tuple[str, str, str, str]:
    """
    Predicts the status based on temperature and humidity using the ML model
    or a rule-based fallback.

    Args:
        model: The loaded ML model.
        temperature (float): Temperature value.
        humidity (Optional[float]): Humidity value.

    Returns:
        Tuple[str, str, str, str]: (Label, Color, Emoji, Command)
    """
    # Rule-based fallback constants
    HOT_LABEL = ("Panas", "#FF4B4B", "🔥", "BUZZER_ON")
    NORMAL_LABEL = ("Normal", "#00C851", "✅", "BUZZER_OFF")

    # Fallback if model is not loaded
    if model is None:
        return HOT_LABEL if temperature > CONFIG.TEMP_THRESHOLD else NORMAL_LABEL

    try:
        # Prepare input features
        if humidity is not None:
            features = np.array([[temperature, humidity]])
        else:
            features = np.array([[temperature]])

        # Predict
        prediction = model.predict(features)[0]

        # Map prediction to output
        if prediction == 1 or prediction == "Panas":
            return HOT_LABEL
        else:
            return NORMAL_LABEL

    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        # Fallback on error
        return HOT_LABEL if temperature > CONFIG.TEMP_THRESHOLD else NORMAL_LABEL


# ============================================
# DATA LOGGING
# ============================================
def log_to_csv(data: Dict[str, Any]) -> None:
    """
    Logs a single data point to the CSV file.

    Args:
        data (Dict[str, Any]): Dictionary containing timestamp, temp, hum, and prediction.
    """
    try:
        df_new = pd.DataFrame([{
            'timestamp': data['timestamp'],
            'temperature': data['temp'],
            'humidity': data['hum'],
            'prediction': data['prediction']
        }])

        header = not os.path.exists(CONFIG.LOG_FILE)
        mode = 'w' if header else 'a'
        
        df_new.to_csv(CONFIG.LOG_FILE, mode=mode, header=header, index=False)
        
    except Exception as e:
        print(f"⚠️ Error logging to CSV: {e}")


# ============================================
# MQTT HANDLERS
# ============================================
def on_connect(client: mqtt.Client, userdata: Any, flags: Any, rc: int) -> None:
    """Callback for MQTT connection."""
    if rc == 0:
        st.session_state.connected = True
        client.subscribe(CONFIG.SENSOR_TOPIC)
        print(f"✅ Connected to MQTT Broker: {CONFIG.MQTT_BROKER}")
        print(f"📥 Subscribed to: {CONFIG.SENSOR_TOPIC}")
    else:
        st.session_state.connected = False
        print(f"❌ Connection failed with code {rc}")


def on_message(client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
    """Callback for MQTT message reception."""
    try:
        payload = json.loads(msg.payload.decode())
        temp = float(payload.get('temp', 0))
        hum = float(payload.get('hum', 0))
        timestamp = payload.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Get model from global scope (loaded via st.cache_resource)
        # Note: In a pure class-based structure, this would be passed differently,
        # but for Streamlit callbacks, accessing the cached resource is acceptable.
        
        prediction, color, emoji, command = predict_temperature(ml_model, temp, hum)

        # Update Session State safely
        if 'message_count' in st.session_state:
            st.session_state.message_count += 1
            if prediction == "Panas":
                st.session_state.total_panas += 1
            else:
                st.session_state.total_normal += 1

            data_point = {
                'timestamp': timestamp,
                'temp': temp,
                'hum': hum,
                'prediction': prediction,
                'color': color,
                'emoji': emoji
            }

            st.session_state.data_buffer.append(data_point)
            st.session_state.latest_data = data_point
            st.session_state.last_prediction = prediction
            st.session_state.last_command = command

        # Send feedback and log
        client.publish(CONFIG.OUTPUT_TOPIC, command)
        
        log_data = {
            'timestamp': timestamp,
            'temp': temp,
            'hum': hum,
            'prediction': prediction,
            'color': color,
            'emoji': emoji
        }
        log_to_csv(log_data)

        msg_count = st.session_state.get('message_count', 0)
        print(f"📊 [{msg_count}] Temp: {temp}°C | {emoji} {prediction} | Sent: {command}")

    except Exception as e:
        print(f"⚠️ Error processing message: {e}")


def setup_mqtt() -> None:
    """Initializes and connects the MQTT client."""
    if st.session_state.mqtt_client is None:
        client_id = f"Streamlit_Dashboard_{int(time.time())}"
        client = mqtt.Client(client_id=client_id)
        client.on_connect = on_connect
        client.on_message = on_message

        try:
            client.connect(CONFIG.MQTT_BROKER, CONFIG.MQTT_PORT, 60)
            client.loop_start()
            st.session_state.mqtt_client = client
            # Allow time for connection to establish
            time.sleep(2)
        except Exception as e:
            st.error(f"❌ MQTT Connection Error: {e}")


# ============================================
# UI COMPONENTS
# ============================================
def render_sidebar(model: Optional[Any]) -> None:
    """Renders the sidebar configuration and stats."""
    with st.sidebar:
        st.markdown("### ⚙️ System Configuration")
        st.markdown(f"""
            <div class="status-card">
                <strong>MQTT Broker:</strong><br>{CONFIG.MQTT_BROKER}<br><br>
                <strong>Sensor Topic:</strong><br>{CONFIG.SENSOR_TOPIC}<br><br>
                <strong>Output Topic:</strong><br>{CONFIG.OUTPUT_TOPIC}
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Live Statistics")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", st.session_state.message_count)
        with col2:
            status = st.session_state.last_command.replace("BUZZER_", "") if st.session_state.last_command else "-"
            st.metric("Status", status)

        st.metric("🔥 Panas", st.session_state.total_panas)
        st.metric("✅ Normal", st.session_state.total_normal)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎯 ML Model Info")

        if model is not None:
            st.success("✅ Model Loaded")
            st.write(f"**Model Type:** {type(model).__name__}")
        else:
            st.warning("⚠️ Using Rule-Based")

        st.info("**Classification:**")
        st.write(f"- 🔥 **Panas**: Temp > {CONFIG.TEMP_THRESHOLD}°C")
        st.write(f"- ✅ **Normal**: Temp ≤ {CONFIG.TEMP_THRESHOLD}°C")
        
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔄 Reset Statistics", use_container_width=True):
            st.session_state.message_count = 0
            st.session_state.total_panas = 0
            st.session_state.total_normal = 0
            st.rerun()

        if os.path.exists(CONFIG.LOG_FILE):
            with open(CONFIG.LOG_FILE, 'rb') as f:
                st.download_button(
                    label="📥 Download CSV Log",
                    data=f,
                    file_name=CONFIG.LOG_FILE,
                    mime='text/csv',
                    use_container_width=True
                )


def render_dashboard() -> None:
    """Renders the main dashboard content."""
    # Header
    st.markdown("""
        <div class="animated-header">
            <h1>🌡️ IoT + Machine Learning Dashboard</h1>
            <p>Real-time Temperature Monitoring & Prediction System</p>
        </div>
    """, unsafe_allow_html=True)

    # Connection Status
    _, col_status, _ = st.columns([1, 2, 1])
    with col_status:
        if st.session_state.connected:
            st.markdown("""
                <div style="text-align: center;">
                    <span class="status-badge status-connected">
                        ✅ MQTT Connected to broker.hivemq.com
                    </span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="text-align: center;">
                    <span class="status-badge status-disconnected">
                        ❌ MQTT Disconnected
                    </span>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Content
    if st.session_state.latest_data:
        latest = st.session_state.latest_data
        
        # Cards
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
                <div class="prediction-card" style="background: linear-gradient(135deg, {latest['color']} 0%, {latest['color']}dd 100%);">
                    {latest['emoji']} {latest['prediction']}
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🌡️ Temperature</div>
                    <div class="metric-value">{latest['temp']:.1f}°C</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">💧 Humidity</div>
                    <div class="metric-value">{latest['hum']:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Charts
        if len(st.session_state.data_buffer) > 0:
            render_charts()
            render_data_table()
            
    else:
        render_waiting_state()


def render_charts() -> None:
    """Renders the Plotly charts."""
    df = pd.DataFrame(list(st.session_state.data_buffer))
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('🌡️ Temperature Monitoring', '💧 Humidity Monitoring'),
        vertical_spacing=0.12
    )
    
    # Temperature Trace
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], y=df['temp'],
            mode='lines+markers', name='Temperature',
            line=dict(color='#FF6B6B', width=3, shape='spline'),
            marker=dict(size=8, color='#FF6B6B', line=dict(width=2, color='white')),
            fill='tozeroy', fillcolor='rgba(255, 107, 107, 0.2)'
        ), row=1, col=1
    )
    
    # Threshold Line
    fig.add_hline(
        y=CONFIG.TEMP_THRESHOLD, 
        line_dash="dash", line_color="red", line_width=2,
        annotation_text=f"Threshold: {CONFIG.TEMP_THRESHOLD}°C",
        annotation_position="right", row=1, col=1
    )
    
    # Humidity Trace
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], y=df['hum'],
            mode='lines+markers', name='Humidity',
            line=dict(color='#4ECDC4', width=3, shape='spline'),
            marker=dict(size=8, color='#4ECDC4', line=dict(width=2, color='white')),
            fill='tozeroy', fillcolor='rgba(78, 205, 196, 0.2)'
        ), row=2, col=1
    )
    
    # Layout Updates
    fig.update_xaxes(title_text="Time", row=2, col=1, showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1, showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(title_text="Humidity (%)", row=2, col=1, showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    
    fig.update_layout(
        height=650, showlegend=False, hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_data_table() -> None:
    """Renders the recent data table."""
    st.markdown("### 📋 Recent Data Log")
    df = pd.DataFrame(list(st.session_state.data_buffer))
    display_df = df[['timestamp', 'temp', 'hum', 'prediction']].tail(10).sort_values('timestamp', ascending=False)
    display_df.columns = ['⏰ Timestamp', '🌡️ Temperature (°C)', '💧 Humidity (%)', '🎯 Prediction']
    
    def highlight_prediction(row):
        color = '#ffebee' if row['🎯 Prediction'] == 'Panas' else '#e8f5e9'
        return [f'background-color: {color}'] * len(row)
    
    styled_df = display_df.style.apply(highlight_prediction, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def render_waiting_state() -> None:
    """Renders the empty state when waiting for data."""
    st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 5rem; margin-bottom: 1rem;">⏳</div>
            <h2 style="color: #667eea;">Waiting for Sensor Data...</h2>
            <p style="font-size: 1.2rem; color: #888; margin-top: 1rem;">
                Make sure your ESP32 is connected and publishing data
            </p>
            <div style="margin-top: 2rem; text-align: left; display: inline-block;">
                <p>✅ ESP32 connected to WiFi</p>
                <p>✅ Publishing to MQTT topic</p>
                <p>✅ DHT22 sensor reading data</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ============================================
# MAIN EXECUTION
# ============================================
def init_session_state() -> None:
    """Initializes Streamlit session state variables."""
    defaults = {
        'data_buffer': deque(maxlen=CONFIG.MAX_DATA_POINTS),
        'mqtt_client': None,
        'connected': False,
        'latest_data': None,
        'message_count': 0,
        'total_panas': 0,
        'total_normal': 0,
        'last_prediction': None,
        'last_command': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="IoT ML Dashboard",
        page_icon="🌡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inject_custom_css()
    init_session_state()
    
    # Load model globally for access in callbacks
    global ml_model
    ml_model = load_model(CONFIG.MODEL_FILE)
    
    setup_mqtt()
    render_sidebar(ml_model)
    render_dashboard()
    
    # Auto-refresh logic
    refresh_rate = 3 if st.session_state.latest_data else 5
    time.sleep(refresh_rate)
    st.rerun()


if __name__ == "__main__":
    main()