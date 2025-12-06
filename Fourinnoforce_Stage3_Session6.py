import streamlit as st
import pandas as pd
import json
import time
import pickle
import numpy as np
import os
import paho.mqtt.client as mqtt
from datetime import datetime

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SENSOR = "fourinnoforce/class/session5/sensor"
TOPIC_OUTPUT = "fourinnoforce/class/session5/output"
MODEL_FILE = "iot_temp_model.pkl"

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="IoT Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------
# MODERN CSS STYLING
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .stApp {
        background: transparent;
    }
    
    /* Header Styling */
    h1 {
        color: white !important;
        font-weight: 700 !important;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 2rem !important;
    }
    
    h2, h3 {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Card Styling */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #666 !important;
        font-weight: 600 !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Status Box */
    .status-box {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .status-hot {
        border-left: 5px solid #FF4B4B;
    }
    
    .status-normal {
        border-left: 5px solid #00C851;
    }
    
    /* Data Info Box */
    .info-box {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Download Button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(17, 153, 142, 0.4);
    }
    
    /* Chart Container */
    .stPlotlyChart {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white !important;
        font-weight: 600;
    }
    
    /* DataFrame */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# ML MODEL LOADING
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        print("✅ ML Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

ml_model = load_model()

def predict_status(temp, hum):
    if ml_model is None:
        return "Panas" if temp > 30 else "Normal"
    
    try:
        features = np.array([[temp, hum]])
        prediction = ml_model.predict(features)[0]
        
        if prediction == 1 or prediction == "Panas":
            return "Panas"
        else:
            return "Normal"
    except:
        return "Panas" if temp > 30 else "Normal"

# -------------------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------------------
if "connected" not in st.session_state:
    st.session_state.connected = False

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last_data" not in st.session_state:
    st.session_state.last_data = None

if "mqtt" not in st.session_state:
    st.session_state.mqtt = None

# -------------------------------------------------------------
# MQTT CALLBACKS
# -------------------------------------------------------------
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        st.session_state.connected = True
        client.subscribe(TOPIC_SENSOR)
        print(f"✅ Connected to {MQTT_BROKER}")
    else:
        st.session_state.connected = False
        print("❌ MQTT connection failed")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        ts = datetime.now().strftime("%H:%M:%S")
        temp = float(data.get("temp", 0))
        hum = float(data.get("hum", 0))
        
        status = predict_status(temp, hum)
        
        row = {
            "ts": ts, 
            "temp": temp, 
            "hum": hum, 
            "status": status
        }

        st.session_state.last_data = row
        st.session_state.logs.append(row)
        
        command = "BUZZER_ON" if status == "Panas" else "BUZZER_OFF"
        client.publish(TOPIC_OUTPUT, command)

    except Exception as e:
        print("Parse error:", e)

# -------------------------------------------------------------
# START MQTT CLIENT
# -------------------------------------------------------------
if st.session_state.mqtt is None:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    st.session_state.mqtt = client

# -------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------
st.title("🌡️ IoT Machine Learning Dashboard")

# Connection Status Banner
if st.session_state.connected:
    st.success("✅ Connected to MQTT Broker: " + MQTT_BROKER)
else:
    st.error("❌ Disconnected from MQTT Broker")

st.markdown("<br>", unsafe_allow_html=True)

# Main Layout
col1, col2 = st.columns([1, 2])

# ===================== LEFT PANEL ============================
with col1:
    # Model Status
    st.markdown("### 🎯 System Status")
    if ml_model is not None:
        st.success("✅ ML Model Active")
    else:
        st.warning("⚠️ Using Rule-Based System")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Latest Reading
    st.markdown("### 📊 Latest Reading")
    
    if st.session_state.last_data:
        data = st.session_state.last_data
        
        # Temperature & Humidity Metrics
        col_temp, col_hum = st.columns(2)
        with col_temp:
            st.metric("🌡️ Temp", f"{data['temp']}°C")
        with col_hum:
            st.metric("💧 Hum", f"{data['hum']}%")
        
        # Status Box
        status_class = "status-hot" if data['status'] == "Panas" else "status-normal"
        status_emoji = "🔥" if data['status'] == "Panas" else "✅"
        status_color = "#FF4B4B" if data['status'] == "Panas" else "#00C851"
        
        st.markdown(f"""
        <div class="status-box {status_class}">
            <h2 style="color: {status_color}; margin: 0;">{status_emoji} {data['status']}</h2>
            <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Classification Status</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("⏳ Waiting for sensor data...")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Manual Control
    st.markdown("### 🎛️ Manual Control")
    col_on, col_off = st.columns(2)
    with col_on:
        if st.button("🔔 BUZZER ON", use_container_width=True):
            st.session_state.mqtt.publish(TOPIC_OUTPUT, "BUZZER_ON")
            st.success("Sent!")
    with col_off:
        if st.button("🔕 BUZZER OFF", use_container_width=True):
            st.session_state.mqtt.publish(TOPIC_OUTPUT, "BUZZER_OFF")
            st.success("Sent!")

# ===================== RIGHT PANEL ===========================
with col2:
    st.markdown("### 📈 Real-Time Monitoring")
    
    if len(st.session_state.logs) > 0:
        df = pd.DataFrame(st.session_state.logs)
        
        # Line Chart
        st.line_chart(
            df.set_index("ts")[["temp", "hum"]], 
            use_container_width=True,
            height=400
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data Table in Expander
        with st.expander("📋 View Raw Data", expanded=False):
            # Show recent 15 records
            recent_df = df.tail(15).iloc[::-1]  # Reverse to show newest first
            st.dataframe(
                recent_df, 
                use_container_width=True,
                hide_index=True
            )
            
            # Download Button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Log (CSV)",
                data=csv,
                file_name=f"iot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("📡 No data received yet. Waiting for ESP32...")

# -------------------------------------------------------------
# MQTT LOOP POLLING
# -------------------------------------------------------------
if st.session_state.mqtt is not None:
    st.session_state.mqtt.loop(timeout=0.05)

time.sleep(1)
st.rerun()