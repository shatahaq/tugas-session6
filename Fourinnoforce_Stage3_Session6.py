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
    # Rule-based fallback
    if ml_model is None:
        return "Panas" if temp > 30 else "Normal"
    
    try:
        # Predict using ML
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
        
        # ML Prediction
        status = predict_status(temp, hum)
        
        row = {
            "ts": ts, 
            "temp": temp, 
            "hum": hum, 
            "status": status
        }

        # Update Session State
        st.session_state.last_data = row
        st.session_state.logs.append(row)
        
        # Auto-Control (Feedback)
        command = "BUZZER_ON" if status == "Panas" else "BUZZER_OFF"
        client.publish(TOPIC_OUTPUT, command)

    except Exception as e:
        print("Parse error:", e)

# -------------------------------------------------------------
# START MQTT CLIENT
# -------------------------------------------------------------
if st.session_state.mqtt is None:
    # Use VERSION2 for compatibility
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start() # Use loop_start for non-blocking background thread
    # Note: User template used polling loop(), but loop_start() is generally safer 
    # if we want to avoid blocking the UI render. 
    # However, user explicitly asked for polling template.
    # Let's stick to user's request for polling loop at the end of script.
    client.loop_stop() # Stop it if it was started, we will use manual loop
    
    st.session_state.mqtt = client

# -------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------
st.title("🔥 IoT Realtime Dashboard (Polling Mode)")

left, right = st.columns([1, 2])

# ===================== LEFT PANEL ============================
with left:
    st.subheader("System Status")
    st.metric("MQTT Connected", "Yes" if st.session_state.connected else "No")
    
    if ml_model is not None:
        st.success("✅ ML Model Active")
    else:
        st.warning("⚠️ Using Rule-Based")

    st.subheader("Latest Reading")
    if st.session_state.last_data:
        data = st.session_state.last_data
        st.metric("Temperature", f"{data['temp']} °C")
        st.metric("Humidity", f"{data['hum']} %")
        
        status_color = "red" if data['status'] == "Panas" else "green"
        st.markdown(f"Status: :{status_color}[**{data['status']}**]")
    else:
        st.info("Waiting for data...")

    st.subheader("Manual Control")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔔 ON"):
            st.session_state.mqtt.publish(TOPIC_OUTPUT, "BUZZER_ON")
    with col_b:
        if st.button("🔕 OFF"):
            st.session_state.mqtt.publish(TOPIC_OUTPUT, "BUZZER_OFF")

# ===================== RIGHT PANEL ===========================
with right:
    st.subheader("Live Monitor")

    if len(st.session_state.logs) > 0:
        df = pd.DataFrame(st.session_state.logs)
        st.line_chart(df.set_index("ts")[["temp", "hum"]])
        
        with st.expander("View Raw Data"):
            st.dataframe(df.tail(10))
            st.download_button("Download CSV", df.to_csv().encode("utf-8"), "log.csv")
    else:
        st.info("No data yet. Waiting for ESP32...")

# -------------------------------------------------------------
# MQTT LOOP POLLING
# -------------------------------------------------------------
# This is the key part for stability on Streamlit Cloud
if st.session_state.mqtt is not None:
    st.session_state.mqtt.loop(timeout=0.05)

# Auto refresh every 1 second
time.sleep(1)
st.rerun()