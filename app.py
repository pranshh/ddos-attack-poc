import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Pretrained Model, Scaler, and Encoders
best_model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Feature Names
features = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count",
    "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP",
    "ARP", "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT",
    "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight"
]

# Streamlit App
st.set_page_config(page_title="DDOS Attack Predictor", layout="wide")

st.title("DDOS Attack Prediction")
st.write("Enter values for the features below, and the app will predict the attack type.")

# Input Fields
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"{feature}:", value=0.0)

if st.button("Predict"):
    # Convert Input Data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess Input Data
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])
    scaled_input = scaler.transform(input_df)

    # Predict Label
    prediction = best_model.predict(scaled_input)[0]

    # Decode Prediction if Encoded
    if "label" in label_encoders:
        prediction_label = label_encoders["label"].inverse_transform([prediction])[0]
    else:
        prediction_label = prediction

    st.write(f"### Predicted Label: {prediction_label}")
