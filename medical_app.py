import streamlit as st
import pandas as pd
import requests
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Health Monitoring System",
    layout="wide"
)

st.title("🏥 Smart Health Monitoring System")

# =========================================================
# GOOGLE SHEETS CONNECTION
# =========================================================

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

credentials = ServiceAccountCredentials.from_json_keyfile_name(
    "credentials.json",
    scope
)

client = gspread.authorize(credentials)

sheet = client.open("HealthMonitoringData").sheet1

# =========================================================
# API URL
# =========================================================

API_URL = "https://healthmonitoring-api.onrender.com/latest"

# =========================================================
# SYSTEM STATUS
# =========================================================

st.markdown("---")
st.header("📡 System Status")

col1, col2 = st.columns(2)

with col1:
    st.success("Google Sheets Connected")

with col2:
    st.success("API Connected")

# =========================================================
# LIVE ESP32 DATA
# =========================================================

st.markdown("---")
st.header("📶 Live ESP32 Health Data")

auto_refresh = st.toggle("Enable Auto Refresh")

if auto_refresh:
    time.sleep(5)
    st.rerun()

try:
    response = requests.get(API_URL)

    if response.status_code == 200:

        data = response.json()

        if data:

            st.success("Live ESP32 data received")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "🌡 Temperature",
                    f"{data.get('temperature', 0)} °C"
                )

                st.metric(
                    "❤️ Heart Rate",
                    f"{data.get('heart_rate', 0)} BPM"
                )

            with col2:
                st.metric(
                    "🫁 SpO₂",
                    f"{data.get('spo2', 0)} %"
                )

                st.metric(
                    "⚖ BMI",
                    f"{data.get('bmi', 0)}"
                )

            with col3:
                st.metric(
                    "📈 Risk Score",
                    f"{data.get('risk_score', 0)}"
                )

                st.metric(
                    "🚨 Severity",
                    f"{data.get('severity', 'NORMAL')}"
                )

        else:
            st.warning("No live ESP32 data available.")
            st.info("You can activate Manual Override Mode below.")

    else:
        st.warning("ESP32 API is not responding.")

except:
    st.warning("No live ESP32 data available.")
    st.info("You can activate Manual Override Mode below.")

# =========================================================
# MANUAL OVERRIDE SECTION
# =========================================================

st.markdown("---")
st.header("🛠 Emergency Manual Override")

manual_override = st.toggle("Enable Manual Vital Entry")

if manual_override:

    st.subheader("Enter Student Health Information")

    student_id = st.text_input("Student ID")

    student_name = st.text_input("Student Name")

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=30.0,
        max_value=45.0,
        value=36.5
    )

    heart_rate = st.number_input(
        "Heart Rate (BPM)",
        min_value=30,
        max_value=220,
        value=75
    )

    spo2 = st.number_input(
        "SpO₂ (%)",
        min_value=50,
        max_value=100,
        value=98
    )

    systolic = st.number_input(
        "Systolic BP",
        min_value=50,
        max_value=250,
        value=120
    )

    diastolic = st.number_input(
        "Diastolic BP",
        min_value=30,
        max_value=150,
        value=80
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=10.0,
        max_value=300.0,
        value=70.0
    )

    height = st.number_input(
        "Height (m)",
        min_value=0.5,
        max_value=2.5,
        value=1.70
    )

    # =====================================================
    # BMI CALCULATION
    # =====================================================

    bmi = round(weight / (height * height), 2)

    st.metric("⚖ BMI", bmi)

    # =====================================================
    # RISK SCORE
    # =====================================================

    risk_score = 0

    if temperature > 38:
        risk_score += 2

    if heart_rate > 120:
        risk_score += 2

    if spo2 < 94:
        risk_score += 3

    if systolic > 140 or diastolic > 90:
        risk_score += 2

    if bmi > 30:
        risk_score += 1

    # =====================================================
    # SEVERITY LEVEL
    # =====================================================

    if risk_score <= 2:
        severity = "LOW"

    elif risk_score <= 5:
        severity = "MODERATE"

    else:
        severity = "HIGH"

    # =====================================================
    # SHOW RESULTS
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:
        st.metric("📈 Risk Score", risk_score)

    with col2:
        st.metric("🚨 Severity", severity)

    # =====================================================
    # SAVE TO GOOGLE SHEETS
    # =====================================================

    if st.button("💾 Save Health Record"):

        row = [
            student_id,
            student_name,
            temperature,
            heart_rate,
            spo2,
            systolic,
            diastolic,
            weight,
            height,
            bmi,
            risk_score,
            severity
        ]

        try:

            sheet.append_row(row)

            st.success(
                "Health record saved successfully to Google Sheets."
            )

        except Exception as e:

            st.error(f"Google Sheets Error: {e}")

# =========================================================
# SHOW GOOGLE SHEET DATA
# =========================================================

st.markdown("---")
st.header("📄 Google Sheets Health Records")

try:

    records = sheet.get_all_records()

    if records:

        df = pd.DataFrame(records)

        st.dataframe(
            df,
            use_container_width=True
        )

    else:
        st.info("No health records saved yet.")

except Exception as e:

    st.error(f"Unable to load Google Sheets data: {e}")
