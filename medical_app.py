import streamlit as st
import requests
import pandas as pd
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Health Kiosk",
    layout="wide"
)

st.title("🏥 Smart Health Kiosk Dashboard")

# =========================================================
# GOOGLE SHEETS CONNECTION
# =========================================================

sheet = None
google_connected = False

try:

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

    google_connected = True

except Exception as e:

    st.error(f"Google Sheets Error: {e}")

# =========================================================
# API SETTINGS
# =========================================================

API_URL = "https://healthmonitoring-api.onrender.com/latest"

# =========================================================
# GET LIVE DATA
# =========================================================

def get_live_data():

    try:

        response = requests.get(API_URL, timeout=10)

        if response.status_code == 200:

            data = response.json()

            if isinstance(data, dict):
                return data

        return None

    except:
        return None

# =========================================================
# LIVE DATA
# =========================================================

live_data = get_live_data()

# =========================================================
# CONNECTION STATUS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.success("✅ Streamlit Dashboard Running")

with col2:

    if google_connected:
        st.success("✅ Google Sheets Connected")
    else:
        st.error("❌ Google Sheets Not Connected")

with col3:

    if live_data:
        st.success("✅ ESP32 API Connected")
    else:
        st.warning("⚠️ Waiting for ESP32 data...")

st.divider()

# =========================================================
# LIVE HEALTH DATA
# =========================================================

st.header("📡 Live Student Health Data")

if live_data:

    # =====================================================
    # STUDENT INFO
    # =====================================================

    st.subheader("👤 Student Information")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.info(
            f"Student ID: {live_data.get('student_id', 'N/A')}"
        )

    with info2:
        st.info(
            f"Name: {live_data.get('name', 'N/A')}"
        )

    with info3:
        st.info(
            f"Severity: {live_data.get('severity', 'N/A')}"
        )

    st.divider()

    # =====================================================
    # VITAL SIGNS
    # =====================================================

    st.subheader("🩺 Vital Signs")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="🌡 Temperature",
            value=f"{live_data.get('temperature', '--')} °C"
        )

    with c2:
        st.metric(
            label="❤️ Heart Rate",
            value=f"{live_data.get('heart_rate', '--')} BPM"
        )

    with c3:
        st.metric(
            label="🫁 SpO₂",
            value=f"{live_data.get('spo2', '--')} %"
        )

    with c4:
        st.metric(
            label="⚖ BMI",
            value=f"{live_data.get('bmi', '--')}"
        )

    st.divider()

    # =====================================================
    # EXTRA INFO
    # =====================================================

    st.subheader("📋 Additional Information")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.info(
            f"Blood Pressure: {live_data.get('bp', 'N/A')}"
        )

    with d2:
        st.info(
            f"Risk Score: {live_data.get('risk_score', 'N/A')}"
        )

    with d3:
        st.info(
            f"Alert Status: {live_data.get('alert', 'N/A')}"
        )

else:

    st.warning("No live health data received from ESP32 yet.")
    st.info("You can switch to Manual Override Mode below.")

st.divider()

# =========================================================
# AUTO REFRESH
# =========================================================

st.subheader("🔄 Live Monitoring")

auto_refresh = st.toggle("Enable Auto Refresh")

if auto_refresh:

    time.sleep(5)
    st.rerun()

st.divider()

# =========================================================
# MANUAL OVERRIDE
# =========================================================

st.header("🛠 Emergency Manual Override")

manual_override = st.toggle(
    "Enable Manual Vital Entry"
)

# =========================================================
# MANUAL FORM
# =========================================================

if manual_override:

    st.subheader("📋 Manual Health Data Entry")

    with st.form("manual_form"):

        col1, col2 = st.columns(2)

        with col1:

            student_id = st.text_input("Student ID")

            name = st.text_input("Student Name")

            temperature = st.number_input(
                "Temperature (°C)",
                value=36.5
            )

            heart_rate = st.number_input(
                "Heart Rate (BPM)",
                value=75
            )

            spo2 = st.number_input(
                "SpO₂ (%)",
                value=98
            )

        with col2:

            weight = st.number_input(
                "Weight (kg)",
                value=70.0
            )

            height = st.number_input(
                "Height (m)",
                value=1.70
            )

            systolic = st.number_input(
                "Systolic BP",
                value=120
            )

            diastolic = st.number_input(
                "Diastolic BP",
                value=80
            )

        # =================================================
        # BMI CALCULATION
        # =================================================

        bmi = round(
            weight / (height * height),
            2
        )

        # =================================================
        # RISK SCORE CALCULATION
        # =================================================

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

        # =================================================
        # SEVERITY
        # =================================================

        if risk_score <= 2:
            severity = "NORMAL"

        elif risk_score <= 5:
            severity = "WARNING"

        else:
            severity = "CRITICAL"

        # =================================================
        # DISPLAY ANALYSIS
        # =================================================

        st.divider()

        st.subheader("📊 Health Analysis")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("⚖ BMI", bmi)

        with r2:
            st.metric("📈 Risk Score", risk_score)

        with r3:
            st.metric("🚨 Severity", severity)

        # =================================================
        # SUBMIT BUTTON
        # =================================================

        submit = st.form_submit_button(
            "💾 Save Health Record"
        )

        # =================================================
        # SAVE TO GOOGLE SHEETS
        # =================================================

        if submit:

            bp = f"{systolic}/{diastolic}"

            manual_data = {
                "student_id": student_id,
                "name": name,
                "temperature": temperature,
                "heart_rate": heart_rate,
                "spo2": spo2,
                "bmi": bmi,
                "bp": bp,
                "risk_score": risk_score,
                "severity": severity
            }

            if google_connected:

                try:

                    sheet.append_row([
                        student_id,
                        name,
                        temperature,
                        heart_rate,
                        spo2,
                        bmi,
                        bp,
                        risk_score,
                        severity
                    ])

                    st.success(
                        "Health data saved successfully."
                    )

                except Exception as e:

                    st.error(
                        f"Google Sheets Save Error: {e}"
                    )

            else:

                st.error(
                    "Google Sheets not connected."
                )

            st.json(manual_data)

st.divider()

# =========================================================
# GOOGLE SHEETS TABLE
# =========================================================

st.header("📄 Google Sheets Health Records")

if google_connected:

    try:

        records = sheet.get_all_records()

        if len(records) > 0:

            df = pd.DataFrame(records)

            st.dataframe(
                df,
                use_container_width=True
            )

        else:

            st.info(
                "No records currently stored."
            )

    except Exception as e:

        st.error(
            f"Unable to load Google Sheet records: {e}"
        )

else:

    st.warning(
        "Google Sheets connection unavailable."
    )
