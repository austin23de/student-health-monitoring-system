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
# API SETTINGS
# =========================================================

API_URL = "https://healthmonitoring-api.onrender.com/latest"

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

    # YOUR GOOGLE SHEET NAME
    sheet = client.open(
        "Student Health Monitoring System"
    ).worksheet("Live_Records")

    google_connected = True

except Exception as e:

    st.error(f"Google Sheets Error: {e}")

# =========================================================
# GET LIVE ESP32 DATA
# =========================================================

def get_live_data():

    try:

        response = requests.get(API_URL, timeout=10)

        if response.status_code == 200:
            return response.json()

        return None

    except:
        return None


# =========================================================
# CALCULATE RISK SCORE
# =========================================================

def calculate_risk(temp, hr, spo2):

    score = 0

    if temp >= 38:
        score += 2

    if hr >= 120:
        score += 2

    if spo2 <= 94:
        score += 3

    return score


# =========================================================
# DETERMINE SEVERITY
# =========================================================

def determine_severity(score):

    if score >= 5:
        return "CRITICAL"

    elif score >= 2:
        return "WARNING"

    else:
        return "NORMAL"


# =========================================================
# ALERT STATUS
# =========================================================

def determine_alert(severity):

    if severity == "CRITICAL":
        return "ALERT"

    return "OK"


# =========================================================
# BMI CALCULATION
# =========================================================

def calculate_bmi(weight, height_cm):

    try:

        height_m = height_cm / 100

        bmi = weight / (height_m * height_m)

        return round(bmi, 2)

    except:
        return 0


# =========================================================
# SAVE TO GOOGLE SHEETS
# =========================================================

def save_to_google_sheets(data):

    if sheet:

        try:

            sheet.append_row([
                data["student_id"],
                data["name"],
                data["temperature"],
                data["heart_rate"],
                data["bp"],
                data["spo2"],
                data["bmi"],
                data["risk_score"],
                data["severity"],
                data["alert"],
                time.strftime("%Y-%m-%d %H:%M:%S")
            ])

            return True

        except Exception as e:

            st.error(f"Google Sheet Save Error: {e}")

            return False

    return False


# =========================================================
# CONNECTION STATUS
# =========================================================

live_data = get_live_data()

col1, col2, col3 = st.columns(3)

with col1:
    st.success("✅ Streamlit Dashboard Online")

with col2:

    if google_connected:
        st.success("✅ Google Sheets Connected")
    else:
        st.error("❌ Google Sheets Offline")

with col3:

    if live_data:
        st.success("✅ ESP32 Connected")
    else:
        st.warning("⚠️ Waiting for ESP32 data...")


st.divider()

# =========================================================
# LIVE ESP32 DATA
# =========================================================

st.header("📡 Live ESP32 Health Data")

if live_data:

    student_id = live_data.get("student_id", "ST001")
    name = live_data.get("name", "Unknown")

    temperature = float(live_data.get("temperature", 0))
    heart_rate = int(live_data.get("heart_rate", 0))
    spo2 = int(live_data.get("spo2", 0))

    bp = live_data.get("bp", "120/80")

    weight = float(live_data.get("weight", 70))
    height = float(live_data.get("height", 170))

    bmi = calculate_bmi(weight, height)

    risk_score = calculate_risk(
        temperature,
        heart_rate,
        spo2
    )

    severity = determine_severity(risk_score)

    alert = determine_alert(severity)

    # =====================================================
    # SAVE LIVE DATA TO GOOGLE SHEETS
    # =====================================================

    live_record = {
        "student_id": student_id,
        "name": name,
        "temperature": temperature,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "bp": bp,
        "bmi": bmi,
        "risk_score": risk_score,
        "severity": severity,
        "alert": alert
    }

    save_to_google_sheets(live_record)

    # =====================================================
    # STUDENT INFO
    # =====================================================

    st.subheader("👤 Student Information")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.info(f"Student ID: {student_id}")

    with info2:
        st.info(f"Name: {name}")

    with info3:
        st.info(f"Severity: {severity}")

    st.divider()

    # =====================================================
    # VITALS
    # =====================================================

    st.subheader("🩺 Vital Signs")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "🌡 Temperature",
            f"{temperature} °C"
        )

    with c2:
        st.metric(
            "❤️ Heart Rate",
            f"{heart_rate} BPM"
        )

    with c3:
        st.metric(
            "🫁 SpO2",
            f"{spo2}%"
        )

    with c4:
        st.metric(
            "⚖️ BMI",
            f"{bmi}"
        )

    st.divider()

    # =====================================================
    # RISK SECTION
    # =====================================================

    st.subheader("📋 Risk Analysis")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.warning(f"Risk Score: {risk_score}")

    with r2:
        st.error(f"Severity: {severity}")

    with r3:
        st.info(f"Alert Status: {alert}")

else:

    st.warning("⚠️ No live ESP32 data available.")

    st.info("You can activate Manual Override Mode below.")


st.divider()

# =========================================================
# AUTO REFRESH
# =========================================================

st.header("🔄 Live Monitoring")

auto_refresh = st.toggle("Enable Auto Refresh")

if auto_refresh:

    time.sleep(5)

    st.rerun()


st.divider()

# =========================================================
# MANUAL OVERRIDE SECTION
# =========================================================

st.header("🛠 Emergency Manual Override")

manual_mode = st.toggle("Enable Manual Vital Entry")

if manual_mode:

    st.subheader("✍️ Enter Health Data Manually")

    with st.form("manual_form"):

        col1, col2 = st.columns(2)

        with col1:

            student_id = st.text_input(
                "Student ID",
                value="ST001"
            )

            name = st.text_input(
                "Student Name",
                value="Frank Lee"
            )

            temperature = st.number_input(
                "Temperature",
                value=36.5
            )

            heart_rate = st.number_input(
                "Heart Rate",
                value=75
            )

        with col2:

            spo2 = st.number_input(
                "SpO2",
                value=98
            )

            weight = st.number_input(
                "Weight (kg)",
                value=70.0
            )

            height = st.number_input(
                "Height (cm)",
                value=170.0
            )

            bp = st.text_input(
                "Blood Pressure",
                value="120/80"
            )

        submit = st.form_submit_button(
            "Save Manual Record"
        )

        if submit:

            bmi = calculate_bmi(weight, height)

            risk_score = calculate_risk(
                temperature,
                heart_rate,
                spo2
            )

            severity = determine_severity(
                risk_score
            )

            alert = determine_alert(
                severity
            )

            manual_record = {
                "student_id": student_id,
                "name": name,
                "temperature": temperature,
                "heart_rate": heart_rate,
                "spo2": spo2,
                "bp": bp,
                "bmi": bmi,
                "risk_score": risk_score,
                "severity": severity,
                "alert": alert
            }

            success = save_to_google_sheets(
                manual_record
            )

            if success:

                st.success(
                    "✅ Manual record saved to Google Sheets"
                )

            st.json(manual_record)


st.divider()

# =========================================================
# SHOW GOOGLE SHEET DATA
# =========================================================

st.header("📄 Google Sheets Health Records")

if google_connected:

    try:

        records = sheet.get_all_records()

        if records:

            df = pd.DataFrame(records)

            st.dataframe(
                df,
                use_container_width=True
            )

        else:

            st.info("No records found.")

    except Exception as e:

        st.error(f"Error loading sheet data: {e}")

else:

    st.warning("Google Sheets not connected.")
