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

st.caption(
    "This system is for educational and health monitoring support only. "
    "It does not replace professional medical diagnosis or treatment."
)

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

    credentials = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp_service_account"],
        scope
    )

    client = gspread.authorize(credentials)

    sheet = client.open(
        "Student Health Monitoring System"
    ).worksheet("Live_Records")

    google_connected = True

except Exception as e:
    st.error("❌ GOOGLE SHEETS CONNECTION ERROR")
    st.exception(e)

# =========================================================
# GET LIVE ESP32 DATA
# =========================================================

def get_live_data():
    try:
        response = requests.get(API_URL, timeout=10)

        if response.status_code == 200:
            return response.json()

        return None

    except Exception:
        return None


# =========================================================
# HEALTH CALCULATIONS
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


def determine_severity(score):
    if score >= 5:
        return "CRITICAL"
    elif score >= 2:
        return "WARNING"
    else:
        return "NORMAL"


def determine_alert(severity):
    if severity == "CRITICAL":
        return "ALERT"
    return "OK"


def calculate_bmi(weight, height_cm):
    try:
        height_m = height_cm / 100
        bmi = weight / (height_m * height_m)
        return round(bmi, 2)
    except Exception:
        return 0


# =========================================================
# AI HEALTH ASSISTANT FUNCTIONS
# =========================================================

def explain_risk(score, severity, temp, hr, spo2, bmi):
    explanation = f"""
Based on the current readings:

- Risk Score: {score}
- Severity Level: {severity}
- Temperature: {temp} °C
- Heart Rate: {hr} BPM
- SpO2: {spo2}%
- BMI: {bmi}
"""

    if severity == "CRITICAL":
        explanation += """
Interpretation:
The student may require urgent attention because one or more vital signs are outside the safe range.

Recommended Action:
Inform a school health officer or healthcare professional immediately.
"""
    elif severity == "WARNING":
        explanation += """
Interpretation:
Some readings are outside the normal range. The student should be monitored closely.

Recommended Action:
Allow the student to rest, repeat the vital checks, and seek medical review if symptoms continue.
"""
    else:
        explanation += """
Interpretation:
The readings appear generally normal based on the thresholds used in this system.

Recommended Action:
Continue routine monitoring and encourage healthy habits.
"""

    return explanation


def symptom_faq(question, score, severity, temp, hr, spo2, bmi):
    question = question.lower()

    if "fever" in question or "temperature" in question:
        return (
            "A high temperature may suggest fever or infection. "
            "The student should rest, drink fluids, and be monitored. "
            "Medical attention is advised if fever is high or persistent."
        )

    elif "heart" in question or "pulse" in question or "bpm" in question:
        return (
            "A high heart rate may occur due to stress, fever, dehydration, exercise, or illness. "
            "If it remains very high, the student should be reviewed by a healthcare professional."
        )

    elif "spo2" in question or "oxygen" in question or "breathing" in question:
        return (
            "Low SpO2 may suggest reduced oxygen level. "
            "If oxygen level is low or the student has breathing difficulty, urgent medical help is needed."
        )

    elif "bmi" in question or "weight" in question:
        return (
            "BMI is a simple estimate of body weight status using height and weight. "
            "It should not be used alone to judge a student’s health."
        )

    elif "risk" in question or "score" in question or "severity" in question:
        return explain_risk(score, severity, temp, hr, spo2, bmi)

    elif "recommend" in question or "advice" in question or "what should" in question:
        if severity == "CRITICAL":
            return "Recommendation: Alert a health officer immediately and arrange urgent clinical assessment."
        elif severity == "WARNING":
            return "Recommendation: Monitor the student, allow rest, repeat vital checks, and seek medical review if symptoms continue."
        else:
            return "Recommendation: Continue routine monitoring and encourage healthy habits."

    elif "project" in question or "system" in question:
        return (
            "This Smart Health Kiosk monitors student vital signs using IoT sensor data, "
            "calculates a health risk score, stores records in Google Sheets, and uses "
            "an AI-style assistant to explain health readings in simple language."
        )

    else:
        return (
            "I can answer questions about fever, heart rate, SpO2, BMI, risk score, "
            "severity level, recommendations, and how this health kiosk system works."
        )


# =========================================================
# SAVE TO GOOGLE SHEETS
# =========================================================

def save_to_google_sheets(data):
    if sheet is not None:
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
            st.error("❌ GOOGLE SHEET SAVE ERROR")
            st.exception(e)
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
# DEFAULT VARIABLES
# =========================================================

student_id = "ST001"
name = "Unknown"
temperature = 0.0
heart_rate = 0
spo2 = 0
bp = "120/80"
weight = 70.0
height = 170.0
bmi = 0
risk_score = 0
severity = "NORMAL"
alert = "OK"
current_record_available = False


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
    risk_score = calculate_risk(temperature, heart_rate, spo2)
    severity = determine_severity(risk_score)
    alert = determine_alert(severity)

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
    current_record_available = True

    st.subheader("👤 Student Information")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.info(f"Student ID: {student_id}")

    with info2:
        st.info(f"Name: {name}")

    with info3:
        st.info(f"Severity: {severity}")

    st.divider()

    st.subheader("🩺 Vital Signs")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🌡 Temperature", f"{temperature} °C")

    with c2:
        st.metric("❤️ Heart Rate", f"{heart_rate} BPM")

    with c3:
        st.metric("🫁 SpO2", f"{spo2}%")

    with c4:
        st.metric("⚖️ BMI", f"{bmi}")

    st.divider()

    st.subheader("📋 Risk Analysis")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.warning(f"Risk Score: {risk_score}")

    with r2:
        if severity == "CRITICAL":
            st.error(f"Severity: {severity}")
        elif severity == "WARNING":
            st.warning(f"Severity: {severity}")
        else:
            st.success(f"Severity: {severity}")

    with r3:
        st.info(f"Alert Status: {alert}")

else:
    st.warning("⚠️ No live ESP32 data available.")
    st.info("You can activate Manual Override Mode below.")


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
            student_id = st.text_input("Student ID", value="ST001")
            name = st.text_input("Student Name", value="Frank Lee")
            temperature = st.number_input("Temperature", value=36.5)
            heart_rate = st.number_input("Heart Rate", value=75)

        with col2:
            spo2 = st.number_input("SpO2", value=98)
            weight = st.number_input("Weight (kg)", value=70.0)
            height = st.number_input("Height (cm)", value=170.0)
            bp = st.text_input("Blood Pressure", value="120/80")

        submit = st.form_submit_button("Save Manual Record")

        if submit:
            bmi = calculate_bmi(weight, height)
            risk_score = calculate_risk(temperature, heart_rate, spo2)
            severity = determine_severity(risk_score)
            alert = determine_alert(severity)

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

            success = save_to_google_sheets(manual_record)

            if success:
                st.success("✅ Manual record saved to Google Sheets")

            st.json(manual_record)
            current_record_available = True


st.divider()


# =========================================================
# AI HEALTH ASSISTANT
# =========================================================

st.header("🤖 MedExplain AI Health Assistant")

st.caption(
    "This assistant explains health readings in simple language. "
    "It is for educational support only and is not a medical diagnosis tool."
)

if current_record_available or manual_mode:
    ai_col1, ai_col2 = st.columns([2, 1])

    with ai_col1:
        user_question = st.text_input(
            "Ask MedExplain AI a question",
            placeholder="Example: What does this risk score mean?"
        )

        if user_question:
            response = symptom_faq(
                user_question,
                risk_score,
                severity,
                temperature,
                heart_rate,
                spo2,
                bmi
            )

            st.info(response)

    with ai_col2:
        st.subheader("Suggested Questions")
        st.write("- What does this risk score mean?")
        st.write("- What does low SpO2 mean?")
        st.write("- What should I do next?")
        st.write("- Explain the BMI result")
        st.write("- What does this system do?")

    with st.expander("📌 Automatic AI Interpretation"):
        st.write(
            explain_risk(
                risk_score,
                severity,
                temperature,
                heart_rate,
                spo2,
                bmi
            )
        )

else:
    st.warning("AI Assistant will activate when live or manual health data is available.")


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
# SHOW GOOGLE SHEET DATA
# =========================================================

st.header("📄 Google Sheets Health Records")

if google_connected:
    try:
        records = sheet.get_all_records()

        if records:
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No records found.")

    except Exception as e:
        st.error("❌ ERROR LOADING GOOGLE SHEET DATA")
        st.exception(e)

else:
    st.warning("Google Sheets not connected.")
