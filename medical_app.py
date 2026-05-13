import streamlit as st
from datetime import datetime
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import smtplib
from email.message import EmailMessage
import pandas as pd
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Health Monitoring Kiosk",
    layout="wide"
)

st.title("🏥 Smart Health Monitoring Kiosk")

# =====================================================
# API CONFIG
# =====================================================
API_BASE_URL = "https://healthmonitoring-api.onrender.com"


# =====================================================
# GET LIVE ESP32 DATA
# =====================================================
def get_latest_api_reading():
    try:
        response = requests.get(
            f"{API_BASE_URL}/latest-reading",
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()

            if data.get("status") == "success":
                return data.get("data"), None

        return None, "No ESP32 reading available."

    except Exception as e:
        return None, str(e)


# =====================================================
# GOOGLE SHEETS CONNECTION
# =====================================================
def connect_to_google_sheet():
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]

        creds_dict = dict(st.secrets["gcp_service_account"])

        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            creds_dict,
            scope
        )

        client = gspread.authorize(creds)

        spreadsheet = client.open(
            "Student Health Monitoring System"
        )

        worksheet = spreadsheet.worksheet("Live_Records")

        return worksheet, None

    except Exception as e:
        return None, str(e)


sheet, sheet_error = connect_to_google_sheet()


# =====================================================
# EMAIL ALERT FUNCTION
# =====================================================
def send_doctor_email(
        student_id,
        name,
        temperature,
        heart_rate,
        bp,
        spo2,
        bmi,
        risk_score,
        severity
):

    try:
        sender_email = st.secrets["email"]["sender_email"]
        app_password = st.secrets["email"]["app_password"]
        doctor_email = st.secrets["email"]["doctor_email"]

        subject = f"URGENT HEALTH ALERT - {name}"

        body = f"""
SMART HEALTH MONITORING ALERT

Student ID: {student_id}
Student Name: {name}

Temperature: {temperature} °C
Heart Rate: {heart_rate} bpm
Blood Pressure: {bp}
SpO₂: {spo2} %
BMI: {bmi}

Risk Score: {risk_score}
Severity: {severity}

Timestamp:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Immediate medical attention may be required.
"""

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = doctor_email
        msg.set_content(body)

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as smtp:

            smtp.login(sender_email, app_password)
            smtp.send_message(msg)

        return True

    except Exception as e:
        st.error(f"Email Error: {e}")
        return False


# =====================================================
# CONNECTION STATUS
# =====================================================
st.subheader("🟢 System Status")

c1, c2, c3 = st.columns(3)

with c1:
    st.success("✅ Streamlit Dashboard Online")

with c2:
    if sheet_error:
        st.error("❌ Google Sheets Offline")
    else:
        st.success("✅ Google Sheets Connected")

latest_data, api_error = get_latest_api_reading()

with c3:
    if api_error:
        st.warning("⚠️ Waiting for ESP32...")
    else:
        st.success("✅ ESP32 API Connected")


st.divider()

# =====================================================
# LIVE ESP32 DATA
# =====================================================
st.header("📡 Live Student Health Data")

if latest_data:

    # -----------------------------------
    # STUDENT INFORMATION
    # -----------------------------------
    st.subheader("👤 Student Information")

    s1, s2, s3 = st.columns(3)

    with s1:
        st.info(
            f"Student ID: "
            f"{latest_data.get('student_id', 'N/A')}"
        )

    with s2:
        st.info(
            f"Name: "
            f"{latest_data.get('name', 'N/A')}"
        )

    with s3:
        st.info(
            f"Timestamp: "
            f"{latest_data.get('timestamp', 'N/A')}"
        )

    st.divider()

    # -----------------------------------
    # LIVE VITALS
    # -----------------------------------
    st.subheader("🩺 Vital Signs")

    v1, v2, v3, v4, v5 = st.columns(5)

    with v1:
        st.metric(
            "🌡 Temperature",
            f"{latest_data.get('temperature', '--')} °C"
        )

    with v2:
        st.metric(
            "❤️ Heart Rate",
            f"{latest_data.get('heart_rate', '--')} bpm"
        )

    with v3:
        st.metric(
            "🫁 SpO₂",
            f"{latest_data.get('spo2', '--')} %"
        )

    with v4:
        st.metric(
            "🩸 Blood Pressure",
            latest_data.get('bp', '--')
        )

    with v5:
        st.metric(
            "⚖️ BMI",
            latest_data.get('bmi', '--')
        )

    st.divider()

    # -----------------------------------
    # HEALTH ASSESSMENT
    # -----------------------------------
    st.subheader("🧠 AI Health Assessment")

    a1, a2, a3 = st.columns(3)

    with a1:
        st.metric(
            "Risk Score",
            latest_data.get('risk_score', '--')
        )

    with a2:
        st.metric(
            "Severity",
            latest_data.get('severity', '--')
        )

    with a3:
        st.metric(
            "Alert",
            latest_data.get('alert', '--')
        )

    # -----------------------------------
    # CRITICAL ALERT DISPLAY
    # -----------------------------------
    severity = latest_data.get("severity", "")

    if severity == "CRITICAL":
        st.error(
            "🚨 CRITICAL HEALTH CONDITION DETECTED"
        )

    elif severity == "WARNING":
        st.warning(
            "⚠️ Student Requires Monitoring"
        )

    else:
        st.success(
            "✅ Student Condition Stable"
        )

    st.divider()

    # =====================================================
    # AUTO SAVE LIVE RECORD
    # =====================================================
    if sheet:

        timestamp = latest_data.get(
            "timestamp",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        row_data = [
            latest_data.get("student_id", ""),
            latest_data.get("name", ""),
            latest_data.get("temperature", ""),
            latest_data.get("heart_rate", ""),
            latest_data.get("bp", ""),
            latest_data.get("spo2", ""),
            latest_data.get("bmi", ""),
            latest_data.get("risk_score", ""),
            latest_data.get("severity", ""),
            latest_data.get("alert", ""),
            timestamp
        ]

        try:
            existing = sheet.get_all_values()

            if len(existing) == 0 or row_data not in existing:
                sheet.append_row(row_data)

                if severity == "CRITICAL":

                    send_doctor_email(
                        latest_data.get("student_id", ""),
                        latest_data.get("name", ""),
                        latest_data.get("temperature", ""),
                        latest_data.get("heart_rate", ""),
                        latest_data.get("bp", ""),
                        latest_data.get("spo2", ""),
                        latest_data.get("bmi", ""),
                        latest_data.get("risk_score", ""),
                        severity
                    )

        except Exception as e:
            st.error(f"Google Sheets Error: {e}")

else:
    st.info("Waiting for ESP32 sensor readings...")


st.divider()

# =====================================================
# LIVE AUTO REFRESH
# =====================================================
st.subheader("🔄 Live Monitoring")

auto_refresh = st.toggle(
    "Enable auto refresh every 5 seconds",
    value=True
)

if auto_refresh:
    time.sleep(5)
    st.rerun()


st.divider()

# =====================================================
# MANUAL MAINTENANCE MODE
# =====================================================
st.header("🛠 Maintenance / Manual Entry")

maintenance_mode = st.checkbox(
    "Enable Maintenance Mode"
)

if maintenance_mode:

    st.warning(
        "Maintenance mode enabled."
    )

    with st.form("manual_entry"):

        col1, col2 = st.columns(2)

        with col1:
            student_id = st.text_input("Student ID")
            name = st.text_input("Student Name")

            temperature = st.number_input(
                "Temperature"
            )

            heart_rate = st.number_input(
                "Heart Rate"
            )

        with col2:
            spo2 = st.number_input(
                "SpO₂"
            )

            bp = st.text_input(
                "Blood Pressure"
            )

            bmi = st.number_input(
                "BMI"
            )

            severity = st.selectbox(
                "Severity",
                ["NORMAL", "WARNING", "CRITICAL"]
            )

        submit = st.form_submit_button(
            "Save Manual Record"
        )

        if submit:

            if sheet:

                timestamp = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                risk_score = 0

                if severity == "WARNING":
                    risk_score = 4

                elif severity == "CRITICAL":
                    risk_score = 7

                row_data = [
                    student_id,
                    name,
                    temperature,
                    heart_rate,
                    bp,
                    spo2,
                    bmi,
                    risk_score,
                    severity,
                    "MANUAL",
                    timestamp
                ]

                try:
                    sheet.append_row(row_data)

                    st.success(
                        "Manual record saved."
                    )

                    if severity == "CRITICAL":

                        send_doctor_email(
                            student_id,
                            name,
                            temperature,
                            heart_rate,
                            bp,
                            spo2,
                            bmi,
                            risk_score,
                            severity
                        )

                        st.warning(
                            "Doctor alert sent."
                        )

                except Exception as e:
                    st.error(f"Save Error: {e}")


# =====================================================
# GOOGLE SHEETS RECORDS PREVIEW
# =====================================================
st.divider()

st.header("📄 Google Sheets Records")

if sheet:

    try:
        records = sheet.get_all_values()

        if records:

            df = pd.DataFrame(records)

            st.dataframe(
                df,
                use_container_width=True
            )

        else:
            st.info("No records found.")

    except Exception as e:
        st.error(
            f"Unable to load records: {e}"
        )
