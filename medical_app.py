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

        return None, "No live ESP32 reading."

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

        creds_dict = dict(
            st.secrets["gcp_service_account"]
        )

        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            creds_dict,
            scope
        )

        client = gspread.authorize(creds)

        spreadsheet = client.open(
            "Student Health Monitoring System"
        )

        worksheet = spreadsheet.worksheet(
            "Live_Records"
        )

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
SMART HEALTH ALERT

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

            smtp.login(
                sender_email,
                app_password
            )

            smtp.send_message(msg)

        return True

    except Exception as e:
        st.error(f"Email Error: {e}")
        return False


# =====================================================
# RISK CALCULATION
# =====================================================
def calculate_risk(
        temperature,
        heart_rate,
        spo2,
        bmi,
        bp
):

    risk_score = 0
    warnings = []

    # Temperature
    if temperature > 39:
        risk_score += 3
        warnings.append("High Fever")

    elif temperature > 38:
        risk_score += 2
        warnings.append("Fever")

    # Heart Rate
    if heart_rate > 120:
        risk_score += 3
        warnings.append("Very High Heart Rate")

    elif heart_rate > 100:
        risk_score += 2
        warnings.append("High Heart Rate")

    # SpO2
    if spo2 < 92:
        risk_score += 3
        warnings.append("Low Oxygen Saturation")

    elif spo2 < 95:
        risk_score += 2
        warnings.append("Borderline Oxygen Saturation")

    # BMI
    if bmi > 30:
        risk_score += 1
        warnings.append("High BMI")

    elif bmi < 18.5:
        risk_score += 1
        warnings.append("Low BMI")

    # Blood Pressure
    try:

        sys_bp, dia_bp = bp.split("/")

        sys_bp = int(sys_bp)
        dia_bp = int(dia_bp)

        if sys_bp > 180 or dia_bp > 120:
            risk_score += 3
            warnings.append("Critical Blood Pressure")

        elif sys_bp >= 140 or dia_bp >= 90:
            risk_score += 1
            warnings.append("High Blood Pressure")

    except:
        pass

    # Severity
    if risk_score >= 6:
        severity = "CRITICAL"
        alert = "ALERT"

    elif risk_score >= 3:
        severity = "WARNING"
        alert = "WATCH"

    else:
        severity = "NORMAL"
        alert = "OK"

    return risk_score, severity, alert, warnings


# =====================================================
# SYSTEM STATUS
# =====================================================
st.subheader("🟢 System Status")

s1, s2, s3 = st.columns(3)

with s1:
    st.success("✅ Streamlit Online")

with s2:

    if sheet_error:
        st.error("❌ Google Sheets Offline")

    else:
        st.success("✅ Google Sheets Connected")

latest_data, api_error = get_latest_api_reading()

with s3:

    if api_error:
        st.warning("⚠️ Waiting For ESP32")

    else:
        st.success("✅ ESP32 API Connected")


st.divider()

# =====================================================
# LIVE ESP32 DATA
# =====================================================
st.header("📡 Live ESP32 Health Data")

if latest_data:

    i1, i2, i3 = st.columns(3)

    with i1:
        st.info(
            f"Student ID: "
            f"{latest_data.get('student_id', 'N/A')}"
        )

    with i2:
        st.info(
            f"Name: "
            f"{latest_data.get('name', 'N/A')}"
        )

    with i3:
        st.info(
            f"Timestamp: "
            f"{latest_data.get('timestamp', 'N/A')}"
        )

    st.divider()

    # VITALS
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

    # AI ASSESSMENT
    st.subheader("🧠 Health Assessment")

    risk_score = latest_data.get("risk_score", 0)
    severity = latest_data.get("severity", "NORMAL")
    alert = latest_data.get("alert", "OK")

    a1, a2, a3 = st.columns(3)

    with a1:
        st.metric("Risk Score", risk_score)

    with a2:
        st.metric("Severity", severity)

    with a3:
        st.metric("Alert", alert)

    # ALERT DISPLAY
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

else:
    st.info(
        "No live ESP32 data available."
    )


st.divider()

# =====================================================
# AUTO REFRESH
# =====================================================
st.subheader("🔄 Live Monitoring")

auto_refresh = st.toggle(
    "Enable Auto Refresh",
    value=True
)

if auto_refresh:
    time.sleep(5)
    st.rerun()


st.divider()

# =====================================================
# EMERGENCY MANUAL OVERRIDE
# =====================================================
st.header("🛠 Emergency Manual Override")

manual_override = st.toggle(
    "Enable Manual Vital Entry"
)

if manual_override:

    st.warning(
        "Manual override active."
    )

    with st.form("manual_vitals_form"):

        col1, col2 = st.columns(2)

        with col1:

            student_id = st.text_input(
                "Student ID"
            )

            name = st.text_input(
                "Student Name"
            )

            temperature = st.number_input(
                "Temperature (°C)",
                value=36.5
            )

            heart_rate = st.number_input(
                "Heart Rate",
                value=75
            )

        with col2:

            spo2 = st.number_input(
                "SpO₂ (%)",
                value=98
            )

            bp = st.text_input(
                "Blood Pressure",
                value="120/80"
            )

            bmi = st.number_input(
                "BMI",
                value=22.0
            )

        submit_manual = st.form_submit_button(
            "Process Manual Record"
        )

        if submit_manual:

            risk_score, severity, alert, warnings = calculate_risk(
                temperature,
                heart_rate,
                spo2,
                bmi,
                bp
            )

            # DISPLAY RESULT
            st.subheader(
                "🧠 Manual Health Assessment"
            )

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "Risk Score",
                    risk_score
                )

            with m2:
                st.metric(
                    "Severity",
                    severity
                )

            with m3:
                st.metric(
                    "Alert",
                    alert
                )

            # WARNINGS
            if warnings:

                st.warning(
                    "⚠️ Issues Detected"
                )

                for item in warnings:
                    st.write(f"- {item}")

            else:
                st.success(
                    "✅ No major abnormalities detected."
                )

            # SAVE TO SHEETS
            if sheet:

                timestamp = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

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
                    alert,
                    timestamp
                ]

                try:

                    sheet.append_row(row_data)

                    st.success(
                        "Record saved to Google Sheets."
                    )

                    # EMAIL ALERT
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

                        st.error(
                            "🚨 Doctor Alert Sent"
                        )

                except Exception as e:
                    st.error(
                        f"Save Error: {e}"
                    )


st.divider()

# =====================================================
# GOOGLE SHEETS RECORDS
# =====================================================
st.header("📄 Google Sheets Health Records")

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

            st.info(
                "No records available."
            )

    except Exception as e:

        st.error(
            f"Unable to load records: {e}"
        )
