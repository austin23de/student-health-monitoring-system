import streamlit as st
from datetime import datetime
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import smtplib
from email.message import EmailMessage
import time

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Smart Health Monitoring Kiosk", layout="wide")
st.title("Smart Health Monitoring Kiosk")

# -----------------------------
# API CONNECTION
# -----------------------------
API_BASE_URL = "https://healthmonitoring-api.onrender.com"


def get_latest_api_reading():
    try:
        response = requests.get(f"{API_BASE_URL}/latest-reading", timeout=10)

        if response.status_code == 200:
            data = response.json()

            if data.get("status") == "success":
                return data.get("data"), None
            else:
                return None, "No live ESP32 reading received yet."

        return None, f"API returned status code {response.status_code}"

    except Exception as e:
        return None, str(e)


# -----------------------------
# GOOGLE SHEETS CONNECTION
# -----------------------------
def connect_to_google_sheet():
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]

        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)

        spreadsheet = client.open("Student Health Monitoring System")
        worksheet = spreadsheet.worksheet("Live_Records")

        return worksheet, None

    except Exception as e:
        return None, str(e)


sheet, connection_error = connect_to_google_sheet()

# -----------------------------
# EMAIL FUNCTION
# -----------------------------
def send_doctor_email(student_id, name, temperature, heart_rate, bp, spo2, bmi, risk_score, severity):
    try:
        sender_email = st.secrets["email"]["sender_email"]
        app_password = st.secrets["email"]["app_password"]
        doctor_email = st.secrets["email"]["doctor_email"]

        subject = f"URGENT: Health Alert for {name} ({student_id})"

        body = f"""
Student Health Alert

Student ID: {student_id}
Student Name: {name}
Temperature: {temperature} °C
Heart Rate: {heart_rate} bpm
Blood Pressure: {bp}
SpO2: {spo2} %
BMI: {bmi:.2f}
Risk Score: {risk_score}
Severity: {severity}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Immediate medical attention may be required.
"""

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = doctor_email
        msg.set_content(body)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)

        return True, "Doctor email sent successfully."

    except Exception as e:
        return False, f"Email error: {e}"


# -----------------------------
# CONNECTION STATUS
# -----------------------------
st.subheader("Connection Status")

col_a, col_b = st.columns(2)

with col_a:
    if connection_error:
        st.error(f"Google Sheets connection failed: {connection_error}")
    else:
        st.success("Connected to Google Sheets successfully.")

with col_b:
    latest_data, api_error = get_latest_api_reading()
    if api_error:
        st.warning(f"API status: {api_error}")
    else:
        st.success("Connected to live ESP32 API.")


# -----------------------------
# LIVE ESP32 DASHBOARD
# -----------------------------
st.subheader("Live ESP32 Reading")

auto_refresh = st.toggle("Live auto-refresh every 5 seconds", value=False)

if latest_data:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Student", latest_data.get("name", "N/A"))
    c2.metric("Temperature", f"{latest_data.get('temperature', 'N/A')} °C")
    c3.metric("Heart Rate", f"{latest_data.get('heart_rate', 'N/A')} bpm")
    c4.metric("SpO₂", f"{latest_data.get('spo2', 'N/A')} %")

    c5, c6, c7 = st.columns(3)

    c5.metric("Risk Score", latest_data.get("risk_score", "N/A"))
    c6.metric("Severity", latest_data.get("severity", "N/A"))
    c7.metric("Alert", latest_data.get("alert", "N/A"))

    st.write("Latest API data:")
    st.json(latest_data)

else:
    st.info("No live ESP32 data yet. Send data from Wokwi/ESP32 to your API.")


if auto_refresh:
    time.sleep(5)
    st.rerun()


# -----------------------------
# MANUAL INPUT SECTION
# -----------------------------
st.subheader("Manual Student Health Data Entry")

student_id = st.text_input("Student ID / RFID")
name = st.text_input("Student Name")

temperature = st.slider("Temperature (°C)", 30.0, 42.0, 36.5)
heart_rate = st.slider("Heart Rate (bpm)", 40, 180, 75)
bp = st.text_input("Blood Pressure (e.g. 120/80)")
spo2 = st.slider("Blood Oxygen / SpO₂ (%)", 70, 100, 98)

height = st.slider("Height (cm)", 100, 220, 170)
weight = st.slider("Weight (kg)", 30, 150, 70)

# -----------------------------
# RISK CALCULATION
# -----------------------------
bmi = weight / ((height / 100) ** 2)

risk_score = 0
warnings = []

if temperature > 39.0:
    risk_score += 3
    warnings.append("High fever")
elif temperature > 38.0:
    risk_score += 2
    warnings.append("Fever")

if heart_rate > 120:
    risk_score += 3
    warnings.append("Very high heart rate")
elif heart_rate > 100:
    risk_score += 2
    warnings.append("High heart rate")

if spo2 < 92:
    risk_score += 3
    warnings.append("Low oxygen saturation")
elif spo2 < 95:
    risk_score += 2
    warnings.append("Borderline oxygen saturation")

if bp.strip():
    try:
        sys_bp, dia_bp = bp.split("/")
        sys_bp = int(sys_bp.strip())
        dia_bp = int(dia_bp.strip())

        if sys_bp > 180 or dia_bp > 120:
            risk_score += 3
            warnings.append("Very high blood pressure")
        elif sys_bp >= 140 or dia_bp >= 90:
            risk_score += 1
            warnings.append("High blood pressure")
        elif sys_bp < 90:
            risk_score += 3
            warnings.append("Low blood pressure")
    except Exception:
        warnings.append("Invalid BP format. Use format like 120/80")

if bmi > 30:
    risk_score += 1
    warnings.append("High BMI")
elif bmi < 18.5:
    risk_score += 1
    warnings.append("Low BMI")

if risk_score >= 6:
    severity = "CRITICAL"
    alert_status = "ALERT"
elif risk_score >= 3:
    severity = "WARNING"
    alert_status = "WATCH"
else:
    severity = "NORMAL"
    alert_status = "OK"


# -----------------------------
# ASSESSMENT OUTPUT
# -----------------------------
st.subheader("Health Assessment")

m1, m2, m3, m4 = st.columns(4)

m1.metric("BMI", f"{bmi:.2f}")
m2.metric("Risk Score", risk_score)
m3.metric("Severity", severity)
m4.metric("Alert", alert_status)

if warnings:
    st.warning("Issues detected:")
    for item in warnings:
        st.write(f"- {item}")
else:
    st.success("No major warning signs detected.")

if severity == "CRITICAL":
    st.error("Immediate medical attention may be required.")
elif severity == "WARNING":
    st.warning("Monitor patient closely.")
else:
    st.success("Patient appears stable.")


# -----------------------------
# SAVE MANUAL RECORD
# -----------------------------
if st.button("Process Manual Record"):
    if not sheet:
        st.error("Cannot save because Google Sheets is not connected.")
    elif not student_id.strip():
        st.error("Student ID is required.")
    elif not name.strip():
        st.error("Student Name is required.")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row_data = [
            student_id,
            name,
            temperature,
            heart_rate,
            bp,
            spo2,
            round(bmi, 2),
            risk_score,
            severity,
            alert_status,
            timestamp
        ]

        try:
            sheet.append_row(row_data)

            if severity == "CRITICAL":
                email_sent, email_message = send_doctor_email(
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

                if email_sent:
                    st.success("Record saved to Google Sheets and doctor notified.")
                else:
                    st.warning("Record saved, but doctor email was not sent.")
                    st.info(email_message)
            else:
                st.success("Record saved to Google Sheets successfully.")

        except Exception as e:
            st.error(f"Google Sheets save failed: {e}")


# -----------------------------
# SAVE LIVE API RECORD
# -----------------------------
st.subheader("Save Live ESP32 Reading")

if st.button("Save Latest ESP32 Reading to Google Sheets"):
    if not sheet:
        st.error("Cannot save because Google Sheets is not connected.")
    elif not latest_data:
        st.error("No live ESP32 data available to save.")
    else:
        timestamp = latest_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

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
            sheet.append_row(row_data)

            if latest_data.get("severity") == "CRITICAL":
                email_sent, email_message = send_doctor_email(
                    latest_data.get("student_id", ""),
                    latest_data.get("name", ""),
                    latest_data.get("temperature", 0),
                    latest_data.get("heart_rate", 0),
                    latest_data.get("bp", ""),
                    latest_data.get("spo2", 0),
                    float(latest_data.get("bmi", 0)),
                    int(latest_data.get("risk_score", 0)),
                    latest_data.get("severity", "")
                )

                if email_sent:
                    st.success("ESP32 record saved and doctor notified.")
                else:
                    st.warning("ESP32 record saved, but email was not sent.")
                    st.info(email_message)
            else:
                st.success("ESP32 record saved to Google Sheets.")

        except Exception as e:
            st.error(f"Google Sheets save failed: {e}")


# -----------------------------
# LIVE RECORDS PREVIEW
# -----------------------------
st.subheader("Google Sheets Records Preview")

if sheet:
    try:
        records = sheet.get_all_values()
        if records:
            st.dataframe(records, use_container_width=True)
        else:
            st.info("Sheet is empty.")
    except Exception as e:
        st.error(f"Could not read records from Google Sheets: {e}")
