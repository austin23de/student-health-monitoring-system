import streamlit as st
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import smtplib
from email.message import EmailMessage

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Smart Health Monitoring Kiosk", layout="centered")
st.title("Smart Health Monitoring Kiosk")

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

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
                smtp.login(sender_email, app_password)
                smtp.send_message(msg)
            return True, "Doctor email sent successfully."

        except Exception as e1:
            try:
                with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as smtp:
                    smtp.ehlo()
                    smtp.starttls()
                    smtp.ehlo()
                    smtp.login(sender_email, app_password)
                    smtp.send_message(msg)
                return True, "Doctor email sent successfully using TLS fallback."
            except Exception as e2:
                return False, f"SSL failed: {e1} | TLS failed: {e2}"

    except Exception as e:
        return False, f"Email configuration error: {e}"

# -----------------------------
# CONNECTION STATUS
# -----------------------------
st.subheader("Connection Status")
if connection_error:
    st.error(f"Google Sheets connection failed: {connection_error}")
else:
    st.success("Connected to Google Sheets successfully.")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("Enter Student Health Data")

student_id = st.text_input("Student ID / RFID")
name = st.text_input("Student Name")

temperature = st.slider("Temperature (°C)", 30.0, 42.0, 36.5)
heart_rate = st.slider("Heart Rate (bpm)", 40, 180, 75)
bp = st.text_input("Blood Pressure (e.g. 120/80)")
spo2 = st.slider("Blood Oxygen / SpO₂ (%)", 70, 100, 98)

height = st.slider("Height (cm)", 100, 220, 170)
weight = st.slider("Weight (kg)", 30, 150, 70)

# -----------------------------
# CALCULATIONS
# -----------------------------
bmi = weight / ((height / 100) ** 2)

risk_score = 0
warnings = []

# Temperature
if temperature > 39:
    risk_score += 3
    warnings.append("High fever")
elif temperature > 38:
    risk_score += 2
    warnings.append("Fever")
elif temperature < 35:
    risk_score += 3
    warnings.append("Possible hypothermia")

# Heart Rate
if heart_rate > 130 or heart_rate < 50:
    risk_score += 3
    warnings.append("Critical heart rate")
elif heart_rate > 100:
    risk_score += 2
    warnings.append("High heart rate")

# Blood Pressure
if bp.strip():
    try:
        sys_bp, dia_bp = bp.split("/")
        sys_bp = int(sys_bp.strip())
        dia_bp = int(dia_bp.strip())

        if sys_bp > 180 or dia_bp > 120:
            risk_score += 3
            warnings.append("Hypertensive crisis")
        elif sys_bp > 140 or dia_bp > 90:
            risk_score += 2
            warnings.append("High blood pressure")
    except Exception:
        warnings.append("Invalid BP format. Use format like 120/80")

# SpO2
if spo2 < 90:
    risk_score += 3
    warnings.append("Dangerously low oxygen level")
elif spo2 < 95:
    risk_score += 2
    warnings.append("Low oxygen level")

# BMI
if bmi > 30:
    risk_score += 1
    warnings.append("Obesity risk")
elif bmi < 18.5:
    risk_score += 1
    warnings.append("Underweight risk")

# -----------------------------
# DECISION
# -----------------------------
if risk_score >= 6:
    severity = "CRITICAL"
elif risk_score >= 3:
    severity = "WARNING"
else:
    severity = "NORMAL"

alert_status = "ALERT" if severity == "CRITICAL" else "OK"

# -----------------------------
# OUTPUT SECTION
# -----------------------------
st.subheader("Health Assessment")

st.write(f"**BMI:** {bmi:.2f}")
st.write(f"**Risk Score:** {risk_score}")
st.write(f"**Severity Level:** {severity}")
st.write(f"**Alert Status:** {alert_status}")

if warnings:
    st.write("Issues Detected:")
    for item in warnings:
        st.write(f"- {item}")

if severity == "CRITICAL":
    st.error("Immediate medical attention required!")
elif severity == "WARNING":
    st.warning("Monitor patient closely.")
else:
    st.success("Patient is stable.")

# -----------------------------
# SAVE TO GOOGLE SHEETS
# -----------------------------
if st.button("Process Record"):
    if not sheet:
        st.error("Cannot save because Google Sheets is not connected.")
    elif not student_id.strip():
        st.error("Student ID is required.")
    elif not name.strip():
        st.error("Student Name is required.")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        email_status = "Not needed"

        if severity == "CRITICAL":
            email_sent, email_message = send_doctor_email(
                student_id, name, temperature, heart_rate, bp, spo2, bmi, risk_score, severity
            )
            if email_sent:
                email_status = "Sent"
            else:
                email_status = f"Failed: {email_message}"

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
                if email_status == "Sent":
                    st.success("Record saved to Google Sheets and doctor notified.")
                else:
                    st.warning("Record saved, but email was not sent.")
                    st.info(email_status)
            else:
                st.success("Record saved to Google Sheets successfully.")
        except Exception as e:
            st.error(f"Google Sheets save failed: {e}")

# -----------------------------
# LIVE RECORDS PREVIEW
# -----------------------------
st.subheader("Live Records Preview")

if sheet:
    try:
        records = sheet.get_all_values()
        if records:
            st.dataframe(records, use_container_width=True)
        else:
            st.info("Sheet is empty.")
    except Exception as e:
        st.error(f"Could not read records from Google Sheets: {e}")
