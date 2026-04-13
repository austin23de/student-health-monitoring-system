import streamlit as st
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Smart Health Monitoring Kiosk", layout="centered")
st.title("NEW VERSION - Smart Health Monitoring Kiosk")

# -----------------------------
# GOOGLE SHEETS CONNECTION
# -----------------------------
def connect_to_google_sheet():
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]

        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)

        spreadsheet = client.open("Student Health Monitoring System")
        worksheet = spreadsheet.worksheet("Live_Records")

        return worksheet, None

    except Exception as e:
        return None, str(e)

sheet, connection_error = connect_to_google_sheet()

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

# Heart rate
if heart_rate > 130 or heart_rate < 50:
    risk_score += 3
    warnings.append("Critical heart rate")
elif heart_rate > 100:
    risk_score += 2
    warnings.append("High heart rate")

# Blood pressure
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
    st.write("⚠️ Issues Detected:")
    for item in warnings:
        st.write(f"- {item}")

if severity == "CRITICAL":
    st.error("🚨 Immediate medical attention required!")
elif severity == "WARNING":
    st.warning("⚠️ Monitor patient closely")
else:
    st.success("✅ Patient is stable")

# -----------------------------
# DEBUG SECTION
# -----------------------------
st.subheader("Debug Information")

st.write("Student ID:", student_id)
st.write("Name:", name)
st.write("Will save to sheet:", "Yes" if sheet else "No")
st.write("Google connection error:", connection_error if connection_error else "None")

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

        st.write("Data about to be saved:")
        st.write(row_data)

        try:
            sheet.append_row(row_data)
            st.success("Record saved to Google Sheets successfully.")
        except Exception as e:
            st.error(f"Google Sheets save failed: {e}")

# -----------------------------
# VIEW SAVED RECORDS
# -----------------------------
st.subheader("Live Records Preview")

if sheet:
    try:
        records = sheet.get_all_values()
        if records:
            st.dataframe(records)
        else:
            st.info("Sheet is empty.")
    except Exception as e:
        st.error(f"Could not read records from Google Sheets: {e}")