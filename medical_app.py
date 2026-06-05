import streamlit as st
import requests
import pandas as pd
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

try:
    from sklearn.tree import DecisionTreeClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


st.set_page_config(page_title="Smart Health Kiosk", layout="wide")

st.title("🏥 AI-Powered Smart Health Kiosk Dashboard")

st.caption(
    "Educational health monitoring support only. "
    "This system does not replace professional medical diagnosis or treatment."
)

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
# BASIC FUNCTIONS
# =========================================================

def get_live_data():
    try:
        response = requests.get(API_URL, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def calculate_bmi(weight, height_cm):
    try:
        height_m = height_cm / 100
        return round(weight / (height_m * height_m), 2)
    except Exception:
        return 0


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
    return "ALERT" if severity == "CRITICAL" else "OK"


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


def load_google_sheet_records():
    if sheet is not None:
        try:
            records = sheet.get_all_records()
            return pd.DataFrame(records)
        except Exception as e:
            st.error("❌ ERROR LOADING GOOGLE SHEET DATA")
            st.exception(e)

    return pd.DataFrame()


# =========================================================
# AI EXPLANATION FUNCTIONS
# =========================================================

def generate_ai_interpretation(temp, hr, spo2, bmi, risk_score, severity):
    interpretation = f"""
### AI Health Interpretation

The student currently has a **{severity}** health status based on the kiosk risk scoring logic.

**Current Readings**
- Temperature: {temp} °C
- Heart Rate: {hr} BPM
- SpO₂: {spo2}%
- BMI: {bmi}
- Risk Score: {risk_score}

"""

    if severity == "CRITICAL":
        interpretation += """
**Meaning:**  
One or more readings are significantly outside the expected range. This may require urgent attention.

**Recommended Action:**  
Notify a school health officer or healthcare professional immediately. The student should be assessed promptly.
"""
    elif severity == "WARNING":
        interpretation += """
**Meaning:**  
Some readings are outside the normal range. The student may need closer monitoring.

**Recommended Action:**  
Allow the student to rest, repeat the vital signs check, and seek medical review if symptoms continue.
"""
    else:
        interpretation += """
**Meaning:**  
The readings appear generally normal based on the thresholds used by this system.

**Recommended Action:**  
Continue routine monitoring and encourage healthy habits.
"""

    return interpretation


def symptom_faq(question, temp, hr, spo2, bmi, risk_score, severity):
    question = question.lower()

    if "fever" in question or "temperature" in question:
        return (
            "A high temperature may suggest fever or infection. "
            "The student should rest, hydrate, and be monitored."
        )

    if "heart" in question or "pulse" in question or "bpm" in question:
        return (
            "A high heart rate may happen because of stress, fever, dehydration, activity, or illness. "
            "If it remains high, medical review is advised."
        )

    if "spo2" in question or "oxygen" in question or "breathing" in question:
        return (
            "Low SpO₂ may suggest reduced oxygen level. "
            "If the student has breathing difficulty or very low oxygen level, urgent care is needed."
        )

    if "bmi" in question or "weight" in question:
        return (
            "BMI estimates body weight status using height and weight. "
            "It is useful for screening but should not be used alone to judge health."
        )

    if "risk" in question or "score" in question or "severity" in question:
        return generate_ai_interpretation(temp, hr, spo2, bmi, risk_score, severity)

    if "recommend" in question or "advice" in question or "what should" in question:
        if severity == "CRITICAL":
            return "Recommendation: Alert a health officer immediately and arrange urgent clinical assessment."
        elif severity == "WARNING":
            return "Recommendation: Monitor the student, allow rest, repeat checks, and seek medical review if symptoms continue."
        else:
            return "Recommendation: Continue routine monitoring."

    if "project" in question or "system" in question:
        return (
            "This project is an AI-powered IoT student health monitoring system. "
            "It collects vital signs, calculates health risk, stores records in Google Sheets, "
            "visualizes health trends, and provides AI-style explanations and recommendations."
        )

    return (
        "I can answer questions about temperature, fever, heart rate, SpO₂, BMI, risk score, "
        "severity level, recommendations, and how this system works."
    )


# =========================================================
# SIMPLE PREDICTIVE AI MODEL
# =========================================================

def train_predictive_model(df):
    if not SKLEARN_AVAILABLE:
        return None

    required_cols = ["temperature", "heart_rate", "spo2", "bmi", "severity"]

    if df.empty:
        return None

    for col in required_cols:
        if col not in df.columns:
            return None

    try:
        clean_df = df[required_cols].dropna()

        if len(clean_df) < 5:
            return None

        X = clean_df[["temperature", "heart_rate", "spo2", "bmi"]]
        y = clean_df["severity"]

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)

        return model

    except Exception:
        return None


def predict_ai_status(model, temp, hr, spo2, bmi):
    if model is None:
        return "Prediction unavailable: not enough historical data yet."

    try:
        prediction = model.predict([[temp, hr, spo2, bmi]])[0]
        return prediction
    except Exception:
        return "Prediction unavailable."


# =========================================================
# CONNECTION STATUS
# =========================================================

live_data = get_live_data()
df_records = load_google_sheet_records()
ai_model = train_predictive_model(df_records)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.success("✅ Dashboard Online")

with col2:
    if google_connected:
        st.success("✅ Google Sheets Connected")
    else:
        st.error("❌ Google Sheets Offline")

with col3:
    if live_data:
        st.success("✅ ESP32 Connected")
    else:
        st.warning("⚠️ Waiting for ESP32 Data")

with col4:
    if ai_model is not None:
        st.success("✅ Predictive AI Ready")
    else:
        st.warning("⚠️ AI Model Pending Data")

st.divider()


# =========================================================
# DEFAULT VALUES
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
# LIVE DATA SECTION
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

else:
    st.warning("⚠️ No live ESP32 data available.")
    st.info("Use Manual Override Mode below for testing.")


# =========================================================
# DISPLAY CURRENT VITALS
# =========================================================

if current_record_available:
    st.subheader("👤 Student Information")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.info(f"Student ID: {student_id}")

    with info2:
        st.info(f"Name: {name}")

    with info3:
        st.info(f"Severity: {severity}")

    st.subheader("🩺 Vital Signs")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🌡 Temperature", f"{temperature} °C")

    with c2:
        st.metric("❤️ Heart Rate", f"{heart_rate} BPM")

    with c3:
        st.metric("🫁 SpO₂", f"{spo2}%")

    with c4:
        st.metric("⚖️ BMI", f"{bmi}")

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

st.divider()


# =========================================================
# MANUAL OVERRIDE SECTION
# =========================================================

st.header("🛠 Manual Vital Entry")

manual_mode = st.toggle("Enable Manual Vital Entry")

if manual_mode:
    with st.form("manual_form"):
        col1, col2 = st.columns(2)

        with col1:
            student_id = st.text_input("Student ID", value="ST001")
            name = st.text_input("Student Name", value="Frank Lee")
            temperature = st.number_input("Temperature °C", value=36.5)
            heart_rate = st.number_input("Heart Rate BPM", value=75)

        with col2:
            spo2 = st.number_input("SpO₂ %", value=98)
            weight = st.number_input("Weight kg", value=70.0)
            height = st.number_input("Height cm", value=170.0)
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
# AI INTERPRETATION SECTION
# =========================================================

st.header("🤖 MedExplain AI Assistant")

if current_record_available or manual_mode:
    st.markdown(
        generate_ai_interpretation(
            temperature,
            heart_rate,
            spo2,
            bmi,
            risk_score,
            severity
        )
    )

    predicted_status = predict_ai_status(
        ai_model,
        temperature,
        heart_rate,
        spo2,
        bmi
    )

    st.subheader("🧠 Predictive AI Result")
    st.info(f"Predicted Health Status: {predicted_status}")

    ai_col1, ai_col2 = st.columns([2, 1])

    with ai_col1:
        user_question = st.text_input(
            "Ask MedExplain AI",
            placeholder="Example: What does this risk score mean?"
        )

        if user_question:
            response = symptom_faq(
                user_question,
                temperature,
                heart_rate,
                spo2,
                bmi,
                risk_score,
                severity
            )
            st.info(response)

    with ai_col2:
        st.subheader("Suggested Questions")
        st.write("- What does this risk score mean?")
        st.write("- What does low SpO₂ mean?")
        st.write("- What should I do next?")
        st.write("- Explain the BMI result")
        st.write("- What does this system do?")

else:
    st.warning("AI Assistant will activate when live or manual health data is available.")

st.divider()


# =========================================================
# TREND CHARTS
# =========================================================

st.header("📈 Health Trend Analysis")

if not df_records.empty:
    chart_df = df_records.copy()

    numeric_columns = ["temperature", "heart_rate", "spo2", "bmi", "risk_score"]

    for col in numeric_columns:
        if col in chart_df.columns:
            chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce")

    available_chart_cols = [
        col for col in numeric_columns if col in chart_df.columns
    ]

    if available_chart_cols:
        st.subheader("Vital Signs and Risk Score Trends")
        st.line_chart(chart_df[available_chart_cols])

    if "severity" in chart_df.columns:
        st.subheader("Severity Distribution")
        st.bar_chart(chart_df["severity"].value_counts())

else:
    st.info("No Google Sheets records available for trend analysis yet.")

st.divider()


# =========================================================
# GOOGLE SHEET RECORDS AND DOWNLOAD
# =========================================================

st.header("📄 Google Sheets Health Records")

if google_connected:
    if not df_records.empty:
        st.dataframe(df_records, use_container_width=True)

        csv_data = df_records.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Health Records as CSV",
            data=csv_data,
            file_name="student_health_records.csv",
            mime="text/csv"
        )

    else:
        st.info("No records found yet.")

else:
    st.warning("Google Sheets not connected.")

st.divider()


# =========================================================
# AUTO REFRESH
# =========================================================

st.header("🔄 Live Monitoring")

auto_refresh = st.toggle("Enable Auto Refresh")

if auto_refresh:
    time.sleep(5)
    st.rerun()
