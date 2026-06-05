import streamlit as st
import requests
import pandas as pd
import time
import gspread
import zipfile
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(
    page_title="AI Smart Health Kiosk",
    layout="wide"
)

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
        bmi = weight / (height_m * height_m)
        return round(bmi, 2)
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
    if severity == "CRITICAL":
        return "ALERT"
    return "OK"


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
# AI MODEL TRAINING FROM ZIP DATASET
# =========================================================

@st.cache_resource
def train_predictive_model():
    try:
        zip_path = "archive.zip"

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            csv_files = [
                file for file in zip_ref.namelist()
                if file.endswith(".csv")
            ]

            if not csv_files:
                st.error("No CSV file found inside archive.zip")
                return None, None, None

            csv_file = csv_files[0]

            with zip_ref.open(csv_file) as file:
                df = pd.read_csv(file)

        required_columns = [
            "Body Temperature",
            "Heart Rate",
            "Oxygen Saturation",
            "Derived_BMI",
            "Risk Category"
        ]

        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing column in dataset: {col}")
                return None, None, None

        df = df[required_columns].dropna()

        X = df[
            [
                "Body Temperature",
                "Heart Rate",
                "Oxygen Saturation",
                "Derived_BMI"
            ]
        ]

        y = df["Risk Category"]

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X, y)

        feature_importance = pd.DataFrame({
            "Feature": [
                "Body Temperature",
                "Heart Rate",
                "Oxygen Saturation",
                "BMI"
            ],
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        return model, feature_importance, len(df)

    except Exception as e:
        st.error("❌ AI MODEL TRAINING ERROR")
        st.exception(e)
        return None, None, None


def predict_ai_status(model, temperature, heart_rate, spo2, bmi):
    if model is None:
        return "Unavailable", 0

    try:
        input_data = [[
            temperature,
            heart_rate,
            spo2,
            bmi
        ]]

        prediction = model.predict(input_data)[0]
        confidence = max(model.predict_proba(input_data)[0])

        return prediction, confidence

    except Exception:
        return "Unavailable", 0


# =========================================================
# AI EXPLANATION FUNCTIONS
# =========================================================

def generate_ai_interpretation(
    temp,
    hr,
    spo2,
    bmi,
    risk_score,
    severity,
    predicted_status,
    confidence
):
    interpretation = f"""
### 🧠 AI Health Interpretation

The student's current rule-based health status is **{severity}**.

The machine-learning model predicts: **{predicted_status}**  
Prediction confidence: **{confidence * 100:.1f}%**

**Current Readings**
- Temperature: {temp} °C
- Heart Rate: {hr} BPM
- SpO₂: {spo2}%
- BMI: {bmi}
- Risk Score: {risk_score}
"""

    if severity == "CRITICAL" or str(predicted_status).lower() in ["critical", "high risk", "high"]:
        interpretation += """
**Meaning:**  
The readings suggest a potentially serious health risk.

**Recommended Action:**  
Notify a health officer immediately and arrange urgent clinical assessment.
"""

    elif severity == "WARNING" or str(predicted_status).lower() in ["warning", "medium risk", "moderate risk", "medium"]:
        interpretation += """
**Meaning:**  
Some readings are outside the expected range and require closer monitoring.

**Recommended Action:**  
Allow the student to rest, repeat the vital signs check, and seek medical review if symptoms continue.
"""

    else:
        interpretation += """
**Meaning:**  
The readings appear generally normal based on the current system thresholds and AI prediction.

**Recommended Action:**  
Continue routine monitoring.
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
            "A high heart rate may occur due to stress, fever, dehydration, activity, or illness. "
            "If it remains high, medical review is advised."
        )

    if "spo2" in question or "oxygen" in question or "breathing" in question:
        return (
            "Low SpO₂ may indicate reduced oxygen level. "
            "If the student has breathing difficulty or very low oxygen level, urgent care is needed."
        )

    if "bmi" in question or "weight" in question:
        return (
            "BMI estimates body weight status from height and weight. "
            "It is useful for screening but should not be used alone to judge health."
        )

    if "risk" in question or "score" in question or "severity" in question:
        return (
            f"The current risk score is {risk_score}, giving a severity level of {severity}. "
            "This is based on temperature, heart rate, and oxygen saturation thresholds."
        )

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
            "visualizes trends, and uses machine learning to predict health risk status."
        )

    return (
        "I can answer questions about temperature, fever, heart rate, SpO₂, BMI, risk score, "
        "severity level, recommendations, and how this AI health monitoring system works."
    )


# =========================================================
# LOAD DATA AND TRAIN AI
# =========================================================

live_data = get_live_data()
df_records = load_google_sheet_records()
ai_model, feature_importance, training_rows = train_predictive_model()


# =========================================================
# CONNECTION STATUS
# =========================================================

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
        st.warning("⚠️ AI Model Not Ready")

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

else:
    st.warning("⚠️ No live ESP32 data available.")
    st.info("Use Manual Vital Entry below for testing.")

st.divider()


# =========================================================
# MANUAL ENTRY
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
# DISPLAY CURRENT VITALS
# =========================================================

if current_record_available or manual_mode:
    st.header("👤 Current Student Health Summary")

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

    st.progress(min(risk_score / 7, 1.0))

st.divider()


# =========================================================
# AI PREDICTIVE SECTION
# =========================================================

st.header("🤖 AI Predictive Health Assessment")

if current_record_available or manual_mode:
    predicted_status, confidence = predict_ai_status(
        ai_model,
        temperature,
        heart_rate,
        spo2,
        bmi
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric("Predicted Risk Status", predicted_status)

    with p2:
        st.metric("AI Confidence", f"{confidence * 100:.1f}%")

    with p3:
        if training_rows:
            st.metric("Training Records", f"{training_rows:,}")
        else:
            st.metric("Training Records", "Unavailable")

    st.markdown(
        generate_ai_interpretation(
            temperature,
            heart_rate,
            spo2,
            bmi,
            risk_score,
            severity,
            predicted_status,
            confidence
        )
    )

    st.subheader("💬 Symptom FAQ Assistant")

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

else:
    st.warning("AI assessment will activate when live or manual data is available.")

st.divider()


# =========================================================
# AI FEATURE IMPORTANCE
# =========================================================

st.header("📊 AI Model Feature Importance")

if feature_importance is not None:
    st.dataframe(feature_importance, use_container_width=True)
    st.bar_chart(
        feature_importance.set_index("Feature")
    )
else:
    st.info("Feature importance will appear when the AI model is trained.")

st.divider()


# =========================================================
# GOOGLE SHEET TREND ANALYSIS
# =========================================================

st.header("📈 Google Sheets Health Trend Analysis")

if not df_records.empty:
    chart_df = df_records.copy()

    numeric_columns = [
        "temperature",
        "heart_rate",
        "spo2",
        "bmi",
        "risk_score"
    ]

    for col in numeric_columns:
        if col in chart_df.columns:
            chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce")

    available_chart_cols = [
        col for col in numeric_columns
        if col in chart_df.columns
    ]

    if available_chart_cols:
        st.line_chart(chart_df[available_chart_cols])

    if "severity" in chart_df.columns:
        st.subheader("Severity Distribution")
        st.bar_chart(chart_df["severity"].value_counts())

else:
    st.info("No Google Sheets records available yet.")

st.divider()


# =========================================================
# GOOGLE SHEET RECORDS
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
