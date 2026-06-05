import os
import time
import zipfile
import requests
import pandas as pd
import streamlit as st
import gspread

from oauth2client.service_account import ServiceAccountCredentials
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# PAGE CONFIG
# =========================================================

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
        return round(weight / (height_m * height_m), 2)
    except Exception:
        return 0


def split_bp(bp):
    try:
        systolic, diastolic = bp.split("/")
        return int(systolic), int(diastolic)
    except Exception:
        return 120, 80


def validate_vitals(temp, hr, spo2, bmi, systolic, diastolic):
    issues = []

    if temp < 30 or temp > 45:
        issues.append("Temperature reading is unrealistic. Check sensor/manual input.")

    if hr < 30 or hr > 220:
        issues.append("Heart rate reading is unrealistic. Check sensor/manual input.")

    if spo2 < 50 or spo2 > 100:
        issues.append("SpO₂ reading is impossible or unrealistic. Check sensor/manual input.")

    if bmi < 10 or bmi > 70:
        issues.append("BMI reading is unrealistic. Check height and weight.")

    if systolic < 70 or systolic > 250:
        issues.append("Systolic blood pressure reading is unrealistic.")

    if diastolic < 40 or diastolic > 150:
        issues.append("Diastolic blood pressure reading is unrealistic.")

    return issues


def emergency_override(temp, hr, spo2, systolic, diastolic, data_issues):
    if data_issues:
        return True

    if temp >= 40:
        return True

    if spo2 <= 90:
        return True

    if hr >= 130:
        return True

    if systolic >= 180 or diastolic >= 120:
        return True

    return False


def calculate_risk(temp, hr, spo2, bmi, systolic, diastolic, data_issues=None):
    score = 0

    if data_issues:
        score += 6

    if temp >= 40:
        score += 5
    elif temp >= 38:
        score += 3
    elif temp < 35:
        score += 3

    if hr >= 130:
        score += 4
    elif hr >= 100:
        score += 2
    elif hr < 50:
        score += 2

    if spo2 <= 90:
        score += 5
    elif spo2 <= 94:
        score += 3

    if bmi >= 40:
        score += 2
    elif bmi >= 30:
        score += 1
    elif bmi < 18.5:
        score += 1

    if systolic >= 180 or diastolic >= 120:
        score += 5
    elif systolic >= 140 or diastolic >= 90:
        score += 2

    return score


def determine_severity(score, override=False):
    if override:
        return "CRITICAL"

    if score >= 8:
        return "CRITICAL"
    elif score >= 4:
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
            return pd.DataFrame(sheet.get_all_records())
        except Exception as e:
            st.error("❌ ERROR LOADING GOOGLE SHEET DATA")
            st.exception(e)

    return pd.DataFrame()


# =========================================================
# AI MODEL TRAINING
# =========================================================

@st.cache_resource
def train_predictive_model():
    try:
        possible_zip_files = [
            "vital_signs_dataset.zip",
            "archive.zip"
        ]

        zip_path = None

        for file in possible_zip_files:
            if os.path.exists(file):
                zip_path = file
                break

        if zip_path is None:
            st.warning("No AI training ZIP file found.")
            return None, None, None, None, None

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            csv_files = [
                file for file in zip_ref.namelist()
                if file.endswith(".csv")
            ]

            if not csv_files:
                st.error("No CSV file found inside ZIP.")
                return None, None, None, None, None

            with zip_ref.open(csv_files[0]) as file:
                df = pd.read_csv(file)

        rename_map = {
            "Body Temperature": "temperature",
            "Heart Rate": "heart_rate",
            "Oxygen Saturation": "spo2",
            "Derived_BMI": "bmi",
            "Systolic Blood Pressure": "systolic_bp",
            "Diastolic Blood Pressure": "diastolic_bp",
            "Age": "age",
            "Risk Category": "risk_category"
        }

        df = df.rename(columns=rename_map)

        required_columns = [
            "temperature",
            "heart_rate",
            "spo2",
            "bmi",
            "systolic_bp",
            "diastolic_bp",
            "age",
            "risk_category"
        ]

        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing dataset column: {col}")
                return None, None, None, None, None

        df = df[required_columns].dropna()

        # Limit for Streamlit Cloud stability
        if len(df) > 50000:
            df = df.sample(n=50000, random_state=42)

        X = df[
            [
                "temperature",
                "heart_rate",
                "spo2",
                "bmi",
                "systolic_bp",
                "diastolic_bp",
                "age"
            ]
        ]

        y = df["risk_category"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }

        labels = sorted(y.unique())

        cm = pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=labels),
            index=[f"Actual {label}" for label in labels],
            columns=[f"Predicted {label}" for label in labels]
        )

        feature_importance = pd.DataFrame({
            "Feature": [
                "Temperature",
                "Heart Rate",
                "SpO₂",
                "BMI",
                "Systolic BP",
                "Diastolic BP",
                "Age"
            ],
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        return model, feature_importance, len(df), metrics, cm

    except zipfile.BadZipFile:
        st.error("❌ AI MODEL TRAINING ERROR: ZIP file is invalid.")
        return None, None, None, None, None

    except Exception as e:
        st.error("❌ AI MODEL TRAINING ERROR")
        st.exception(e)
        return None, None, None, None, None


def get_confidence_level(confidence):
    if confidence >= 0.90:
        return "High Confidence"
    elif confidence >= 0.70:
        return "Moderate Confidence"
    else:
        return "Low Confidence"


def predict_ai_status(
    model,
    temp,
    hr,
    spo2,
    bmi,
    systolic,
    diastolic,
    age,
    override
):
    if override:
        return "Emergency Override", 1.0

    if model is None:
        return "Unavailable", 0

    try:
        input_data = [[
            temp,
            hr,
            spo2,
            bmi,
            systolic,
            diastolic,
            age
        ]]

        prediction = model.predict(input_data)[0]
        confidence = max(model.predict_proba(input_data)[0])

        return prediction, confidence

    except Exception:
        return "Unavailable", 0


# =========================================================
# AI INTERPRETATION
# =========================================================

def generate_ai_interpretation(
    temp,
    hr,
    spo2,
    bmi,
    systolic,
    diastolic,
    risk_score,
    severity,
    predicted_status,
    confidence,
    data_issues,
    override
):
    confidence_level = get_confidence_level(confidence)

    interpretation = f"""
### 🧠 AI Health Interpretation

**Rule-Based Status:** {severity}  
**AI Predicted Status:** {predicted_status}  
**AI Confidence:** {confidence * 100:.1f}% ({confidence_level})  

**Current Readings**
- Temperature: {temp} °C
- Heart Rate: {hr} BPM
- SpO₂: {spo2}%
- BMI: {bmi}
- Blood Pressure: {systolic}/{diastolic}
- Risk Score: {risk_score}
"""

    if data_issues:
        interpretation += """

### ⚠️ Data Quality Warning

The system detected impossible or unrealistic readings. This may indicate sensor error or incorrect manual input.

**Recommended Action:**  
Repeat the measurement and verify sensor placement before relying on the result.
"""
        return interpretation

    if override:
        interpretation += """

### 🚨 Emergency Override Activated

One or more readings crossed a high-risk emergency threshold.

**Recommended Action:**  
Alert a health officer immediately and repeat the measurement for confirmation.
"""
        return interpretation

    if severity == "CRITICAL":
        interpretation += """

**Meaning:**  
The readings suggest a potentially serious health risk.

**Recommended Action:**  
Notify a health officer immediately and arrange urgent clinical assessment.
"""
    elif severity == "WARNING":
        interpretation += """

**Meaning:**  
Some readings are outside the expected range and require close monitoring.

**Recommended Action:**  
Allow the student to rest, repeat vital checks, and seek medical review if symptoms continue.
"""
    else:
        interpretation += """

**Meaning:**  
The readings appear generally normal based on system thresholds and AI prediction.

**Recommended Action:**  
Continue routine monitoring.
"""

    return interpretation


def symptom_faq(question, temp, hr, spo2, bmi, risk_score, severity):
    question = question.lower()

    if "fever" in question or "temperature" in question:
        return "Temperature helps detect fever or abnormal body heat. Very high readings should be repeated."

    if "heart" in question or "pulse" in question or "bpm" in question:
        return "Heart rate can rise due to fever, stress, dehydration, exercise, or illness."

    if "spo2" in question or "oxygen" in question:
        return "SpO₂ measures blood oxygen level. Values above 100% are not possible and indicate sensor error."

    if "bmi" in question or "weight" in question:
        return "BMI estimates body weight status using height and weight, but it should not be used alone for diagnosis."

    if "risk" in question or "score" in question:
        return f"The current risk score is {risk_score}, giving a severity level of {severity}."

    if "project" in question or "system" in question:
        return (
            "This is an AI-powered IoT health monitoring system using vital signs, "
            "Google Sheets storage, risk scoring, machine learning prediction, "
            "model evaluation, and emergency override logic."
        )

    return "I can explain temperature, heart rate, SpO₂, BMI, risk score, AI prediction, and recommendations."


# =========================================================
# LOAD DATA
# =========================================================

live_data = get_live_data()
df_records = load_google_sheet_records()
ai_model, feature_importance, training_rows, model_metrics, confusion_df = train_predictive_model()


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
age = 18

systolic_bp = 120
diastolic_bp = 80
bmi = 0
risk_score = 0
severity = "NORMAL"
alert = "OK"

data_issues = []
override_active = False
current_record_available = False


# =========================================================
# LIVE DATA
# =========================================================

st.header("📡 Live ESP32 Health Data")

if live_data:
    student_id = live_data.get("student_id", "ST001")
    name = live_data.get("name", "Unknown")

    temperature = float(live_data.get("temperature", 0))
    heart_rate = int(live_data.get("heart_rate", 0))
    spo2 = int(live_data.get("spo2", 0))
    bp = live_data.get("bp", "120/80")
    age = int(live_data.get("age", 18))

    weight = float(live_data.get("weight", 70))
    height = float(live_data.get("height", 170))

    systolic_bp, diastolic_bp = split_bp(bp)
    bmi = calculate_bmi(weight, height)

    data_issues = validate_vitals(
        temperature,
        heart_rate,
        spo2,
        bmi,
        systolic_bp,
        diastolic_bp
    )

    override_active = emergency_override(
        temperature,
        heart_rate,
        spo2,
        systolic_bp,
        diastolic_bp,
        data_issues
    )

    risk_score = calculate_risk(
        temperature,
        heart_rate,
        spo2,
        bmi,
        systolic_bp,
        diastolic_bp,
        data_issues
    )

    severity = determine_severity(risk_score, override_active)
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
            name = st.text_input("Student Name", value="Michael Lee")
            age = st.number_input("Age", value=18, min_value=1, max_value=120)
            temperature = st.number_input("Temperature °C", value=36.5)
            heart_rate = st.number_input("Heart Rate BPM", value=75)

        with col2:
            spo2 = st.number_input("SpO₂ %", value=98)
            weight = st.number_input("Weight kg", value=70.0)
            height = st.number_input("Height cm", value=170.0)
            bp = st.text_input("Blood Pressure", value="120/80")

        submit = st.form_submit_button("Save Manual Record")

        if submit:
            systolic_bp, diastolic_bp = split_bp(bp)
            bmi = calculate_bmi(weight, height)

            data_issues = validate_vitals(
                temperature,
                heart_rate,
                spo2,
                bmi,
                systolic_bp,
                diastolic_bp
            )

            override_active = emergency_override(
                temperature,
                heart_rate,
                spo2,
                systolic_bp,
                diastolic_bp,
                data_issues
            )

            risk_score = calculate_risk(
                temperature,
                heart_rate,
                spo2,
                bmi,
                systolic_bp,
                diastolic_bp,
                data_issues
            )

            severity = determine_severity(risk_score, override_active)
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
# CURRENT SUMMARY
# =========================================================

if current_record_available or manual_mode:
    st.header("👤 Current Student Health Summary")

    if data_issues:
        st.error("🚨 Data Quality / Sensor Warning Detected")
        for issue in data_issues:
            st.warning(issue)

    if override_active:
        st.error("🚨 Emergency Override Activated")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.info(f"Student ID: {student_id}")

    with info2:
        st.info(f"Name: {name}")

    with info3:
        if severity == "CRITICAL":
            st.error(f"Severity: {severity}")
        elif severity == "WARNING":
            st.warning(f"Severity: {severity}")
        else:
            st.success(f"Severity: {severity}")

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

    bp1, bp2, bp3 = st.columns(3)

    with bp1:
        st.metric("🩸 Systolic BP", f"{systolic_bp}")

    with bp2:
        st.metric("🩸 Diastolic BP", f"{diastolic_bp}")

    with bp3:
        st.metric("🎂 Age", f"{age}")

    st.subheader("📋 Risk Analysis")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.warning(f"Risk Score: {risk_score}")

    with r2:
        st.warning(f"Severity: {severity}")

    with r3:
        if alert == "ALERT":
            st.error(f"Alert Status: {alert}")
        else:
            st.info(f"Alert Status: {alert}")

    st.progress(min(risk_score / 18, 1.0))

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
        bmi,
        systolic_bp,
        diastolic_bp,
        age,
        override_active
    )

    confidence_level = get_confidence_level(confidence)

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric("Predicted Risk Status", predicted_status)

    with p2:
        st.metric("AI Confidence", f"{confidence * 100:.1f}%")

    with p3:
        st.metric("Confidence Level", confidence_level)

    if training_rows:
        st.info(f"Model trained and evaluated using {training_rows:,} records.")

    st.markdown(
        generate_ai_interpretation(
            temperature,
            heart_rate,
            spo2,
            bmi,
            systolic_bp,
            diastolic_bp,
            risk_score,
            severity,
            predicted_status,
            confidence,
            data_issues,
            override_active
        )
    )

    st.subheader("💬 Symptom FAQ Assistant")

    user_question = st.text_input(
        "Ask MedExplain AI",
        placeholder="Example: What does this AI prediction mean?"
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
# MODEL EVALUATION
# =========================================================

st.header("📊 AI Model Evaluation")

if model_metrics is not None:
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Accuracy", f"{model_metrics['Accuracy'] * 100:.2f}%")

    with m2:
        st.metric("Precision", f"{model_metrics['Precision'] * 100:.2f}%")

    with m3:
        st.metric("Recall", f"{model_metrics['Recall'] * 100:.2f}%")

    with m4:
        st.metric("F1 Score", f"{model_metrics['F1 Score'] * 100:.2f}%")

    st.subheader("Confusion Matrix")
    st.dataframe(confusion_df, use_container_width=True)

else:
    st.info("Model evaluation will appear when the AI model is trained.")

st.divider()


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.header("📌 AI Feature Importance")

if feature_importance is not None:
    feature_importance["Importance (%)"] = (
        feature_importance["Importance"] * 100
    ).round(2)

    st.dataframe(feature_importance, use_container_width=True)

    st.bar_chart(
        feature_importance.set_index("Feature")["Importance (%)"]
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
