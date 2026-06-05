import os
import time
import zipfile
import requests
import pandas as pd
import streamlit as st
import gspread

from oauth2client.service_account import ServiceAccountCredentials
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


st.set_page_config(page_title="Hybrid AI Smart Health Kiosk", layout="wide")

st.title("🏥 Hybrid AI-Powered Smart Health Kiosk Dashboard")

st.caption(
    "This system uses Random Forest, Artificial Neural Network, Fuzzy Logic, "
    "Ensemble AI, and a symptom-screening assistant for educational health monitoring support only."
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
# FUZZY LOGIC ENGINE
# =========================================================

def fuzzy_membership_low(value, low, high):
    if value <= low:
        return 1
    if value >= high:
        return 0
    return (high - value) / (high - low)


def fuzzy_membership_high(value, low, high):
    if value <= low:
        return 0
    if value >= high:
        return 1
    return (value - low) / (high - low)


def fuzzy_health_risk(temp, hr, spo2, bmi, systolic, diastolic):
    temp_high = fuzzy_membership_high(temp, 37.5, 40)
    hr_high = fuzzy_membership_high(hr, 100, 130)
    spo2_low = fuzzy_membership_low(spo2, 90, 95)
    bmi_high = fuzzy_membership_high(bmi, 30, 40)

    bp_high = max(
        fuzzy_membership_high(systolic, 140, 180),
        fuzzy_membership_high(diastolic, 90, 120)
    )

    fuzzy_score = (
        temp_high * 0.25 +
        hr_high * 0.20 +
        spo2_low * 0.30 +
        bmi_high * 0.10 +
        bp_high * 0.15
    )

    if fuzzy_score >= 0.65:
        return "CRITICAL", fuzzy_score
    elif fuzzy_score >= 0.35:
        return "WARNING", fuzzy_score
    else:
        return "NORMAL", fuzzy_score


# =========================================================
# AI MODEL TRAINING
# =========================================================

@st.cache_resource
def train_hybrid_models():
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
            return None, None, None, None, None, None, None

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            csv_files = [
                file for file in zip_ref.namelist()
                if file.endswith(".csv")
            ]

            if not csv_files:
                st.error("No CSV file found inside ZIP.")
                return None, None, None, None, None, None, None

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
                return None, None, None, None, None, None, None

        df = df[required_columns].dropna()

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

        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            class_weight="balanced"
        )

        ann_model = Pipeline([
            ("scaler", StandardScaler()),
            ("ann", MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation="relu",
                solver="adam",
                max_iter=300,
                random_state=42
            ))
        ])

        rf_model.fit(X_train, y_train)
        ann_model.fit(X_train, y_train)

        rf_pred = rf_model.predict(X_test)
        ann_pred = ann_model.predict(X_test)

        labels = sorted(y.unique())

        rf_metrics = {
            "Accuracy": accuracy_score(y_test, rf_pred),
            "Precision": precision_score(y_test, rf_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, rf_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, rf_pred, average="weighted", zero_division=0)
        }

        ann_metrics = {
            "Accuracy": accuracy_score(y_test, ann_pred),
            "Precision": precision_score(y_test, ann_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, ann_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, ann_pred, average="weighted", zero_division=0)
        }

        rf_cm = pd.DataFrame(
            confusion_matrix(y_test, rf_pred, labels=labels),
            index=[f"Actual {label}" for label in labels],
            columns=[f"Predicted {label}" for label in labels]
        )

        ann_cm = pd.DataFrame(
            confusion_matrix(y_test, ann_pred, labels=labels),
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
            "Importance": rf_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        return (
            rf_model,
            ann_model,
            feature_importance,
            len(df),
            rf_metrics,
            ann_metrics,
            {
                "Random Forest": rf_cm,
                "ANN": ann_cm
            }
        )

    except zipfile.BadZipFile:
        st.error("❌ AI MODEL TRAINING ERROR: ZIP file is invalid.")
        return None, None, None, None, None, None, None

    except Exception as e:
        st.error("❌ AI MODEL TRAINING ERROR")
        st.exception(e)
        return None, None, None, None, None, None, None


def get_confidence_level(confidence):
    if confidence >= 0.90:
        return "High Confidence"
    elif confidence >= 0.70:
        return "Moderate Confidence"
    else:
        return "Low Confidence"


def majority_vote(predictions):
    return max(set(predictions), key=predictions.count)


def predict_hybrid_ai(
    rf_model,
    ann_model,
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
        return {
            "Random Forest": "Emergency Override",
            "ANN": "Emergency Override",
            "Fuzzy Logic": "Emergency Override",
            "Ensemble": "CRITICAL",
            "Confidence": 1.0
        }

    if rf_model is None or ann_model is None:
        return {
            "Random Forest": "Unavailable",
            "ANN": "Unavailable",
            "Fuzzy Logic": "Unavailable",
            "Ensemble": "Unavailable",
            "Confidence": 0.0
        }

    input_data = [[
        temp,
        hr,
        spo2,
        bmi,
        systolic,
        diastolic,
        age
    ]]

    rf_prediction = rf_model.predict(input_data)[0]
    ann_prediction = ann_model.predict(input_data)[0]

    rf_confidence = max(rf_model.predict_proba(input_data)[0])
    ann_confidence = max(ann_model.predict_proba(input_data)[0])

    fuzzy_prediction, fuzzy_score = fuzzy_health_risk(
        temp,
        hr,
        spo2,
        bmi,
        systolic,
        diastolic
    )

    ensemble_prediction = majority_vote([
        str(rf_prediction),
        str(ann_prediction),
        str(fuzzy_prediction)
    ])

    ensemble_confidence = (rf_confidence + ann_confidence + fuzzy_score) / 3

    return {
        "Random Forest": rf_prediction,
        "ANN": ann_prediction,
        "Fuzzy Logic": fuzzy_prediction,
        "Ensemble": ensemble_prediction,
        "Confidence": ensemble_confidence
    }


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
    hybrid_result,
    data_issues,
    override
):
    confidence = hybrid_result["Confidence"]
    confidence_level = get_confidence_level(confidence)

    interpretation = f"""
### 🧠 Hybrid AI Health Interpretation

**Rule-Based Status:** {severity}  
**Random Forest Prediction:** {hybrid_result["Random Forest"]}  
**ANN Prediction:** {hybrid_result["ANN"]}  
**Fuzzy Logic Prediction:** {hybrid_result["Fuzzy Logic"]}  
**Final Ensemble Decision:** {hybrid_result["Ensemble"]}  
**Ensemble Confidence:** {confidence * 100:.1f}% ({confidence_level})

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

    if severity == "CRITICAL" or str(hybrid_result["Ensemble"]).upper() in ["CRITICAL", "HIGH RISK", "HIGH"]:
        interpretation += """

**Meaning:**  
The hybrid AI system indicates a potentially serious health risk.

**Recommended Action:**  
Notify a health officer immediately and arrange clinical assessment.
"""
    elif severity == "WARNING" or str(hybrid_result["Ensemble"]).upper() in ["WARNING", "MODERATE RISK", "MEDIUM RISK", "MEDIUM"]:
        interpretation += """

**Meaning:**  
Some readings suggest possible health concern and require monitoring.

**Recommended Action:**  
Allow rest, repeat vital checks, and seek medical review if symptoms continue.
"""
    else:
        interpretation += """

**Meaning:**  
The readings appear generally normal based on rule-based and hybrid AI assessment.

**Recommended Action:**  
Continue routine monitoring.
"""

    return interpretation


# =========================================================
# CLINICAL INTERVIEW ASSISTANT
# =========================================================

def clinical_interview_assistant(severity, alert):
    st.subheader("🩺 Clinical Interview Assistant")

    if severity == "CRITICAL" or alert == "ALERT":
        st.error("🚨 Critical condition detected. Further questioning is skipped.")
        st.warning("Immediate notification to a doctor or health officer is recommended.")
        return

    st.info("Patient is stable enough for additional symptom screening.")

    with st.form("clinical_interview_form"):
        headache = st.selectbox("Are you having headache?", ["No", "Yes"])
        cough = st.selectbox("Are you coughing?", ["No", "Yes"])
        catarrh = st.selectbox("Do you have catarrh or runny nose?", ["No", "Yes"])
        sore_throat = st.selectbox("Do you have sore throat?", ["No", "Yes"])
        chest_pain = st.selectbox("Do you have chest pain?", ["No", "Yes"])
        breathing = st.selectbox("Are you having difficulty breathing?", ["No", "Yes"])
        dizziness = st.selectbox("Are you feeling dizzy or weak?", ["No", "Yes"])
        nausea = st.selectbox("Do you feel nausea or vomiting?", ["No", "Yes"])
        body_pain = st.selectbox("Do you have body pain or weakness?", ["No", "Yes"])

        duration = st.selectbox(
            "How long have you had these symptoms?",
            ["No symptoms", "Less than 1 day", "1–3 days", "More than 3 days"]
        )

        medication = st.text_input("Have you taken any medication? If yes, specify.")
        extra_notes = st.text_area("Any other complaint?")

        submit_interview = st.form_submit_button("Generate Symptom Summary")

    if submit_interview:
        symptoms = []

        if headache == "Yes":
            symptoms.append("headache")
        if cough == "Yes":
            symptoms.append("cough")
        if catarrh == "Yes":
            symptoms.append("catarrh/runny nose")
        if sore_throat == "Yes":
            symptoms.append("sore throat")
        if chest_pain == "Yes":
            symptoms.append("chest pain")
        if breathing == "Yes":
            symptoms.append("breathing difficulty")
        if dizziness == "Yes":
            symptoms.append("dizziness/weakness")
        if nausea == "Yes":
            symptoms.append("nausea/vomiting")
        if body_pain == "Yes":
            symptoms.append("body pain/weakness")

        st.subheader("🧠 AI Symptom Screening Summary")

        if not symptoms:
            st.success("No major symptoms were reported during the interview.")
            st.write("Recommendation: Continue routine monitoring.")
        else:
            st.warning(f"Reported symptoms: {', '.join(symptoms)}")
            st.write(f"Symptom duration: {duration}")

            if medication:
                st.write(f"Medication reported: {medication}")
            else:
                st.write("No medication was reported.")

            if extra_notes:
                st.write(f"Additional complaint: {extra_notes}")

            if breathing == "Yes" or chest_pain == "Yes":
                st.error(
                    "Possible serious symptom pattern detected. "
                    "Medical review is recommended as soon as possible."
                )
            elif cough == "Yes" and catarrh == "Yes" and headache == "Yes":
                st.warning(
                    "Symptoms may suggest a respiratory or flu-like illness pattern. "
                    "Rest, hydration, and medical review are recommended if symptoms persist."
                )
            elif headache == "Yes" and dizziness == "Yes":
                st.warning(
                    "Headache with dizziness may require closer monitoring, especially if symptoms continue."
                )
            elif nausea == "Yes" and dizziness == "Yes":
                st.warning(
                    "Nausea with dizziness may suggest dehydration or general weakness. "
                    "Further observation is recommended."
                )
            else:
                st.info(
                    "Mild symptoms reported. Continue monitoring and seek medical review if symptoms worsen."
                )

        st.caption(
            "This symptom screening is for support only and does not provide a medical diagnosis."
        )


def symptom_faq(question, temp, hr, spo2, bmi, risk_score, severity):
    question = question.lower()

    if "ann" in question or "neural" in question:
        return "The ANN is an Artificial Neural Network using MLPClassifier with hidden layers to learn patterns from vital signs."

    if "fuzzy" in question:
        return "The fuzzy logic engine converts readings like high temperature, low SpO₂, and high blood pressure into fuzzy risk levels using membership functions."

    if "random forest" in question:
        return "Random Forest is a supervised machine-learning classifier that combines many decision trees to predict health risk category."

    if "ensemble" in question:
        return "The ensemble combines Random Forest, ANN, and Fuzzy Logic predictions using majority voting."

    if "spo2" in question or "oxygen" in question:
        return "SpO₂ measures blood oxygen saturation. Values above 100% are not physiologically possible and suggest sensor error."

    if "risk" in question or "score" in question:
        return f"The current risk score is {risk_score}, giving a severity level of {severity}."

    return "I can explain ANN, Random Forest, Fuzzy Logic, Ensemble AI, temperature, heart rate, SpO₂, BMI, and risk score."


# =========================================================
# LOAD DATA
# =========================================================

live_data = get_live_data()
df_records = load_google_sheet_records()

(
    rf_model,
    ann_model,
    feature_importance,
    training_rows,
    rf_metrics,
    ann_metrics,
    confusion_matrices
) = train_hybrid_models()


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
    if rf_model is not None and ann_model is not None:
        st.success("✅ Hybrid AI Ready")
    else:
        st.warning("⚠️ Hybrid AI Not Ready")

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
# HYBRID AI SECTION
# =========================================================

st.header("🤖 Hybrid AI Predictive Health Assessment")

if current_record_available or manual_mode:
    hybrid_result = predict_hybrid_ai(
        rf_model,
        ann_model,
        temperature,
        heart_rate,
        spo2,
        bmi,
        systolic_bp,
        diastolic_bp,
        age,
        override_active
    )

    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.metric("Random Forest", hybrid_result["Random Forest"])

    with h2:
        st.metric("ANN", hybrid_result["ANN"])

    with h3:
        st.metric("Fuzzy Logic", hybrid_result["Fuzzy Logic"])

    with h4:
        st.metric("Final Ensemble", hybrid_result["Ensemble"])

    st.metric(
        "Ensemble Confidence",
        f"{hybrid_result['Confidence'] * 100:.1f}%"
    )

    if training_rows:
        st.info(f"Hybrid AI trained and evaluated using {training_rows:,} records.")

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
            hybrid_result,
            data_issues,
            override_active
        )
    )

    clinical_interview_assistant(severity, alert)

    st.subheader("💬 MedExplain AI Assistant")

    user_question = st.text_input(
        "Ask MedExplain AI",
        placeholder="Example: What is the difference between ANN and Fuzzy Logic?"
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
    st.warning("Hybrid AI assessment will activate when live or manual data is available.")

st.divider()


# =========================================================
# MODEL EVALUATION
# =========================================================

st.header("📊 AI Model Evaluation")

if rf_metrics is not None and ann_metrics is not None:
    comparison_df = pd.DataFrame([
        {"Model": "Random Forest", **rf_metrics},
        {"Model": "Artificial Neural Network", **ann_metrics}
    ])

    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        comparison_df[col] = (comparison_df[col] * 100).round(2)

    st.subheader("Model Performance Comparison")
    st.dataframe(comparison_df, use_container_width=True)

    st.subheader("Random Forest Confusion Matrix")
    st.dataframe(confusion_matrices["Random Forest"], use_container_width=True)

    st.subheader("ANN Confusion Matrix")
    st.dataframe(confusion_matrices["ANN"], use_container_width=True)

else:
    st.info("Model evaluation will appear when the hybrid AI model is trained.")

st.divider()


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.header("📌 Random Forest Feature Importance")

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
