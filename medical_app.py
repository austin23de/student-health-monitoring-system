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


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Hybrid AI Clinical Triage System",
    layout="wide"
)

st.title("🏥 Hybrid AI-Based Student Health Monitoring and Clinical Triage System")

st.caption(
    "Educational triage-support system only. "
    "This system does not provide a final medical diagnosis and does not replace a qualified healthcare professional."
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
# BASIC HEALTH FUNCTIONS
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
# HYBRID AI MODEL TRAINING
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
# =========================================================
# HYBRID AI PREDICTION FUNCTIONS
# =========================================================

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

    input_data = [[temp, hr, spo2, bmi, systolic, diastolic, age]]

    rf_prediction = rf_model.predict(input_data)[0]
    ann_prediction = ann_model.predict(input_data)[0]

    rf_confidence = max(rf_model.predict_proba(input_data)[0])
    ann_confidence = max(ann_model.predict_proba(input_data)[0])

    fuzzy_prediction, fuzzy_score = fuzzy_health_risk(
        temp, hr, spo2, bmi, systolic, diastolic
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
# EXPANDED DISEASE KNOWLEDGE BASE
# =========================================================

DISEASE_KNOWLEDGE_BASE = {
    "Common Cold": {
        "symptoms": ["cough", "catarrh", "runny nose", "sore throat", "sneezing", "mild headache"],
        "red_flags": ["breathing difficulty", "chest pain"],
        "follow_up": [
            "Do you have sneezing?",
            "Is your runny nose watery?",
            "Are symptoms mild and gradual?"
        ],
        "advice": "Common cold-like symptoms are usually mild. Rest, hydration, and monitoring are recommended."
    },
    "Influenza / Flu": {
        "symptoms": ["fever", "cough", "sore throat", "headache", "body pain", "fatigue", "chills"],
        "red_flags": ["breathing difficulty", "chest pain", "persistent high fever"],
        "follow_up": [
            "Do you have body pain?",
            "Do you feel unusually weak or tired?",
            "Do you have chills?"
        ],
        "advice": "Flu-like symptoms may require rest, hydration, and medical review if symptoms persist or worsen."
    },
    "COVID-19-like Illness": {
        "symptoms": ["fever", "cough", "sore throat", "loss of smell", "loss of taste", "breathing difficulty", "fatigue"],
        "red_flags": ["breathing difficulty", "low spo2", "chest pain"],
        "follow_up": [
            "Have you lost your sense of smell?",
            "Have you lost your sense of taste?",
            "Have you had recent contact with a sick person?"
        ],
        "advice": "COVID-like symptoms may require isolation guidance, monitoring, and medical review."
    },
    "Allergic Rhinitis": {
        "symptoms": ["catarrh", "runny nose", "sneezing", "itchy eyes", "watery eyes"],
        "red_flags": ["breathing difficulty"],
        "follow_up": [
            "Do your eyes itch?",
            "Are symptoms worse around dust or pollen?",
            "Do you have frequent sneezing?"
        ],
        "advice": "Allergy-like symptoms may be triggered by dust, pollen, or environmental exposure."
    },
    "Tonsillitis / Throat Infection": {
        "symptoms": ["sore throat", "fever", "difficulty swallowing", "headache", "body pain"],
        "red_flags": ["difficulty breathing", "severe throat swelling"],
        "follow_up": [
            "Do you have difficulty swallowing?",
            "Is the throat pain severe?",
            "Do you have fever?"
        ],
        "advice": "Throat infection-like symptoms should be reviewed if pain is severe, persistent, or associated with fever."
    },
    "Pneumonia / Lower Respiratory Concern": {
        "symptoms": ["cough", "fever", "chest pain", "breathing difficulty", "low spo2", "fatigue"],
        "red_flags": ["breathing difficulty", "low spo2", "chest pain"],
        "follow_up": [
            "Do you have chest pain?",
            "Are you finding it difficult to breathe?",
            "Is your cough persistent or worsening?"
        ],
        "advice": "Breathing difficulty, chest pain, or low oxygen level requires urgent medical review."
    },
    "Malaria-like Illness": {
        "symptoms": ["fever", "chills", "headache", "body pain", "fatigue", "nausea", "sweating"],
        "red_flags": ["confusion", "persistent vomiting", "severe weakness"],
        "follow_up": [
            "Do you have chills?",
            "Do you have sweating episodes?",
            "Do you feel weak or nauseous?"
        ],
        "advice": "Malaria-like symptoms require proper clinical testing before treatment."
    },
    "Gastrointestinal Illness": {
        "symptoms": ["nausea", "vomiting", "stomach pain", "diarrhea", "weakness"],
        "red_flags": ["blood in stool", "severe dehydration", "persistent vomiting"],
        "follow_up": [
            "Do you have stomach pain?",
            "Have you vomited?",
            "Do you have diarrhea?"
        ],
        "advice": "Gastrointestinal symptoms may require hydration and clinical review if persistent."
    },
    "Asthma-like Respiratory Episode": {
        "symptoms": ["breathing difficulty", "wheezing", "cough", "chest tightness"],
        "red_flags": ["severe breathing difficulty", "low spo2", "blue lips"],
        "follow_up": [
            "Do you hear wheezing when breathing?",
            "Do you have chest tightness?",
            "Have you had similar episodes before?"
        ],
        "advice": "Breathing difficulty or wheezing should be reviewed urgently if severe or worsening."
    },
    "Sinusitis-like Illness": {
        "symptoms": ["catarrh", "runny nose", "headache", "facial pain", "blocked nose"],
        "red_flags": ["severe headache", "swelling around eyes"],
        "follow_up": [
            "Do you have facial pain?",
            "Is your nose blocked?",
            "Has it lasted more than one week?"
        ],
        "advice": "Sinus-like symptoms may require review if severe, prolonged, or associated with facial pain."
    }
}


SYMPTOM_OPTIONS = sorted(list({
    symptom
    for info in DISEASE_KNOWLEDGE_BASE.values()
    for symptom in info["symptoms"] + info["red_flags"]
}))


def normalize_symptom(symptom):
    return symptom.lower().strip()


def calculate_condition_probabilities(selected_symptoms):
    selected = [normalize_symptom(s) for s in selected_symptoms]
    results = []

    for condition, info in DISEASE_KNOWLEDGE_BASE.items():
        disease_symptoms = [normalize_symptom(s) for s in info["symptoms"]]
        red_flags = [normalize_symptom(s) for s in info["red_flags"]]

        matched = [s for s in selected if s in disease_symptoms]
        matched_red_flags = [s for s in selected if s in red_flags]

        symptom_score = len(matched) / len(disease_symptoms) if disease_symptoms else 0
        red_flag_bonus = min(len(matched_red_flags) * 0.15, 0.30)

        probability = min(symptom_score + red_flag_bonus, 1.0)

        results.append({
            "Possible Condition": condition,
            "Match Score (%)": round(probability * 100, 1),
            "Matched Symptoms": ", ".join(matched) if matched else "None",
            "Red Flags": ", ".join(matched_red_flags) if matched_red_flags else "None",
            "Advice": info["advice"]
        })

    return pd.DataFrame(results).sort_values(
        by="Match Score (%)",
        ascending=False
    )


def get_dynamic_followup_questions(top_conditions):
    questions = []
    for condition in top_conditions:
        questions.extend(DISEASE_KNOWLEDGE_BASE[condition]["follow_up"])

    unique_questions = []
    for q in questions:
        if q not in unique_questions:
            unique_questions.append(q)

    return unique_questions[:8]


def map_followup_to_symptoms(question):
    q = question.lower()
    symptom_map = []

    if "smell" in q:
        symptom_map.append("loss of smell")
    if "taste" in q:
        symptom_map.append("loss of taste")
    if "body pain" in q:
        symptom_map.append("body pain")
    if "weak" in q or "tired" in q:
        symptom_map.append("fatigue")
    if "chills" in q:
        symptom_map.append("chills")
    if "chest" in q:
        symptom_map.append("chest pain")
    if "breathe" in q or "breathing" in q:
        symptom_map.append("breathing difficulty")
    if "itch" in q:
        symptom_map.append("itchy eyes")
    if "sneezing" in q:
        symptom_map.append("sneezing")
    if "swallowing" in q:
        symptom_map.append("difficulty swallowing")
    if "vomited" in q or "vomit" in q:
        symptom_map.append("vomiting")
    if "diarrhea" in q:
        symptom_map.append("diarrhea")
    if "stomach" in q:
        symptom_map.append("stomach pain")
    if "nauseous" in q or "nausea" in q:
        symptom_map.append("nausea")
    if "wheezing" in q:
        symptom_map.append("wheezing")
    if "facial pain" in q:
        symptom_map.append("facial pain")
    if "blocked" in q:
        symptom_map.append("blocked nose")

    return symptom_map


# =========================================================
# MEDICAL EXPLANATION ASSISTANT
# =========================================================

def medical_explanation_assistant(question, risk_score, severity):
    question = question.lower()

    disease_keywords = {
        "sore throat": "A sore throat can occur with common cold, flu, allergies, tonsillitis, throat irritation, or other upper respiratory infections.",
        "cough": "Cough can occur with common cold, flu, allergies, asthma-like irritation, COVID-like illness, or lower respiratory infection.",
        "catarrh": "Catarrh or runny nose can occur with common cold, allergic rhinitis, sinus irritation, or flu-like illness.",
        "runny nose": "Runny nose may occur with common cold, allergies, sinus irritation, or flu-like illness.",
        "fever": "Fever may suggest infection or inflammation. Persistent or high fever requires clinical review.",
        "headache": "Headache may occur with fever, flu-like illness, sinus symptoms, dehydration, or stress.",
        "breathing": "Breathing difficulty is a red-flag symptom and should be reviewed urgently.",
        "chest pain": "Chest pain is a red-flag symptom and should be reviewed urgently.",
        "spo2": "SpO₂ measures oxygen saturation. Low values can be serious, while values above 100% suggest sensor error."
    }

    for keyword, response in disease_keywords.items():
        if keyword in question:
            return response + " Use the Dynamic Clinical Interview Assistant to narrow possible causes."

    if "risk" in question or "score" in question:
        return f"The current risk score is {risk_score}, giving a severity level of {severity}."

    if "ann" in question or "neural" in question:
        return "The ANN learns patterns from vital signs to classify health risk."

    if "fuzzy" in question:
        return "Fuzzy logic handles uncertainty by converting readings into degrees such as mildly high, moderately high, or critically high."

    if "random forest" in question:
        return "Random Forest combines multiple decision trees to classify health risk."

    if "ensemble" in question:
        return "The ensemble combines Random Forest, ANN, and fuzzy logic into one final decision."

    return (
        "Ask about symptoms such as cough, sore throat, fever, catarrh, breathing difficulty, "
        "SpO₂, risk score, or use the clinical interview assistant for condition ranking."
    )


# =========================================================
# CLINICAL INTERVIEW ASSISTANT
# =========================================================

def clinical_interview_assistant(severity, alert):
    st.subheader("🩺 Dynamic Clinical Interview Assistant")

    if severity == "CRITICAL" or alert == "ALERT":
        st.error("🚨 Critical condition detected. Further questioning is skipped.")
        st.warning("Immediate notification to a doctor or health officer is recommended.")
        return

    st.info(
        "Patient is stable enough for symptom screening. "
        "This assistant ranks possible conditions and asks follow-up questions."
    )

    selected_symptoms = st.multiselect(
        "Select the symptoms the patient is experiencing:",
        SYMPTOM_OPTIONS
    )

    duration = st.selectbox(
        "How long have the symptoms been present?",
        [
            "Less than 1 day",
            "1–3 days",
            "More than 3 days",
            "More than 1 week"
        ]
    )

    medication = st.text_input("Has the patient taken any medication? If yes, specify.")
    extra_notes = st.text_area("Describe any other complaint:")

    if st.button("Analyze Symptoms"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
            return

        disease_results = calculate_condition_probabilities(selected_symptoms)

        st.subheader("Initial Possible Condition Ranking")
        st.dataframe(
            disease_results[
                [
                    "Possible Condition",
                    "Match Score (%)",
                    "Matched Symptoms",
                    "Red Flags"
                ]
            ],
            use_container_width=True
        )

        top_conditions = disease_results.head(3)["Possible Condition"].tolist()

        st.session_state["top_conditions"] = top_conditions
        st.session_state["selected_symptoms"] = selected_symptoms
        st.session_state["duration"] = duration
        st.session_state["medication"] = medication
        st.session_state["extra_notes"] = extra_notes

    if "top_conditions" in st.session_state:
        st.subheader("Dynamic Follow-Up Questions")

        questions = get_dynamic_followup_questions(
            st.session_state["top_conditions"]
        )

        followup_yes_symptoms = []

        with st.form("dynamic_followup_form"):
            for question in questions:
                answer = st.selectbox(
                    question,
                    ["No", "Yes"],
                    key=f"followup_{question}"
                )

                if answer == "Yes":
                    followup_yes_symptoms.extend(
                        map_followup_to_symptoms(question)
                    )

            submit_refine = st.form_submit_button("Refine Possible Conditions")

        if submit_refine:
            refined_symptoms = list(set(
                st.session_state["selected_symptoms"] +
                followup_yes_symptoms
            ))

            refined_results = calculate_condition_probabilities(refined_symptoms)
            best_match = refined_results.iloc[0]

            st.subheader("Refined Possible Condition Ranking")
            st.dataframe(
                refined_results[
                    [
                        "Possible Condition",
                        "Match Score (%)",
                        "Matched Symptoms",
                        "Red Flags"
                    ]
                ],
                use_container_width=True
            )

            st.subheader("AI Triage Screening Summary")

            st.warning(
                f"Most likely screening match: {best_match['Possible Condition']} "
                f"({best_match['Match Score (%)']}% symptom match)"
            )

            st.write(f"Duration: {st.session_state['duration']}")

            if st.session_state["medication"]:
                st.write(f"Medication reported: {st.session_state['medication']}")
            else:
                st.write("No medication was reported.")

            if st.session_state["extra_notes"]:
                st.write(f"Additional complaint: {st.session_state['extra_notes']}")

            if best_match["Red Flags"] != "None":
                st.error(
                    "Red-flag symptoms were detected. Medical review is recommended urgently."
                )
            elif best_match["Match Score (%)"] >= 60:
                st.warning(
                    "The symptom pattern is significant. Medical review is recommended for proper diagnosis."
                )
            else:
                st.info(
                    "Symptoms appear mild to moderate based on screening. Continue monitoring and seek review if symptoms worsen."
                )

            st.caption(
                "This is a screening and triage-support result only. It does not confirm a medical diagnosis."
            )


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

    save_to_google_sheets({
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
    })

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

            save_to_google_sheets({
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
            })

            current_record_available = True
            st.success("✅ Manual record processed and saved.")


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

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(f"Student ID: {student_id}")

    with c2:
        st.info(f"Name: {name}")

    with c3:
        st.warning(f"Severity: {severity}")

    st.subheader("🩺 Vital Signs")

    v1, v2, v3, v4 = st.columns(4)

    with v1:
        st.metric("🌡 Temperature", f"{temperature} °C")

    with v2:
        st.metric("❤️ Heart Rate", f"{heart_rate} BPM")

    with v3:
        st.metric("🫁 SpO₂", f"{spo2}%")

    with v4:
        st.metric("⚖️ BMI", f"{bmi}")

    b1, b2, b3 = st.columns(3)

    with b1:
        st.metric("🩸 Systolic BP", f"{systolic_bp}")

    with b2:
        st.metric("🩸 Diastolic BP", f"{diastolic_bp}")

    with b3:
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

    clinical_interview_assistant(severity, alert)

    st.subheader("💬 Medical Explanation Assistant")

    user_question = st.text_input(
        "Ask about symptoms or the AI result",
        placeholder="Example: Why do I have sore throat?"
    )

    if user_question:
        st.info(
            medical_explanation_assistant(
                user_question,
                risk_score,
                severity
            )
        )

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

    st.bar_chart(feature_importance.set_index("Feature")["Importance (%)"])

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
