import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

# Add D drive packages to path
sys.path.insert(0, r'D:\CDPrediction\packages')

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import re
import json
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from fpdf import FPDF

from code.meal_planner import get_personalized_meal_plan
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array
from config.firebase_config import auth

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Verify required directories exist
if not MODELS_DIR.exists():
    st.error("❌ Models directory not found at " + str(MODELS_DIR))
if not DATA_DIR.exists():
    st.error("❌ Data directory not found at " + str(DATA_DIR))

# Load Diabetes Prediction Model
with open(MODELS_DIR / "model_diabetes.sav", "rb") as f:
    model_diabetes = pickle.load(f)
with open(MODELS_DIR / "heart_disease_model.sav", "rb") as f:
    heart_disease = pickle.load(f)
kidney_disease = joblib.load(MODELS_DIR / "kidney_disease_model.pkl")
liver_model = joblib.load(MODELS_DIR / "liver_model.sav")
hypertension_model = joblib.load(MODELS_DIR / "hypertension_model.pkl")

category_map = {
    'red_blood_cells': {'normal': 0, 'abnormal': 1},
    'pus_cell': {'normal': 0, 'abnormal': 1},
    'pus_cell_clumps': {'notpresent': 0, 'present': 1},
    'bacteria': {'notpresent': 0, 'present': 1},
    'hypertension': {'no': 0, 'yes': 1},
    'diabetes_mellitus': {'no': 0, 'yes': 1},
    'coronary_artery_disease': {'no': 0, 'yes': 1},
    'appetite': {'poor': 0, 'good': 1},
    'pedal_edema': {'no': 0, 'yes': 1},
    'anemia': {'no': 0, 'yes': 1}
}



# Define common illness patterns with their symptoms and typical durations
COMMON_ILLNESS_PATTERNS = {
    'Common Cold': {
        'symptoms': ['Cough', 'Fever', 'Fatigue', 'Headache'],
        'min_symptoms': 2,  # Minimum symptoms needed to suggest this pattern
        'typical_duration': (3, 10),  # Days (min, max)
        'severity_pattern': {'Mild': 2, 'Moderate': 1, 'Severe': 0},  # Expected severity distribution
        'description': 'Usually starts with sore throat, followed by nasal symptoms and cough. Typically improves within 7-10 days.',
        'recommendations': [
            'Rest and stay hydrated',
            'Over-the-counter cold medications may help relieve symptoms',
            'Humidifier can ease congestion and sore throat'
        ]
    },
    'Seasonal Allergies': {
        'symptoms': ['Cough', 'Fatigue', 'Shortness of Breath'],
        'min_symptoms': 2,
        'typical_duration': (7, 60),
        'severity_pattern': {'Mild': 3, 'Moderate': 1, 'Severe': 0},
        'description': 'Symptoms often include itchy eyes, runny nose, and sneezing. May worsen during specific seasons.',
        'recommendations': [
            'Avoid known allergens when possible',
            'Consider over-the-counter antihistamines',
            'Keep windows closed during high pollen seasons'
        ]
    },
    'Influenza': {
        'symptoms': ['Fever', 'Fatigue', 'Headache', 'Chest Pain', 'Shortness of Breath'],
        'min_symptoms': 3,
        'typical_duration': (5, 14),
        'severity_pattern': {'Mild': 0, 'Moderate': 2, 'Severe': 2},
        'description': 'Usually begins suddenly with fever, muscle aches, and exhaustion. More severe than common cold.',
        'recommendations': [
            'Rest and stay hydrated',
            'Consult a doctor, especially if symptoms are severe',
            'Consider antiviral medications if diagnosed early',
            'Avoid contact with others to prevent spread'
        ]
    },
    'Gastroenteritis': {
        'symptoms': ['Nausea', 'Fatigue'],
        'min_symptoms': 2,
        'typical_duration': (1, 5),
        'severity_pattern': {'Mild': 1, 'Moderate': 2, 'Severe': 1},
        'description': 'Often includes stomach cramps, vomiting, and diarrhea. Usually resolves within a few days.',
        'recommendations': [
            'Stay hydrated with small, frequent sips of water',
            'Gradually reintroduce bland foods as symptoms improve',
            'Seek medical attention if unable to keep fluids down or signs of dehydration'
        ]
    },
    'COVID-19': {
        'symptoms': ['Fever', 'Cough', 'Fatigue', 'Shortness of Breath', 'Headache', 'Nausea'],
        'min_symptoms': 3,
        'typical_duration': (7, 21),
        'severity_pattern': {'Mild': 1, 'Moderate': 2, 'Severe': 1},
        'description': 'Symptoms vary widely, may include loss of taste/smell. Can range from mild to severe.',
        'recommendations': [
            'Isolate to prevent spread to others',
            'Consider getting tested for COVID-19',
            'Monitor oxygen levels if possible',
            'Seek immediate medical attention for severe symptoms'
        ]
    },
    'Migraine': {
        'symptoms': ['Headache', 'Nausea'],
        'min_symptoms': 2,
        'typical_duration': (0.5, 3),
        'severity_pattern': {'Mild': 0, 'Moderate': 1, 'Severe': 2},
        'description': 'Typically includes throbbing headache, often on one side, sometimes with light/sound sensitivity.',
        'recommendations': [
            'Rest in a dark, quiet room',
            'Apply cold compresses to the forehead',
            'Consider over-the-counter pain relievers',
            'Track triggers to prevent future attacks'
        ]
    }
}

strength_label = ["Very Weak", "Weak", "Moderate", "Strong", "Very Strong"]
bar_color = ["#FF4B4B", "#FF884B", "#FFD93D", "#2ECC71", "#27AE60"]

# Function to create the input datafram
def create_input_df(user_inputs, category_map):
    # Transform categorical inputs using category_map
    for category in category_map:
        if category in user_inputs:
            user_inputs[category] = category_map[category].get(user_inputs[category], -1)  # -1 or other value for missing/unknown categories
    input_df = pd.DataFrame([user_inputs])
    return input_df

language = st.sidebar.selectbox("🌐 Choose Language", ["English", "Tamil"])
st.session_state["language"] = language

with open(BASE_DIR / "translation.json", "r", encoding="utf-8") as f:
    translations = json.load(f)

def t(key):
    lang = st.session_state.get("language", "English")
    return translations.get(key, {}).get(lang, key)

def suggest_email_correction(email):
    common_domains = {
        "gmial.com": "gmail.com",
        "gnail.com": "gmail.com",
        "gamil.com": "gmail.com",
        "gmaill.com": "gmail.com",
        "yaho.com": "yahoo.com",
        "yahho.com": "yahoo.com",
        "hotnail.com": "hotmail.com",
        "hotmai.com": "hotmail.com",
        "outlok.com": "outlook.com"
    }

    if "@" in email:
        local, domain = email.split("@")
        corrected_domain = common_domains.get(domain.lower())
        if corrected_domain:
            return f"{local}@{corrected_domain}"
    return None

def check_password_strength(password):
    score = 0

    # Rules
    if len(password) >= 8:
        score += 1
    if re.search(r"[A-Z]", password):
        score += 1
    if re.search(r"[a-z]", password):
        score += 1
    if re.search(r"\d", password):
        score += 1
    if re.search(r"[\W_]", password):  # symbols
        score += 1

    return score

def calculate_risk_score():
    score = 0

    # Blood Pressure
    if st.session_state.get("blood_pressure", 0) < 90 or st.session_state.get("blood_pressure", 0) > 140:
        score += 20

    # Heart Rate
    if st.session_state.get("heart_rate", 0) < 60 or st.session_state.get("heart_rate", 0) > 100:
        score += 20

    # Blood Sugar
    if st.session_state.get("blood_sugar", 0) < 70 or st.session_state.get("blood_sugar", 0) > 140:
        score += 20

    # Diabetes prediction
    if st.session_state.get("diabetes_diagnosis", "").lower() == "the patient has diabetes":
        score += 20

    # Heart disease prediction
    if st.session_state.get("heart_diagnosis", "").lower() == "positive":
        score += 20

    # Kidney disease prediction
    if st.session_state.get("kidney_diagnosis", "").lower().startswith("the patient is likely to have"):
        score += 20

    # Fever
    if st.session_state.get("temperature", 0) >= 38.0:
        score += 20

    # BMI
    bmi = st.session_state.get("bmi")
    if bmi:
        if bmi < 18.5:
            score += 10
        elif 25 <= bmi < 30:
            score += 10
        elif bmi >= 30:
            score += 20

    return min(score, 100)  # Max 100

def clean_text_for_pdf(text):
    return text.encode('latin-1', 'ignore').decode('latin-1')

def classify_blood_pressure(systolic, diastolic):
    if systolic > 180 or diastolic > 120:
        return "Hypertension Crisis", "🔴", "Seek emergency care immediately."
    elif systolic >= 140 or diastolic >= 90:
        return "Stage 2 Hypertension", "🟥", "Consult your doctor. Lifestyle changes and medication likely needed."
    elif 130 <= systolic <= 139 or 80 <= diastolic <= 89:
        return "Stage 1 Hypertension", "🟧", "Monitor regularly. Improve diet, reduce stress, consult doctor."
    elif 120 <= systolic <= 129 and diastolic < 80:
        return "Elevated", "🟨", "Be cautious. Reduce salt, exercise regularly."
    elif systolic < 120 and diastolic < 80:
        return "Healthy", "🟩", "Keep up the healthy habits!"
    else:
        return "Unclassified", "⚠️", "Double-check readings or consult provider."

# 🚑 Check if patient needs appointment
def needs_appointment():
    if (st.session_state.get("blood_pressure", 0) < 90 or st.session_state.get("blood_pressure", 0) > 140):
        return True
    if (st.session_state.get("heart_rate", 0) < 60 or st.session_state.get("heart_rate", 0) > 100):
        return True
    if (st.session_state.get("blood_sugar", 0) < 70 or st.session_state.get("blood_sugar", 0) > 140):
        return True
    if (st.session_state.get("temperature", 0) > 38.0):
        return True
    if st.session_state.get("diabetes_diagnosis", "").lower() == "the patient has diabetes":
        return True
    if st.session_state.get("heart_diagnosis", "").lower() == "positive":
        return True
    if st.session_state.get("kidney_diagnosis", "").lower().startswith("the patient is likely to have"):
        return True
    return False

def analyze_symptom_patterns(symptom_log):
    """
    Analyze symptom log for patterns that match common illnesses
    Returns a list of possible conditions with confidence levels
    """
    if symptom_log.empty:
        return []
    
    # Convert dates from string to datetime if needed
    if isinstance(symptom_log['Date'].iloc[0], str):
        symptom_log['Date'] = pd.to_datetime(symptom_log['Date'])
    
    # Sort by date
    symptom_log = symptom_log.sort_values('Date')
    
    # Get unique dates to calculate duration
    unique_dates = symptom_log['Date'].unique()
    duration_days = (max(unique_dates) - min(unique_dates)).days + 1
    
    # Get list of all symptoms and their severities
    all_symptoms = symptom_log['Symptom'].tolist()
    symptom_count = Counter(all_symptoms)
    
    # Count severity levels
    severity_counts = Counter(symptom_log['Severity'].tolist())
    
    # Calculate the frequency of symptoms over time for progression analysis
    symptom_progression = {}
    for symptom in set(all_symptoms):
        symptom_dates = symptom_log[symptom_log['Symptom'] == symptom]['Date'].tolist()
        symptom_progression[symptom] = symptom_dates
    
    # Analyze matches with known patterns
    matches = []
    
    for illness, pattern in COMMON_ILLNESS_PATTERNS.items():
        # Calculate how many symptoms match
        matching_symptoms = [s for s in pattern['symptoms'] if s in symptom_count]
        symptom_match_ratio = len(matching_symptoms) / len(pattern['symptoms'])
        
        # Check if minimum number of symptoms are present
        if len(matching_symptoms) < pattern['min_symptoms']:
            continue
            
        # Check duration match (if we have enough data points)
        if len(unique_dates) > 1:
            min_duration, max_duration = pattern['typical_duration']
            duration_match = min_duration <= duration_days <= max_duration
            # Calculate how well the duration matches
            if duration_days < min_duration:
                duration_score = duration_days / min_duration
            elif duration_days > max_duration:
                duration_score = max_duration / duration_days
            else:
                duration_score = 1.0
        else:
            # Not enough data points to determine duration
            duration_score = 0.5  # Neutral score
            duration_match = None
            
        # Check severity pattern match
        severity_match_score = 0
        expected_total = sum(pattern['severity_pattern'].values())
        if expected_total > 0:
            for severity, expected_count in pattern['severity_pattern'].items():
                actual_count = severity_counts.get(severity, 0)
                # Calculate proportion of expected vs actual
                if expected_count > 0:
                    severity_match_score += min(actual_count / expected_count, 1.0) * (expected_count / expected_total)
        else:
            severity_match_score = 0.5  # Neutral score
            
        # Calculate overall confidence score (weighted average)
        confidence = (
            symptom_match_ratio * 0.60 +  # Symptom matching is most important
            duration_score * 0.25 +        # Duration is somewhat important
            severity_match_score * 0.15    # Severity pattern is least important
        ) * 100  # Convert to percentage
            
        if confidence >= 40:  # Only include somewhat confident matches
            matches.append({
                'illness': illness,
                'confidence': confidence,
                'matching_symptoms': matching_symptoms,
                'missing_symptoms': [s for s in pattern['symptoms'] if s not in symptom_count],
                'duration_match': duration_match,
                'description': pattern['description'],
                'recommendations': pattern['recommendations']
            })
    
    # Sort by confidence (highest first)
    matches.sort(key=lambda x: x['confidence'], reverse=True)
    return matches

def get_symptom_progression_chart(symptom_log):
    """Generate a chart showing symptom progression over time"""
    if symptom_log.empty:
        return None
        
    # Prepare data for visualization
    # Convert to long format suitable for Altair
    chart_data = []
    
    for _, row in symptom_log.iterrows():
        severity_value = {'Mild': 1, 'Moderate': 2, 'Severe': 3}.get(row['Severity'], 0)
        chart_data.append({
            'Date': row['Date'],
            'Symptom': row['Symptom'],
            'Severity': row['Severity'],
            'SeverityValue': severity_value
        })
    
    chart_df = pd.DataFrame(chart_data)
    
    # Create Altair chart
    if not chart_df.empty:
        chart = alt.Chart(chart_df).mark_circle(size=100).encode(
            x='Date:T',
            y='Symptom:N',
            color='Severity:N',
            size='SeverityValue:Q',
            tooltip=['Date', 'Symptom', 'Severity']
        ).properties(
            width=600,
            height=300,
            title='Symptom Progression Over Time'
        )
        return chart
    return None

def analyze_and_display_patterns(symptom_log):
    """Analyze symptom patterns and display results in the Streamlit app"""
    st.subheader("🔍 Pattern Recognition Analysis")
    
    if symptom_log.empty:
        st.info("No symptoms have been logged yet. Please log your symptoms to see pattern analysis.")
        return
        
    # Add a chart showing symptom progression
    chart = get_symptom_progression_chart(symptom_log)
    if chart:
        st.altair_chart(chart, use_container_width=True)
    
    # Get pattern matches
    matches = analyze_symptom_patterns(symptom_log)
    
    if not matches:
        st.info("No clear illness patterns detected from the logged symptoms. Continue tracking for better analysis.")
        return
        
    # Display potential illness matches
    st.subheader("🔬 Potential Illness Patterns Detected")
    st.markdown("*Note: This is not a medical diagnosis. Please consult a healthcare professional.*")
    
    for match in matches:
        confidence_color = "#5cb85c" if match['confidence'] >= 70 else "#f0ad4e" if match['confidence'] >= 50 else "#d9534f"
        
        with st.expander(f"{match['illness']} - Confidence: {match['confidence']:.1f}%"):
            st.markdown(f"""
            <div style="padding: 10px; border-left: 5px solid {confidence_color};">
                <p><strong>Description:</strong> {match['description']}</p>
                <p><strong>Matching symptoms:</strong> {', '.join(match['matching_symptoms'])}</p>
                <p><strong>Missing symptoms to watch for:</strong> {', '.join(match['missing_symptoms']) if match['missing_symptoms'] else 'None'}</p>
                <p><strong>Typical duration:</strong> {COMMON_ILLNESS_PATTERNS[match['illness']]['typical_duration'][0]}-{COMMON_ILLNESS_PATTERNS[match['illness']]['typical_duration'][1]} days</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Recommendations:")
            for i, recommendation in enumerate(match['recommendations'], 1):
                st.markdown(f"{i}. {recommendation}")
            
            # Add action button for top matches
            if match['confidence'] >= 60:
                if st.button(f"Schedule Appointment (Based on {match['illness']} Pattern)", key=f"appt_{match['illness']}"):
                    st.session_state['appointment_reason'] = f"Possible {match['illness']} - Symptoms: {', '.join(match['matching_symptoms'])}"
                    st.success("✅ Appointment reason saved. Please go to the Home page to complete scheduling.")

def suggest_next_health_actions(symptom_log, matches):
    """Suggest next health actions based on symptom analysis"""
    st.subheader("📋 Suggested Next Steps")
    
    if not symptom_log.empty and matches:
        top_match = matches[0]
        
        # Determine urgency level
        urgent_symptoms = ['Chest Pain', 'Shortness of Breath', 'Severe Headache']
        has_urgent_symptoms = any(symptom in urgent_symptoms for symptom in symptom_log['Symptom'].unique())
        severe_symptoms = symptom_log[symptom_log['Severity'] == 'Severe']['Symptom'].tolist()
        
        if has_urgent_symptoms or (severe_symptoms and top_match['confidence'] > 60):
            st.error("⚠️ **Urgent Attention Recommended**: Based on your symptoms, we recommend seeking medical attention promptly.")
            st.markdown("### Why this is urgent:")
            if has_urgent_symptoms:
                urgent_found = [s for s in urgent_symptoms if s in symptom_log['Symptom'].unique()]
                st.markdown(f"- You reported {', '.join(urgent_found)}, which may require immediate evaluation")
            if severe_symptoms:
                st.markdown(f"- You rated these symptoms as severe: {', '.join(severe_symptoms)}")
                
        elif top_match['confidence'] > 70:
            st.warning(f"**Recommended Action**: Based on the pattern matching {top_match['illness']}, consider the following:")
            for rec in top_match['recommendations']:
                st.markdown(f"- {rec}")
        else:
            st.info("**Recommended Action**: Continue monitoring your symptoms and update the tracker daily.")
            
        # Suggest symptom tracking focus
        if top_match['missing_symptoms']:
            st.markdown("### Symptoms to watch for:")
            for symptom in top_match['missing_symptoms']:
                st.markdown(f"- {symptom}")
    else:
        st.info("Continue logging your symptoms daily for better pattern recognition.")

# Page Title
st.title(t("\U0001F3E5 Patient Health Dashboard"))
st.write(t("Monitor and log vital health data for better tracking and care."))

# Navigation Menu
#menu = st.sidebar.selectbox("Navigation", ["Login/Register","Home", "Disease Prediction","Hypertension", "Diabetes", "Heart Disease Prediction", "Kidney Disease Prediction","Fever","Symptom Tracker"])
# Sidebar Menu
if "logged_in_user" not in st.session_state:
    # If not logged in: Only show Login/Register
    menu = st.sidebar.selectbox("Navigation", ["Login/Register"])
else:
    # If logged in: Show full Dashboard
    menu = st.sidebar.selectbox("Navigation", [
        "Home", "Disease Prediction", "Hypertension", "Diabetes",
        "Heart Disease Prediction", "Kidney Disease Prediction","Liver Disease Prediction", "Fever", "Symptom Tracker"
    ])

# Logout Button
if "logged_in_user" in st.session_state:
    if st.sidebar.button("Logout"):
        del st.session_state["logged_in_user"]
        st.success("Logged out successfully!")
        st.rerun()

# Handle Login/Register Page
if menu == "Login/Register":
    st.header(t("🔒 Login / 📝 Register"))
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        st.subheader(t("🔐 Login"))
        email = st.text_input(t("Email"), key="login_email")
        password = st.text_input(t("Password"), type="password", key="login_password")
        correction = suggest_email_correction(email)
        if correction:
            st.warning(f"⚠️ Did you mean `{correction}`?")

        if st.button(t("Login")):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state["logged_in_user"] = email
                st.success(f"✅ Welcome, {email}!")
                st.rerun()
            except Exception as e:
                st.error(t("❌ Login failed. Please check your credentials."))

    with register_tab:
        st.subheader(t("📝 Register"))
        email = st.text_input(t("New Email"), key="register_email")
        password = st.text_input(t("New Password"), type="password", key="register_password")
        confirm_password = st.text_input(t("Confirm Password"), type="password", key="confirm_password")
        correction = suggest_email_correction(email)
        if correction:
            st.warning(f"⚠️ Did you mean `{correction}`?")
        strength = check_password_strength(password)
        if password:
            # Cap strength at 4 to match the list indices (0-4)
            strength_idx = min(strength, 4)
            st.markdown(f"""
            <div style='margin-top:10px'>
                <div style='width:100%;background:#ddd;height:15px;border-radius:5px'>
                    <div style='width:{(strength/5)*100}%;height:15px;border-radius:5px;background:{bar_color[strength_idx]}'></div>
                </div>
                <small style='color:{bar_color[strength_idx]}'><b>{strength_label[strength_idx]}</b></small>
            </div>
            """, unsafe_allow_html=True)

        if st.button(t("Register")):
            if password != confirm_password:
                st.warning(t("❌ Passwords do not match."))
            elif strength < 3:
                st.warning(t("❌ Password too weak."))
            elif not email:
                st.warning(t("❌ Email is required."))
            else:
                try:
                    auth.create_user_with_email_and_password(email, password)
                    st.success(t("✅ Account created successfully! You can now log in."))
                except Exception as e:
                    error_text = str(e)
                    if "EMAIL_EXISTS" in error_text:
                        st.warning("⚠️ This email is already registered. Please log in or use a different email.")
                    else:
                        st.error("❌ Registration failed. Please check your details and try again.")
                
# Home Page - Input Fields for Health Data, Data Log, Trends, and Report Download
elif menu == "Home":
    # Patient Information
    st.header(t("Patient Details"))
    patient_name = st.text_input(t("Patient Name"))
    patient_age = st.number_input(t("Age"), min_value=1, max_value=120, step=1)
    patient_gender = st.selectbox(t("Gender"), ["Male", "Female", "Other"])
    st.header(t("Enter Your Health Data"))
    st.session_state["blood_pressure"] = st.number_input(t("Blood Pressure (mmHg)"), min_value=50, max_value=200, step=1)
    st.session_state["heart_rate"] = st.number_input(t("Heart Rate (bpm)"), min_value=30, max_value=200, step=1)
    st.session_state["blood_sugar"] = st.number_input(t("Blood Sugar Level (mg/dL)"), min_value=50, max_value=500, step=1)
    st.session_state["temperature"] = st.number_input(t("Body Temperature (°C)"), min_value=35.0, max_value=42.0, step=0.1)
    st.subheader(t("💪 BMI Calculator"))
    height_cm = st.number_input(t("Height (in cm)"), min_value=100, max_value=250, step=1)
    weight_kg = st.number_input(t("Weight (in kg)"), min_value=20, max_value=250, step=1)

    # BMI Calculation
    if height_cm > 0:
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        st.metric(t("🧮 Your BMI"), f"{bmi:.2f}")

        # BMI Classification
        if bmi < 18.5:
            st.warning(t("🔹 Underweight"))
        elif 18.5 <= bmi < 25:
            st.success(t("✅ Normal weight"))
        elif 25 <= bmi < 30:
            st.warning(t("🟠 Overweight"))
        else:
            st.error(t("🔴 Obese"))
        st.session_state["bmi"] = round(bmi, 2)


    if "health_data" not in st.session_state:
        st.session_state["health_data"] = pd.DataFrame(columns=["Blood Pressure", "Heart Rate", "Blood Sugar", "Temperature"])

    if st.button(t("Save Data")):
        new_data = pd.DataFrame([[st.session_state["blood_pressure"], st.session_state["heart_rate"], 
                                st.session_state["blood_sugar"], st.session_state["temperature"], st.session_state.get("bmi", None)]], 
                                columns=["Blood Pressure", "Heart Rate", "Blood Sugar", "Temperature", "BMI"])
        st.session_state["health_data"] = pd.concat([st.session_state["health_data"], new_data], ignore_index=True)
        st.success(t("✅ Data saved successfully!"))
        st.session_state["show_risk"] = True
        st.rerun()

    st.subheader(t("📊 Health Data Log"))
    st.dataframe(st.session_state["health_data"])
    
    st.subheader(t("📈 Health Trends"))
    if not st.session_state["health_data"].empty:
        health_data_numeric = st.session_state["health_data"].astype(float)
        fig = px.line(health_data_numeric, markers=True)
        st.plotly_chart(fig)
    if st.session_state.get("show_risk"):
        st.header(t("📋 Health Risk Overview"))

        with st.expander(t("Explain Risk Factors (Click to View Details)")):
            risk_table = """
            | **Metric** | **Normal Range** | **Points if Abnormal** |
            |:---|:---|:---|
            | Blood Pressure | 90–140 mmHg | +20 |
            | Heart Rate | 60–100 bpm | +20 |
            | Blood Sugar | 70–140 mg/dL | +20 |
            | BMI | 18.5–24.9 kg/m² | +10–20 |
            | Diabetes Prediction | Positive | +20 |
            | Heart Disease Prediction | Positive | +20 |
            | Kidney Disease Prediction | Positive | +20 |
            | High Fever (Temperature > 38°C) | Detected | +20 |
            """
            st.markdown(risk_table)


        st.header(t("📋 Health Risk Summary"))

        risk_score = calculate_risk_score()

        # Color + Message
        if risk_score <= 30:
            risk_color = "green"
            risk_message = t("Low Risk ✅ Maintain healthy habits!")
            suggestion = [
               t("Continue a balanced diet"),
                t("Stay physically active"),
                t("Get routine checkups")
            ]
        elif risk_score <= 70:
            risk_color = "orange"
            risk_message = t("Moderate Risk ⚠️ Watch your health closely!")
            suggestion = [
                t("Consult a nutritionist"),
                t("Monitor BP, Sugar daily"),
                t("Mild exercise recommended")
            ]
        else:
            risk_color = "red"
            risk_message = t("High Risk ❗ Immediate medical attention advised!")
            suggestion = [
                t("Consult your doctor urgently"),
                t("Strict diet and medication"),
                t("Avoid stress and monitor vitals daily")
            ]

        st.markdown(f"""
        <div style="background-color:{risk_color};padding:10px;border-radius:10px">
        <h2 style="color:white;">Risk Score: {risk_score}/100</h2>
        <h4 style="color:white;">{risk_message}</h4>
        </div>
        """, unsafe_allow_html=True)
        # Show Suggestions
        st.subheader(t("💡 Personalized Suggestions:"))
        for i, tip in enumerate(suggestion, 1):
            st.markdown(f"- {tip}")

        # # 📋 Appointment Request Form
        if risk_score >30:
            st.header(t("🚑 Appointment Request (Recommended)"))

            appointment_name = st.text_input(t("Patient Name"), st.session_state.get("patient_name", ""),key="appointment_patient_name")
            contact_number = st.text_input(t("Contact Number"))
            preferred_date = st.date_input(t("Preferred Appointment Date"))
            preferred_time = st.time_input(t("Preferred Time"))
            symptoms_summary = st.text_area(t("Briefly describe your symptoms:"))

            if st.button(t("Submit Appointment Request")):
                appointment_data = pd.DataFrame([{
                    "Patient Name": appointment_name,
                    "Contact Number": contact_number,
                    "Preferred Date": preferred_date,
                    "Preferred Time": preferred_time,
                    "Symptoms Summary": symptoms_summary,
                    "Request Timestamp": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                }])

                # Save appointment request
                appointments_file = f"appointments_{appointment_name.replace(' ', '_').lower()}.csv"
                if os.path.exists(appointments_file):
                    existing = pd.read_csv(appointments_file)
                    appointment_data = pd.concat([existing, appointment_data], ignore_index=True)

                appointment_data.to_csv(appointments_file, index=False)
                st.success(t("✅ Your appointment request has been submitted!"))
            
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "Patient Health Report", ln=True, align="C")
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%d-%m-%Y %H:%M:%S")
        report_id = current_datetime.strftime("REP-%Y%m%d-%H%M%S")
        pdf.set_font("Arial", "", 12)
        pdf.cell(200, 10, f"Report ID: {report_id}", ln=True)
        pdf.cell(200, 10, f"Generated on: {formatted_datetime}", ln=True)
        pdf.ln(5)
        pdf.ln(10)
        pdf.set_font("Arial", "", 12)
        pdf.cell(200, 10, f"Patient Name: {patient_name}", ln=True)
        pdf.cell(200, 10, f"Age: {patient_age}", ln=True)
        pdf.cell(200, 10, f"Gender: {patient_gender}", ln=True)
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Health Data:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(200, 10, f"Blood Pressure: {st.session_state['blood_pressure']} mmHg", ln=True)
        pdf.cell(200, 10, f"Heart Rate: {st.session_state['heart_rate']} bpm", ln=True)
        pdf.cell(200, 10, f"Blood Sugar: {st.session_state['blood_sugar']} mg/dL", ln=True)
        pdf.cell(200, 10, f"Body Temperature: {st.session_state['temperature']} °C", ln=True)
        if "bmi" in st.session_state:
            pdf.cell(200, 10, f"BMI: {st.session_state['bmi']} kg/m²", ln=True)
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Prediction Results:", ln=True)
        pdf.set_font("Arial", "", 12)



        # Include Symptoms if available
        if "symptoms_selected" in st.session_state:
            symptoms = st.session_state["symptoms_selected"]
            if symptoms:
                pdf.set_font("Arial", "B", 14)
                pdf.cell(200, 10, "Symptoms:", ln=True)
                pdf.set_font("Arial", "", 12)
                for symptom in symptoms:
                    pdf.cell(200, 10, f"- {symptom}", ln=True)
                pdf.ln(5)
        
        # General Disease Prediction (XGBoost based on symptoms)
        if "general_disease_name" in st.session_state:
            disease_name = st.session_state["general_disease_name"]
            disease_prob = st.session_state.get("general_disease_probability", "N/A")
            pdf.cell(200, 10, f"General Disease Prediction (Symptom-Based): {disease_name} ({disease_prob})", ln=True)
        
        # General Disease Description
        if "disease_description" in st.session_state:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, "Disease Description:", ln=True)
            pdf.set_font("Arial", "", 12)
            description_lines = st.session_state["disease_description"].split('\n')
            for line in description_lines:
                pdf.multi_cell(0, 10, line)
            pdf.ln(5)

        # General Disease Precautions
        if "disease_precautions" in st.session_state:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, "Precautions:", ln=True)
            pdf.set_font("Arial", "", 12)
            precautions = st.session_state["disease_precautions"]
            for i, item in enumerate(precautions, 1):
                pdf.cell(200, 10, f"{i}. {item}", ln=True)
            pdf.ln(5)

        # Include Diabetes Result
        if "diabetes_diagnosis" in st.session_state:
            pdf.cell(200, 10, f"Diabetes Prediction: {st.session_state['diabetes_diagnosis']}", ln=True)
        

        # Include Heart Result
        if "heart_diagnosis" in st.session_state:
            pdf.cell(200, 10, f"Heart Disease Prediction: {st.session_state['heart_diagnosis']}", ln=True)
        # Include Kidney Result
        if "kidney_diagnosis" in st.session_state:
            pdf.cell(200, 10, f"Kidney Disease Prediction: {st.session_state['kidney_diagnosis']}", ln=True)
        # Include Liver Result
        if "liver_diagnosis" in st.session_state:
            pdf.cell(200, 10, f"Liver Disease Prediction: {st.session_state['liver_diagnosis']}", ln=True)

        # Hypertension and Fever
        hypertension_status = "Normal" if 90 <= st.session_state['blood_pressure'] <= 140 else "Hypertension Detected"
        pdf.cell(200, 10, f"Hypertension Analysis: {hypertension_status}", ln=True)
        fever_status = "Normal" if st.session_state['temperature'] < 38.0 else "Fever Detected"
        pdf.cell(200, 10, f"Fever Detection: {fever_status}", ln=True)

            # 🎯 New Meal Plan Section
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Personalized Meal Recommendation:", ln=True)
        pdf.set_font("Arial", "", 12)

        # Generate meal plan dynamically
        meal_plan = get_personalized_meal_plan(
            glucose=st.session_state['blood_sugar'],
            bmi=24  # ✅ You can dynamically pass patient's BMI if available
        )

        for meal_time, meals in meal_plan.items():
            pdf.cell(200, 10, f"{meal_time}:", ln=True)
            for meal in meals:
                meal_line = f" - {meal['name']} | {meal['Calories']} kcal, {meal['Fibre']}g Fiber, {meal['Sugars']}g Sugar"
                pdf.cell(200, 10, meal_line, ln=True)
            pdf.ln(5)
        
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Risk Assessment:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(200, 10, f"Risk Score: {risk_score}/100", ln=True)
        pdf.cell(200, 10, f"Risk Level: {clean_text_for_pdf(risk_message)}", ln=True)
        pdf.ln(10)

        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Personalized Suggestions:", ln=True)
        pdf.set_font("Arial", "", 12)
        for tip in suggestion:
            pdf.cell(200, 10, f"- {clean_text_for_pdf(tip)}", ln=True)
        
        pdf.output("patient_report.pdf")
        return "patient_report.pdf"


    if st.button(t("Download Full Report")):
        pdf_file = generate_pdf()
        with open(pdf_file, "rb") as file:
            st.download_button(label=t("Download PDF Report"), data=file, file_name="Patient_Health_Report.pdf", mime="application/pdf")

# Disease Prediction Page
elif menu=="Disease Prediction":
    disease_model = DiseaseModel()
    disease_model.load_xgboost(str(MODELS_DIR / "xgboost_model.json"))

    # Title
    st.write(t('# Disease Prediction using Machine Learning'))

    symptoms = st.multiselect(t('What are your symptoms?'), options=disease_model.all_symptoms)
    st.session_state["symptoms_selected"] = symptoms


    X = prepare_symptoms_array(symptoms)

    # Trigger XGBoost model
    if st.button(t('Predict')): 
        # Run the model with the python script
        
        prediction, prob = disease_model.predict(X)
        st.markdown("## " + t("Disease: {prediction} with {probability}% probability").format(
            prediction=prediction, probability=f"{prob*100:.2f}"))
        st.session_state["general_disease_name"] = prediction
        st.session_state["general_disease_probability"] = f"{prob*100:.2f}%"
        st.session_state["disease_description"] = disease_model.describe_predicted_disease()
        st.session_state["disease_precautions"] = disease_model.predicted_disease_precautions()

        tab1, tab2= st.tabs([t("Description"), t("Precautions")])

        with tab1:
            st.write(disease_model.describe_predicted_disease())

        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f'{i+1}. {precautions[i]}')


# Diabetes Prediction Page
elif menu == "Diabetes":
    st.subheader("\U0001FA7A Diabetes Prediction Test")
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input('Enter the Pregnancies value')
    with col2:
        Glucose = st.number_input('Enter the Glucose value')
    with col1:
        BloodPressure = st.number_input('Enter the Blood Pressure value')
    with col2:
        SkinThickness = st.number_input('Enter the Skin Thickness value')
    with col1:
        Insulin = st.number_input('Enter the Insulin value')
    with col2:
        BMI = st.number_input('Enter the BMI value')
    with col1:
        DiabetesPedigreeFunction = st.number_input('Enter the Diabetes Pedigree Function value')
    with col2:
        Age = st.number_input('Enter the Age value')
    
    if Glucose == 0 or BMI == 0:
        st.warning("⚠️ Please enter valid Glucose and BMI values before prediction.")
    else:
        diabetes_diagnosis = ''
        if st.button('Diabetes Prediction Test'):
            diabetes_prediction = model_diabetes.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            diabetes_diagnosis = 'The patient has diabetes' if diabetes_prediction[0] == 1 else 'The patient does not have diabetes'
            st.session_state["diabetes_diagnosis"] = diabetes_diagnosis
            st.session_state['show_meal_plan'] = True 
            st.success(diabetes_diagnosis)
            if st.session_state.get('show_meal_plan', False):
                st.subheader("🍽 Personalized Meal Plan")
                meal_plan = get_personalized_meal_plan(glucose=Glucose, bmi=BMI)

                for meal_time, meals in meal_plan.items():
                    st.markdown(f"### {meal_time}")
                    for meal in meals:
                        st.markdown(f"- {meal['name']} | {meal['Calories']} kcal | {meal['Fibre']}g Fiber | {meal['Sugars']}g Sugar")
                    st.markdown("---")


# Heart Disease Prediction Analysis
elif menu=="Heart Disease Prediction":
    st.title('Heart Disease Prediction')
    age = st.slider('Age', 18, 100, 50)
    gender_options = ['Male', 'Female']
    gender = st.selectbox('Gender', gender_options)
    gender_num = 1 if gender == 'Male' else 0 
    cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
    cp = st.selectbox('Chest Pain Type', cp_options)
    cp_num = cp_options.index(cp)
    trestbps = st.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.slider('Cholesterol', 100, 600, 250)
    fbs_options = ['False', 'True']
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', fbs_options)
    fbs_num = fbs_options.index(fbs)
    restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
    restecg = st.selectbox('Resting Electrocardiographic Results', restecg_options)
    restecg_num = restecg_options.index(restecg)
    thalach = st.slider('Maximum Heart Rate Achieved', 70, 220, 150)
    exang_options = ['No', 'Yes']
    exang = st.selectbox('Exercise Induced Angina', exang_options)
    exang_num = exang_options.index(exang)
    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    slope_options = ['Upsloping', 'Flat', 'Downsloping']
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', slope_options)
    slope_num = slope_options.index(slope)
    ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 1)
    thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
    thal = st.selectbox('Thalassemia', thal_options)
    thal_num = thal_options.index(thal)

    with open(MODELS_DIR / "mean_std_values.pkl", "rb") as f:
        mean_std_values = pickle.load(f)


    if st.button('Predict'):
        user_input = pd.DataFrame(data={
            'age': [age],
            'sex': [gender_num],  
            'cp': [cp_num],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs_num],
            'restecg': [restecg_num],
            'thalach': [thalach],
            'exang': [exang_num],
            'oldpeak': [oldpeak],
            'slope': [slope_num],
            'ca': [ca],
            'thal': [thal_num]
        })
        # Apply saved transformation to new data
        user_input = (user_input - mean_std_values['mean']) / mean_std_values['std']
        prediction = heart_disease.predict(user_input)
        prediction_proba = heart_disease.predict_proba(user_input)

        if prediction[0] == 1:
            bg_color = 'red'
            prediction_result = 'Positive'
        else:
            bg_color = 'green'
            prediction_result = 'Negative'
        
        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

        st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction: {prediction_result}<br>Confidence: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)
# Kidney Disease Prediction Analysis
elif menu=="Kidney Disease Prediction":
    st.title('Kidney Disease Prediction')

    # categorical input
    red_blood_cells = st.radio('Red Blood Cells', ('normal', 'abnormal'))
    pus_cell = st.radio('Pus Cell', ('normal', 'abnormal'))
    pus_cell_clumps = st.radio('Pus Cell Clumps', ('notpresent', 'present'))
    bacteria = st.radio('Bacteria', ('notpresent', 'present'))
    hypertension = st.radio('Hypertension', ('no', 'yes'))
    diabetes_mellitus = st.radio('Diabetes Mellitus', ('no', 'yes'))
    coronary_artery_disease = st.radio('Coronary Artery Disease', ('no', 'yes'))
    appetite = st.radio('Appetite', ('poor', 'good'))
    pedal_edema = st.radio('Pedal Edema', ('no', 'yes'))
    anemia = st.radio('Anemia', ('no', 'yes'))

    # numerical input
    age = st.slider('Age', 0, 100, 0)
    blood_pressure = st.slider('Blood Pressure', 0, 180, 0)
    specific_gravity = st.slider('Specific Gravity', 0.0, 2.0, 0.0)
    albumin = st.slider('Albumin', 0, 5, 0)
    sugar = st.slider('Sugar', 0, 5, 0)
    blood_glucose_random = st.slider('Blood Glucose Random', 0, 500, 0)
    blood_urea = st.slider('Blood Urea', 0, 200, 0)
    serum_creatinine = st.slider('Serum Creatinine', 0.0, 10.0, 0.0)
    sodium = st.slider('Sodium', 0, 200, 0)
    potassium = st.slider('Potassium', 0, 10, 0)
    hemoglobin = st.slider('Hemoglobin', 0, 20, 0)
    packed_cell_volume = st.slider('Packed Cell Volume', 0, 100, 0)
    white_blood_cell_count = st.slider('White Blood Cell Count', 0, 20000, 0)
    red_blood_cell_count = st.slider('Red Blood Cell Count', 0, 10, 0)

    # User inputs
    user_inputs = {
        'age': age, 
        'blood_pressure': blood_pressure, 
        'specific_gravity': specific_gravity, 
        'albumin': albumin, 
        'sugar': sugar,
        'red_blood_cells': red_blood_cells, 
        'pus_cell': pus_cell, 
        'pus_cell_clumps': pus_cell_clumps, 
        'bacteria': bacteria,
        'blood_glucose_random': blood_glucose_random, 
        'blood_urea':blood_urea, 
        'serum_creatinine': serum_creatinine, 
        'sodium': sodium,
        'potassium': potassium, 
        'hemoglobin': hemoglobin, 
        'packed_cell_volume': packed_cell_volume,
        'white_blood_cell_count': white_blood_cell_count, 
        'red_blood_cell_count': red_blood_cell_count, 
        'hypertension': hypertension,
        'diabetes_mellitus': diabetes_mellitus, 
        'coronary_artery_disease': coronary_artery_disease, 
        'appetite': appetite,
        'pedal_edema': pedal_edema, 
        'anemia': anemia
        }

    # Create a button to predict the output
    if st.button('Predict'):
        input_df = create_input_df(user_inputs, category_map)
        prediction = kidney_disease.predict(input_df)
        # If 0 : Chronic Kidney Disease present
        # If 1 : Chronic Kidney Disease not present
        if prediction[0] == 0:
            kidney_diagnosis='The patient is likely to have Chronic Kidney Disease.'
        else:
            kidney_diagnosis='The patient is likely to not have Chronic Kidney Disease.'
        st.write(kidney_diagnosis)
        st.session_state['kidney_diagnosis']=kidney_diagnosis
        st.write('--'*50)
        
        with st.expander("Show more details"):
            st.write("Details of the prediction:")
            st.json(kidney_disease.get_params())
            st.write('Model used: Support Vector Machine (SVM)')
    st.subheader("Kidney Disease Prediction Risk Factors Heatmap")
    if st.button("Show Heatmap"):
        kidney_data = pd.DataFrame({
            "Blood Urea": [st.sidebar.number_input("Blood Urea", min_value=0.0, max_value=100.0, value=30.0)],
            "Hemoglobin": [st.sidebar.number_input("Hemoglobin", min_value=5.0, max_value=20.0, value=13.0)],
            "Serum Creatinine": [st.sidebar.number_input("Serum Creatinine", min_value=0.0, max_value=10.0, value=1.2)],
            "Age": [age]
        })
        # Heatmap
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(kidney_data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# Liver Disease Prediction
elif menu =="Liver Disease Prediction":
    st.title("Liver Disease Prediction")
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            Sex = 0
        elif value == "female":
            Sex = 1
    with col2:
        age = st.number_input("Entre your age") # 2 
    with col3:
        Total_Bilirubin = st.number_input("Entre your Total_Bilirubin") # 3
    with col1:
        Direct_Bilirubin = st.number_input("Entre your Direct_Bilirubin")# 4

    with col2:
        Alkaline_Phosphotase = st.number_input("Entre your Alkaline_Phosphotase") # 5
    with col3:
        Alamine_Aminotransferase = st.number_input("Entre your Alamine_Aminotransferase") # 6
    with col1:
        Aspartate_Aminotransferase = st.number_input("Entre your Aspartate_Aminotransferase") # 7
    with col2:
        Total_Protiens = st.number_input("Entre your Total_Protiens")# 8
    with col3:
        Albumin = st.number_input("Entre your Albumin") # 9
    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Entre your Albumin_and_Globulin_Ratio") # 10 
    if age == 0 or Total_Bilirubin == 0 or Alkaline_Phosphotase == 0 or Albumin == 0:
        st.warning("⚠️ Please enter valid values in the required fields (Age, Bilirubin, Alkaline Phosphotase, Albumin).")
    else:
        liver_input = [[
            Sex, age, Total_Bilirubin, Direct_Bilirubin,
            Alkaline_Phosphotase, Alamine_Aminotransferase,
            Aspartate_Aminotransferase, Total_Protiens,
            Albumin, Albumin_and_Globulin_Ratio
        ]]
        
        liver_prediction = liver_model.predict(liver_input)

        if liver_prediction[0] == 1:
            liver_dig = "We are really sorry to say but it seems like you have liver disease."
        else:
            liver_dig = "Congratulations, you don't have liver disease."

        st.success(name + ', ' + liver_dig)
        st.session_state["liver_diagnosis"] = liver_dig

# Hypertension Analysis
elif menu == "Hypertension":
    st.title(t("🩺 Hypertension Risk Prediction"))

    # Input fields
    age = st.slider(t("Age"), 1, 100, 30,help='Enter the age of the patient.')
    gender = st.radio(t("Gender"), ["Male", "Female", "Other"])
    has_heart_disease = st.radio(t("Do you have any heart disease?"), ["Yes", "No"],help='Does the patient have a history of heart disease?')
    ever_married = st.radio(t("Ever Married"), ["Yes", "No"],help='Marital status of the patient.')
    work_type = st.selectbox(t("Work Type"), ["Private", "Self-employed", "Govt_job", "children", "Never_worked"], help='Type of occupation of the patient.')
    residence_type = st.radio(t("Residence Type"), ["Urban", "Rural"],help='Residential living area of the patient.')
    avg_glucose = st.number_input(t("Average Glucose Level (mg/dL)"), min_value=50.0, max_value=300.0, step=0.1,help='Average glucose level in the blood.')
    bmi = st.number_input(t("Body Mass Index (BMI)"), min_value=10.0, max_value=60.0, step=0.1,help='Body mass index of the patient.')
    smoking_status = st.selectbox(t("Smoking Status"), ["never smoked", "formerly smoked", "smokes", "Unknown"],help='Smoking behavior of the patient.')
    systolic = st.number_input(t("Systolic (mmHg)"), min_value=70, max_value=250, value=120)
    diastolic = st.number_input(t("Diastolic (mmHg)"), min_value=40, max_value=150, value=80)

    # Encoding
    gender_map = {"Male": 1, "Female": 0, "Other": 2}
    married_map = {"Yes": 1, "No": 0}
    work_map = {"Private": 2, "Self-employed": 3, "Govt_job": 0, "children": 1, "Never_worked": 4}
    residence_map = {"Urban": 1, "Rural": 0}
    smoke_map = {"never smoked": 2, "formerly smoked": 1, "smokes": 3, "Unknown": 0}

    heart_disease = 1 if has_heart_disease == "Yes" else 0

    # Create input DataFrame with correct column order
    user_df = pd.DataFrame([{
        "gender": gender_map[gender],
        "age": age,
        "heart_disease": heart_disease,
        "ever_married": married_map[ever_married],
        "work_type": work_map[work_type],
        "Residence_type": residence_map[residence_type],
        "avg_glucose_level": avg_glucose,
        "bmi": bmi,
        "smoking_status": smoke_map[smoking_status]
    }])

    # Enforce correct column order
    expected_order = [
        'gender', 'age', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'avg_glucose_level',
        'bmi', 'smoking_status'
    ]
    user_df = user_df[expected_order]

    # Prediction
    if st.button(t("Predict Hypertension Risk")):
        model = joblib.load(MODELS_DIR / "hypertension_model.pkl")
        prediction = model.predict(user_df)[0]

        if prediction == 1:
            st.error(t("⚠️ High Risk of Hypertension Detected!"))
        else:
            st.success(t("✅ Low Risk of Hypertension."))
        category, emoji, advice = classify_blood_pressure(systolic, diastolic)
        st.markdown(f"### {emoji} Blood Pressure Category: **{category}**")
        st.markdown(f"**Advice:** {advice}")
        # Optional: Feature importance
        if hasattr(model, "coef_"):
            st.subheader(t("📊 Feature Importance"))
            importance = model.coef_[0]
            fig, ax = plt.subplots()
            ax.barh(expected_order, importance)
            ax.set_xlabel("Weight")
            st.pyplot(fig)

# Fever Detection
elif menu == "Fever":
    st.subheader("🌡️ Fever Detection & Recommendations")

    # Medical disclaimer
    st.warning("""
        ⚠️ **MEDICAL DISCLAIMER**: This tool provides general guidance only and is not a substitute for professional 
        medical advice. Always consult a healthcare provider for medical concerns.
    """)

    # Select unit
    temp_unit = st.radio("Select Temperature Unit:", ["Celsius (°C)", "Fahrenheit (°F)"], horizontal=True)

    try:
        # Load dataset with proper path handling
        fever_data_path = DATA_DIR / "enhanced_fever_medicine_recommendation.csv"
        try:
            if fever_data_path.exists():
                df = pd.read_csv(fever_data_path)
                df.columns = df.columns.str.strip().str.lower()
            else:
                st.error(f"❌ Medication dataset not found at {fever_data_path}")
                df = pd.DataFrame(columns=["fever_severity", "recommended_medication"])
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            df = pd.DataFrame(columns=["fever_severity", "recommended_medication"])

        # Use Home section data if available
        if "temperature" in st.session_state and "temp_unit" in st.session_state:
            original_temp = st.session_state["temperature"]
            original_unit = st.session_state["temp_unit"]

            # Handle unit switching
            if original_unit != temp_unit:
                if original_unit == "Fahrenheit (°F)" and temp_unit == "Celsius (°C)":
                    temperature = (original_temp - 32) * 5/9
                elif original_unit == "Celsius (°C)" and temp_unit == "Fahrenheit (°F)":
                    temperature = (original_temp * 9/5) + 32
                else:
                    temperature = original_temp
            else:
                temperature = original_temp

            st.success(f"✅ Temperature from Home Section: {temperature:.1f} {temp_unit[:-4]}")
        else:
            # Fresh user input
            if temp_unit == "Celsius (°C)":
                temperature = st.number_input("🌡️ Enter your Temperature (°C):", 30.0, 45.0, 37.0, 0.1)
            else:
                temperature = st.number_input("🌡️ Enter your Temperature (°F):", 86.0, 113.0, 98.6, 0.1)

        # Store the final unit and value in session
        st.session_state["temperature"] = temperature
        st.session_state["temp_unit"] = temp_unit

        # Convert to Celsius for logic
        temp_celsius = (temperature - 32) * 5/9 if temp_unit == "Fahrenheit (°F)" else temperature

        # Other inputs (use Home values if available)
        col1, col2 = st.columns(2)
        with col1:
            age = st.session_state.get("age") or st.number_input("🎂 Age", 0, 120, 25)
            chronic_conditions = st.selectbox("🩺 Do you have chronic conditions?", ["No", "Yes"])
        with col2:
            bmi = st.session_state.get("bmi") or st.number_input("⚖️ BMI", 10.0, 50.0, 22.0)
            pregnancy_status = st.selectbox("🤰 Are you pregnant?", ["No", "Yes", "Not applicable"])

        # Show normal range
        if age <= 3:
            normal_range_c = "36.6°C - 37.2°C"
            normal_range_f = "97.9°F - 99.0°F"
        elif age <= 10:
            normal_range_c = "36.5°C - 37.5°C"
            normal_range_f = "97.7°F - 99.5°F"
        else:
            normal_range_c = "36.1°C - 37.2°C"
            normal_range_f = "97.0°F - 99.0°F"
        st.info(f"ℹ️ Normal temperature range for your age: {normal_range_c} / {normal_range_f}")

        # Determine severity
        severity = ""
        emergency = False

        if age <= 3:
            if temp_celsius >= 38.9:
                severity, emergency = "High", True
            elif 38.0 <= temp_celsius < 38.9:
                severity = "Moderate"
            elif 37.2 < temp_celsius < 38.0:
                severity = "Low"
            elif 36.6 <= temp_celsius <= 37.2:
                severity = "None"
            else:
                severity = "Below Normal"
        elif age <= 10:
            if temp_celsius >= 39.4:
                severity, emergency = "High", True
            elif 38.0 <= temp_celsius < 39.4:
                severity = "Moderate"
            elif 37.5 < temp_celsius < 38.0:
                severity = "Low"
            elif 36.5 <= temp_celsius <= 37.5:
                severity = "None"
            else:
                severity = "Below Normal"
        else:
            if temp_celsius >= 39.0:
                severity = "High"
                if age >= 65:
                    emergency = True
            elif 37.5 <= temp_celsius < 39.0:
                severity = "Moderate"
            elif 37.2 < temp_celsius < 37.5:
                severity = "Low"
            elif 36.1 <= temp_celsius <= 37.2:
                severity = "None"
            else:
                severity = "Below Normal"

        # Show severity
        if severity == "High":
            st.error(f"🔴 Fever Severity: `{severity}`")
        elif severity == "Moderate":
            st.warning(f"🟠 Fever Severity: `{severity}`")
        elif severity == "Low":
            st.info(f"🟡 Fever Severity: `{severity}`")
        elif severity == "None":
            st.success(f"🟢 Normal Temperature")
        else:
            st.info(f"🔵 Status: `{severity}`")

        # Emergency alerts
        if emergency:
            st.error("🚨 **EMERGENCY:** High fever at your age group needs urgent attention.")

        if age <= 3 and temp_celsius >= 38.0:
            st.error("🚼 **Infant Alert:** Fevers in toddlers require pediatric consultation.")
        if pregnancy_status == "Yes" and temp_celsius >= 38.0:
            st.error("🤰 **Pregnancy Alert:** Contact your obstetrician immediately.")
        if chronic_conditions == "Yes" and severity in ["Moderate", "High"]:
            st.error("⚠️ **Chronic Condition Alert:** Please consult your doctor.")

        # Medication & Tips
        if severity not in ["None", "Below Normal"]:
            match = df[df["fever_severity"].str.lower() == severity.lower()]
            if not match.empty:
                med = match["recommended_medication"].mode()[0]
                if "acetaminophen" in med.lower() or "paracetamol" in med.lower():
                    dosage = "Child: weight-based. Adult: 500–1000mg every 4–6h, max 4g/day."
                elif "ibuprofen" in med.lower():
                    dosage = "Child: weight-based. Adult: 200–400mg every 4–6h, max 1.2g/day."
                else:
                    dosage = "Check label or consult doctor."
                st.success(f"💊 Recommended Medication: **{med}**")
                st.info(f"📋 Dosage: {dosage}")
                st.warning("⚠️ Always follow label instructions. Avoid self-medication during pregnancy or chronic illness.")

            # Fever care
            st.subheader("🌡️ Fever Care Tips")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                ✅ Rest and hydrate  
                ✅ Use light clothing  
                ✅ Lukewarm sponging  
                ✅ Regularly monitor temperature
                """)
            with col2:
                st.markdown("""
                ❌ Avoid ice baths  
                ❌ Avoid alcohol rubs  
                ❌ Don’t double medications  
                ❌ Don’t ignore persistent fever
                """)

            st.subheader("🚨 Seek Emergency Care If:")
            st.error("""
            - Temp > 103°F (39.4°C) and not reducing  
            - Stiff neck or severe headache  
            - Difficulty breathing  
            - Confusion or drowsiness  
            - Seizures  
            - Severe rash or vomiting
            """)

        elif severity == "Below Normal":
            st.warning("""
                🧊 Temperature is unusually low. May be due to cold exposure or health conditions.
                Seek evaluation if symptoms persist.
            """)
        else:
            st.success("✅ No fever detected. You're good!")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")



elif menu == "Symptom Tracker":
    st.header("🩺 Symptom Tracker")

    # Keep existing symptom tracking code
    selected_symptoms = st.multiselect("Select symptoms you're experiencing today:", 
                                       ["Fever", "Cough", "Fatigue", "Headache", "Chest Pain", 
                                        "Shortness of Breath", "Nausea", "Sore Throat", 
                                        "Runny Nose", "Body Aches", "Dizziness", "Chills"])
    
    log_date = st.date_input("Date", datetime.now().date())

    # Add symptom duration tracking
    duration_options = ["Started today", "1-2 days", "3-5 days", "1-2 weeks", "More than 2 weeks"]
    symptom_duration = {}
    symptom_severity = {}
    
    if selected_symptoms:
        st.subheader("Symptom Details:")
        cols = st.columns(2)
        
        for i, symptom in enumerate(selected_symptoms):
            with cols[i % 2]:
                st.markdown(f"**{symptom}**")
                severity = st.select_slider(
                    f"Severity:",
                    ["Mild", "Moderate", "Severe"],
                    key=f"severity_{symptom}"
                )
                duration = st.selectbox(
                    f"Duration:",
                    duration_options,
                    key=f"duration_{symptom}"
                )
                symptom_severity[symptom] = severity
                symptom_duration[symptom] = duration
                st.markdown("---")

    # Access session state for symptom log
    if "symptom_log" not in st.session_state:
        st.session_state["symptom_log"] = pd.DataFrame(columns=["Date", "Symptom", "Severity", "Duration"])

    if st.button("Log Symptoms"):
        for symptom, severity in symptom_severity.items():
            duration = symptom_duration.get(symptom, "Started today")
            new_entry = pd.DataFrame([[log_date, symptom, severity, duration]], 
                                    columns=["Date", "Symptom", "Severity", "Duration"])
            st.session_state["symptom_log"] = pd.concat([st.session_state["symptom_log"], new_entry], ignore_index=True)
        st.success("Symptoms logged successfully!")
        st.rerun()  # Refresh to show updated analysis

    # Show Symptom Log
    st.subheader("📝 Symptom Log")
    if not st.session_state["symptom_log"].empty:
        st.dataframe(st.session_state["symptom_log"])
        
        # Add pattern analysis
        matches = analyze_symptom_patterns(st.session_state["symptom_log"])
        analyze_and_display_patterns(st.session_state["symptom_log"])
        suggest_next_health_actions(st.session_state["symptom_log"], matches)
    else:
        st.info("No symptoms logged yet. Start tracking your symptoms to see pattern analysis.")