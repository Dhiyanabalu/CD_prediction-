import pyrebase
import streamlit as st

# Load Firebase config from Streamlit secrets or use environment variables
try:
    # Try to load from Streamlit secrets (for Streamlit Cloud deployment)
    firebaseConfig = {
        "apiKey": st.secrets.get("firebase_api_key", "AIzaSyBDcYsiOXYvmarHstAo_dpT84jcV8N10b8"),
        "authDomain": st.secrets.get("firebase_auth_domain", "cdprediction-65490.firebaseapp.com"),
        "projectId": st.secrets.get("firebase_project_id", "cdprediction-65490"),
        "storageBucket": st.secrets.get("firebase_storage_bucket", "cdprediction-65490.firebasestorage.app"),
        "messagingSenderId": st.secrets.get("firebase_messaging_sender_id", "1078142938599"),
        "appId": st.secrets.get("firebase_app_id", "1:1078142938599:web:8be9d3a723f1918cec6142"),
        "measurementId": st.secrets.get("firebase_measurement_id", "G-39WCM4KPLS"),
        "databaseURL": st.secrets.get("firebase_database_url", "")
    }
except Exception as e:
    # Fallback to hardcoded values for local development
    firebaseConfig = {
        "apiKey": "AIzaSyBDcYsiOXYvmarHstAo_dpT84jcV8N10b8",
        "authDomain": "cdprediction-65490.firebaseapp.com",
        "projectId": "cdprediction-65490",
        "storageBucket": "cdprediction-65490.firebasestorage.app",
        "messagingSenderId": "1078142938599",
        "appId": "1:1078142938599:web:8be9d3a723f1918cec6142",
        "measurementId": "G-39WCM4KPLS",
        "databaseURL": ""
    }

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()