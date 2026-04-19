import pyrebase

firebaseConfig = {
    "apiKey": "AIzaSyBDcYsiOXYvmarHstAo_dpT84jcV8N10b8",
    "authDomain": "cdprediction-65490.firebaseapp.com",
    "projectId": "cdprediction-65490",
    "storageBucket": "cdprediction-65490.firebasestorage.app",
    "messagingSenderId": "1078142938599",
   "appId": "1:1078142938599:web:8be9d3a723f1918cec6142",
    "measurementId": "G-39WCM4KPLS",
    "databaseURL":""
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()