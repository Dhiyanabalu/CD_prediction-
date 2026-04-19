# CDPrediction

CDPrediction is a Streamlit-based healthcare dashboard for multi-disease prediction and health monitoring.

## Features

- Login and registration (Firebase)
- Health dashboard (blood pressure, sugar, heart rate, temperature, BMI)
- Disease prediction modules:
  - General symptom-based disease prediction
  - Diabetes disease prediction
  - Heart disease prediction
  - Kidney disease prediction
  - Liver disease prediction
  - Hypertension risk prediction
  - Fever analysis and recommendations
- Symptom tracker with pattern insights
- PDF report generation
- English/Tamil translation support

## Project Structure

- `cdpredict.py`: Main Streamlit app
- `requirements.txt`: Python dependencies
- `models/`: Trained model files
- `data/`: Datasets used by modules
- `config/firebase_config.py`: Firebase auth configuration

## Prerequisites

- Windows
- Python 3.10+ (project currently uses a local venv with Python 3.13)
- Git (optional, for cloning)

## Setup

### 1. Open project folder

Use the folder that contains `cdpredict.py`:

- `D:\CDPrediction\CDPrediction`

### 2. Create and activate virtual environment (if needed)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation due execution policy, run using Python directly (see Run section).

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the app

From `D:\CDPrediction\CDPrediction` run:

```powershell
python -m streamlit run cdpredict.py
```

Or, if your environment uses the existing interpreter path:

```powershell
d:/CDPrediction/.venv/Scripts/python.exe -m streamlit run cdpredict.py
```

By default, Streamlit opens on:

- `http://localhost:8501`

If port 8501 is busy, use another port:

```powershell
python -m streamlit run cdpredict.py --server.port 8502
```

## Firebase Notes

Authentication is configured in:

- `config/firebase_config.py`

If login/register is not working, verify Firebase project credentials and auth method settings in the Firebase console.

## Common Issues

### 1) `No module named streamlit`

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### 2) Model file not found

Make sure you run from the same folder as `cdpredict.py` and keep the `models/` directory intact.

### 3) Registration error `EMAIL_EXISTS`

This means that email is already registered. Use login or register with a different email.

## Quick Start (copy/paste)

```powershell
cd D:\CDPrediction\CDPrediction
python -m pip install -r requirements.txt
python -m streamlit run cdpredict.py
```
