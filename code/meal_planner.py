import pandas as pd
import re
import random
from pathlib import Path

# Dataset path - relative to this file
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "Indian_Food_DF.csv"

# Load and clean meal data
def load_meal_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Meal dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)

    def extract_kcal(energy_str):
        if pd.isna(energy_str):
            return 0
        match = re.search(r"(\d+)\s*kcal", energy_str)
        return int(match.group(1)) if match else 0

    def clean_grams(value):
        if pd.isna(value):
            return 0
        match = re.search(r"([\d.]+)", str(value))
        return float(match.group(1)) if match else 0

    df['Calories'] = df['nutri_energy'].apply(extract_kcal)
    df['Fibre'] = df['nutri_fiber'].apply(clean_grams)
    df['Sugars'] = df['nutri_sugar'].apply(clean_grams)

    return df[['name', 'Calories', 'Fibre', 'Sugars']]

# Generate personalized meal plan
def get_personalized_meal_plan(glucose, bmi):
    df = load_meal_data()

    # Personalization based on glucose and BMI
    if glucose > 250:
        # Very High Glucose ➔ Strict low-carb diet
        filtered = df[
            (df['Calories'] <= 250) &
            (df['Fibre'] >= 3) &
            (df['Sugars'] <= 5)
        ]
    elif glucose > 180 or bmi > 30:
        # Diabetic or High BMI ➔ Moderate low-carb diet
        filtered = df[
            (df['Calories'] <= 350) &
            (df['Fibre'] >= 2) &
            (df['Sugars'] <= 10)
        ]
    else:
        # Normal patients ➔ Balanced meals
        filtered = df[
            (df['Calories'] <= 450) &
            (df['Fibre'] >= 1)
        ]

    # Shuffle for variety
    filtered = filtered.sample(frac=1).reset_index(drop=True)

    # Return Breakfast, Lunch, Dinner suggestions separately
    meal_plan = {
        "Breakfast": filtered.head(3).to_dict(orient="records"),
        "Lunch": filtered.iloc[3:6].to_dict(orient="records"),
        "Dinner": filtered.iloc[6:9].to_dict(orient="records"),
    }

    return meal_plan
