import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Drop ID column if exists
df = df.drop(columns=["id"], errors="ignore")

# Drop rows with missing values (or fillna if needed)
df = df.dropna()



# Encode categorical variables
label_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop(columns=["hypertension", "stroke"])  # Use all features except target
y = df["hypertension"]
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training feature order:", X.columns.tolist())

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
joblib.dump(model, "hypertension_model.pkl")