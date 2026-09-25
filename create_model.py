import os
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
dataset = pd.read_csv(
    "Dataset/PS_20174392719_1491204439457_log.csv"
)

# Target
Y = dataset["isFraud"].to_numpy()

# Remove unused columns
dataset.drop(
    ["step", "type", "isFraud", "isFlaggedFraud"],
    axis=1,
    inplace=True
)

# Encode text columns
label_encoders = {}

for column in dataset.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    dataset[column] = le.fit_transform(
        dataset[column].astype(str)
    )
    label_encoders[column] = le

# Convert to numeric
dataset = dataset.apply(
    pd.to_numeric,
    errors="coerce"
)

# Fill missing values
dataset.fillna(
    dataset.mean(numeric_only=True),
    inplace=True
)

# Features
X = dataset.to_numpy()

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Load existing train/test split
data = np.load(
    "model/data.npy",
    allow_pickle=True
)

X_train, X_test, y_train, y_test = data

print("Training data:", X_train.shape)

# Apply SMOTE
smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train,
    y_train
)

print("Balanced training data:", X_train_balanced.shape)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=10,
    random_state=42,
    n_jobs=1
)

rf.fit(
    X_train_balanced,
    y_train_balanced
)

# Create model directory
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(
    rf,
    "model/random_forest.pkl"
)

# Save scaler
joblib.dump(
    scaler,
    "model/scaler.pkl"
)

# Save label encoders
joblib.dump(
    label_encoders,
    "model/label_encoders.pkl"
)

# Save training column information
joblib.dump(
    list(dataset.columns),
    "model/feature_columns.pkl"
)

print("Model saved successfully.")
print("Created:")
print("model/random_forest.pkl")
print("model/scaler.pkl")
print("model/label_encoders.pkl")
print("model/feature_columns.pkl")