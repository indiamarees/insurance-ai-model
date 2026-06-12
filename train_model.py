import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Use the same directory as this script to locate insurance.csv.
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "insurance_data.csv")

# 1. Load the data using pandas.
df = pd.read_csv(data_path)

# Show the first few rows so we understand the data layout.
print("First rows of the dataset:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# 2. Clean the data: convert categorical text columns into numeric values.
# Here Policy_Type is categorical, so we use one-hot encoding.
if "Policy_Type" in df.columns:
    df = pd.get_dummies(df, columns=["Policy_Type"], drop_first=True)

# If Claim_ID is present, drop it since it is an identifier and not predictive.
if "Claim_ID" in df.columns:
    df = df.drop(columns=["Claim_ID"])

# 3. Separate the target column 'Claim_Status'.
if "Claim_Status" not in df.columns:
    raise ValueError("The dataset must contain a 'Claim_Status' column.")
target = "Claim_Status"
X = df.drop(columns=[target])
y = df[target]

print("\nFeatures used for training:")
print(X.columns.tolist())
print("\nTarget distribution:")
print(y.value_counts())

# 4. Split into training and testing sets with 80/20 ratio.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 5. Train a RandomForestClassifier.
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 6. Print accuracy and a classification report.
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification report:")
print(report)
