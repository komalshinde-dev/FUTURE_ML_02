# -------------------------------------------
# FUTURE INTERNS – MACHINE LEARNING TASK 2
# CHURN PREDICTION SYSTEM
# -------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import os

# ----------------------------
# STEP 1: Load or download dataset
# ----------------------------
file_path = "churn.csv"
if not os.path.exists(file_path):
    print("🔽 Downloading Telco Customer Churn dataset...")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    urllib.request.urlretrieve(url, file_path)
    print("✅ Dataset downloaded successfully as 'churn.csv'")

# Load dataset
data = pd.read_csv(file_path)
print("\n📊 Data Preview:")
print(data.head())

# ----------------------------
# STEP 2: Data Cleaning
# ----------------------------
# Replace spaces in column names
data.columns = data.columns.str.strip()

# Convert 'TotalCharges' to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna(subset=['TotalCharges'])

# Drop CustomerID (not useful for model)
data = data.drop('customerID', axis=1)

# ----------------------------
# STEP 3: Encode categorical features
# ----------------------------
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# ----------------------------
# STEP 4: Split data into features and labels
# ----------------------------
X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# STEP 5: Train Random Forest Classifier
# ----------------------------
print("\n⚙️ Training model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------------------------
# STEP 6: Evaluate model
# ----------------------------
print("\n✅ Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ----------------------------
# STEP 7: Feature Importance Visualization
# ----------------------------
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=importances[:10], y=importances.index[:10])
plt.title("Top 10 Important Features in Churn Prediction")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# ----------------------------
# STEP 8: Save results
# ----------------------------
output_file = "churn_predictions.csv"
predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
predictions_df.to_csv(output_file, index=False)
print(f"\n📁 Predictions saved to '{output_file}'")

print("\n🎉 Task 2 Completed Successfully!")
