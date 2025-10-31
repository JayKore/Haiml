import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Simulate patient data
np.random.seed(42)
data_size = 1000
patient_data = pd.DataFrame({
    'Age': np.random.randint(20, 80, data_size),
    'Gender': np.random.choice(['Male', 'Female'], data_size),
    'Blood_Pressure': np.random.randint(90, 180, data_size),
    'Cholesterol': np.random.randint(150, 300, data_size),
    'Glucose': np.random.randint(70, 200, data_size),
    'Smoking': np.random.choice([0, 1], data_size),
    'Alcohol': np.random.choice([0, 1], data_size),
    'Physical_Activity': np.random.choice([0, 1], data_size, p=[0.3, 0.7]),
    'Family_History': np.random.choice([0, 1], data_size, p=[0.7, 0.3]),
    'Disease_Risk': np.random.choice(['Low', 'Medium', 'High'], data_size, p=[0.5, 0.3, 0.2])
})

# Quick visible data summary (guarantees output)
print(">>> Sample of simulated patient_data (first 5 rows):", flush=True)
print(patient_data.head(), flush=True)
print("\n>>> Class distribution (Disease_Risk counts):", flush=True)
print(patient_data['Disease_Risk'].value_counts(), flush=True)
print("\n>>> Data shape:", patient_data.shape, flush=True)

# Encode categorical features
le_gender = LabelEncoder()
patient_data['Gender'] = le_gender.fit_transform(patient_data['Gender'])

# Separate target encoder so we can inverse transform predictions
le_target = LabelEncoder()
patient_data['Disease_Risk_Encoded'] = le_target.fit_transform(patient_data['Disease_Risk'])
print("\n>>> Target classes (encoded):", flush=True)
print(dict(enumerate(le_target.classes_)), flush=True)

# Prepare features and target
X = patient_data.drop(['Disease_Risk', 'Disease_Risk_Encoded'], axis=1)
y = patient_data['Disease_Risk_Encoded']

# Train/test split (stratify to preserve class ratios)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Train Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
gb_prob = gb_model.predict_proba(X_test)
gb_accuracy = accuracy_score(y_test, gb_predictions)

# Train AdaBoost
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)
ada_predictions = ada_model.predict(X_test)
# AdaBoost may not implement predict_proba if base estimator doesn't; handle safely
try:
    ada_prob = ada_model.predict_proba(X_test)
except Exception:
    ada_prob = None
ada_accuracy = accuracy_score(y_test, ada_predictions)

# Visible performance output
print("\n--- Random Forest Model Performance ---", flush=True)
print(f"Accuracy: {rf_accuracy:.4f}", flush=True)
print("Classification Report:", flush=True)
print(classification_report(y_test, rf_predictions, target_names=le_target.classes_), flush=True)

print("\n--- Gradient Boosting Model Performance ---", flush=True)
print(f"Accuracy: {gb_accuracy:.4f}", flush=True)
print("Classification Report:", flush=True)
print(classification_report(y_test, gb_predictions, target_names=le_target.classes_), flush=True)

print("\n--- AdaBoost Model Performance ---", flush=True)
print(f"Accuracy: {ada_accuracy:.4f}", flush=True)
print("Classification Report:", flush=True)
print(classification_report(y_test, ada_predictions, target_names=le_target.classes_), flush=True)

# Feature importances (from tree-based models)
print("\n>>> Feature importances (Random Forest):", flush=True)
rf_importances = dict(zip(X.columns, rf_model.feature_importances_))
print(rf_importances, flush=True)

print("\n>>> Feature importances (Gradient Boosting):", flush=True)
gb_importances = dict(zip(X.columns, gb_model.feature_importances_))
print(gb_importances, flush=True)

# Example new patient and visible predictions + probabilities
new_patient_data = pd.DataFrame(
    [[55, 0, 130, 220, 100, 0, 0, 1, 1]],
    columns=['Age', 'Gender', 'Blood_Pressure', 'Cholesterol', 'Glucose',
             'Smoking', 'Alcohol', 'Physical_Activity', 'Family_History']
)

print("\n--- New patient features ---", flush=True)
print(new_patient_data, flush=True)

# Random Forest prediction & probability
predicted_risk_rf = rf_model.predict(new_patient_data)
predicted_risk_label_rf = le_target.inverse_transform(predicted_risk_rf)[0]
predicted_risk_rf_proba = rf_model.predict_proba(new_patient_data)[0]
print(f"\nRandom Forest predicted risk: {predicted_risk_label_rf}", flush=True)
print(f"Random Forest class probabilities: {dict(zip(le_target.classes_, predicted_risk_rf_proba))}", flush=True)

# Gradient Boosting prediction & probability
predicted_risk_gb = gb_model.predict(new_patient_data)
predicted_risk_label_gb = le_target.inverse_transform(predicted_risk_gb)[0]
predicted_risk_gb_proba = gb_model.predict_proba(new_patient_data)[0]
print(f"\nGradient Boosting predicted risk: {predicted_risk_label_gb}", flush=True)
print(f"Gradient Boosting class probabilities: {dict(zip(le_target.classes_, predicted_risk_gb_proba))}", flush=True)

# AdaBoost prediction & probability (if available)
predicted_risk_ada = ada_model.predict(new_patient_data)
predicted_risk_label_ada = le_target.inverse_transform(predicted_risk_ada)[0]
print(f"\nAdaBoost predicted risk: {predicted_risk_label_ada}", flush=True)
if ada_prob is not None:
    ada_proba_new = ada_model.predict_proba(new_patient_data)[0]
    print(f"AdaBoost class probabilities: {dict(zip(le_target.classes_, ada_proba_new))}", flush=True)
else:
    print("AdaBoost predict_proba not available for this estimator.", flush=True)

# Ensemble suggestion and how to create a simple voting ensemble (printed so user can run if desired)
print("\n--- Ensemble suggestion ---", flush=True)
print("You can combine models with VotingClassifier for a single combined prediction.", flush=True)
print("Example:", flush=True)
print("from sklearn.ensemble import VotingClassifier", flush=True)
print("voter = VotingClassifier(estimators=[('rf', rf_model), ('gb', gb_model), ('ada', ada_model)], voting='soft')", flush=True)
print("voter.fit(X_train, y_train); voter.predict(new_patient_data)", flush=True)