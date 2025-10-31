import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Simulate patient data
data = {
    'Age': [30, 45, 60, 25, 55, 70, 35, 50, 65, 40],
    'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    'Smoker': [0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
    'Blood_Pressure_Systolic': [120, 140, 160, 110, 150, 170, 125, 145, 165, 130],
    'Cholesterol': [180, 220, 250, 160, 230, 260, 190, 225, 255, 200],
    'Disease_Progression': [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]  # 0 for no progression, 1 for progression
}
df = pd.DataFrame(data)

# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

X = df[['Age', 'Smoker', 'Blood_Pressure_Systolic', 'Cholesterol', 'Gender_M']]
y = df['Disease_Progression']

# Split data into training and testing sets (stratify to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train a Decision Tree Classifier model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
print("--- Predicted Output ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Simulate a new patient for prognosis
new_patient_data = {
    'Age': [58],
    'Smoker': [1],
    'Blood_Pressure_Systolic': [155],
    'Cholesterol': [240],
    'Gender_M': [1]  # Assuming male
}
new_patient_df = pd.DataFrame(new_patient_data)
new_patient_prediction = model.predict(new_patient_df)

print("\n--- New Patient Prognosis ---")
if new_patient_prediction[0] == 1:
    print("Prognosis: High likelihood of disease progression.")
else:
    print("Prognosis: Low likelihood of disease progression.")