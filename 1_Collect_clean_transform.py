import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# 1. Data Collection (Simulated for demonstration)
data = {
    'PatientID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [45, 62, 30, 55, 70, 28, 50, 68, 35, 42],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Fever': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Cough': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No'],
    'Fatigue': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'DifficultyBreathing': ['No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'BloodPressure': [120, 140, 110, 130, 150, 115, 125, 145, 118, 122],
    'Cholesterol': [200, 240, 180, 220, 260, 190, 210, 250, 185, 205],
    'Diagnosis': ['Flu', 'Pneumonia', 'Asthma', 'Bronchitis', 'Pneumonia', 'Flu', 'Asthma', 'Bronchitis', 'Flu',
                  'Asthma'],
    'Hospital_Visit_Date': ['2023-01-15', '2023-01-20', '2023-01-22', '2023-02-01', '2023-02-05',
                            '2023-02-10', '2023-02-12', '2023-02-18', '2023-02-20', '2023-02-25']
}
df = pd.DataFrame(data)

# Simulate a second source with some missing data and different column names
data_source2 = {
    'ID': [11, 12, 13, 14, 15],
    'Age_Years': [38, None, 65, 52, 48],
    'Sex': ['Female', 'Male', 'Female', 'Male', 'Female'],
    'Has_Fever': [1, 0, 1, 0, 1],
    'BP_Systolic': [128, 135, None, 130, 120],
    'Cholesterol_Level': [210, 230, 200, 225, None],
    'Condition': ['Flu', 'Bronchitis', 'Pneumonia', 'Flu', 'Asthma']
}
df_source2 = pd.DataFrame(data_source2)

print("--- Original DataFrames ---")
print("DataFrame 1:")
print(df.head())
print("\nDataFrame 2:")
print(df_source2.head())
print("-" * 30)

# 2. Data Cleaning
# Handle missing values
imputer_age = SimpleImputer(strategy='mean')
df_source2['Age_Years'] = imputer_age.fit_transform(df_source2[['Age_Years']])

imputer_bp = SimpleImputer(strategy='mean')
df_source2['BP_Systolic'] = imputer_bp.fit_transform(df_source2[['BP_Systolic']])

imputer_cholesterol = SimpleImputer(strategy='mean')
df_source2['Cholesterol_Level'] = imputer_cholesterol.fit_transform(df_source2[['Cholesterol_Level']])

# Convert 'Yes'/'No' to numerical (0/1) for symptoms in df
for col in ['Fever', 'Cough', 'Fatigue', 'DifficultyBreathing']:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Convert 'Has_Fever' (0/1) to 'Fever' (0/1) in df_source2
df_source2['Fever'] = df_source2['Has_Fever']
df_source2.drop(columns=['Has_Fever'], inplace=True)

print("\n--- After Cleaning Missing Values and Initial Conversions ---")
print("DataFrame 1:")
print(df.head())
print("\nDataFrame 2:")
print(df_source2.head())
print("-" * 30)

# 3. Data Integration
# Rename columns in df_source2 to match df for integration
df_source2 = df_source2.rename(columns={
    'ID': 'PatientID',
    'Age_Years': 'Age',
    'Sex': 'Gender',
    'BP_Systolic': 'BloodPressure',
    'Cholesterol_Level': 'Cholesterol',
    'Condition': 'Diagnosis'
})

# Add missing symptom columns to df_source2 and fill with 0 (assuming absence if not explicitly stated)
for col in ['Cough', 'Fatigue', 'DifficultyBreathing']:
    if col not in df_source2.columns:
        df_source2[col] = 0

# Ensure Hospital_Visit_Date exists so we can reorder columns to match df
if 'Hospital_Visit_Date' not in df_source2.columns:
    df_source2['Hospital_Visit_Date'] = pd.NaT

# Select and reorder columns in df_source2 to match df
df_source2 = df_source2[df.columns.tolist()]

# Concatenate the dataframes
integrated_df = pd.concat([df, df_source2], ignore_index=True)

print("\n--- After Integration (Concatenation) ---")
print(integrated_df.head(12))
print("-" * 30)

# 4. Data Transformation
# Convert categorical features to numerical using Label Encoding
label_encoder_gender = LabelEncoder()
integrated_df['Gender_Encoded'] = label_encoder_gender.fit_transform(integrated_df['Gender'])

label_encoder_diagnosis = LabelEncoder()
integrated_df['Diagnosis_Encoded'] = label_encoder_diagnosis.fit_transform(integrated_df['Diagnosis'])

# Feature Scaling for numerical features
scaler = StandardScaler()
numerical_cols = ['Age', 'BloodPressure', 'Cholesterol']
integrated_df[numerical_cols] = scaler.fit_transform(integrated_df[numerical_cols])

# Convert 'Hospital_Visit_Date' to datetime objects and extract features
integrated_df['Hospital_Visit_Date'] = pd.to_datetime(integrated_df['Hospital_Visit_Date'])
integrated_df['Visit_Month'] = integrated_df['Hospital_Visit_Date'].dt.month
integrated_df['Visit_DayOfWeek'] = integrated_df['Hospital_Visit_Date'].dt.dayofweek

# Drop original categorical columns if encoded versions are preferred for modeling
transformed_df = integrated_df.drop(columns=['Gender', 'Diagnosis', 'Hospital_Visit_Date'])

print("\n--- After Transformation ---")
print(transformed_df.head(12))
print("\nDataFrame Info after Transformation:")
print(transformed_df.info())
print("\nUnique values for 'Gender_Encoded':", transformed_df['Gender_Encoded'].unique())
print("Unique values for 'Diagnosis_Encoded':", transformed_df['Diagnosis_Encoded'].unique())
print("-" * 30)
print("\n--- Final Transformed Dataset Sample ---")
print(transformed_df.sample(5))