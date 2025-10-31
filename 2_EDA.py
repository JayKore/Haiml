import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Try loading dataset; fall back to a sample if file not found
try:
    df = pd.read_csv('healthcare_data.csv')
except FileNotFoundError:
    print("Error: 'healthcare_data.csv' not found. Using a sample dataset for demonstration.")
    data = {
        'Age': [25, 30, 45, 60, 35, 50, 28, 40, 55, 65],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
        'Blood_Pressure': [120, 130, 140, 150, 125, 135, 122, 138, 145, 160],
        'Cholesterol': [180, 200, 220, 240, 190, 210, 185, 215, 230, 250],
        'Heart_Rate': [70, 75, 80, 85, 72, 78, 71, 82, 88, 90],
        'Diagnosis': ['Normal', 'Normal', 'High BP', 'High BP', 'Normal', 'High BP', 'Normal', 'High BP', 'High BP', 'High BP']
    }
    df = pd.DataFrame(data)

# Basic info and descriptive stats
print("--- Dataset Information ---")
df.info()
print("\n--- First 5 rows of the dataset ---")
print(df.head())
print("\n--- Descriptive Statistics ---")
print(df.describe(include='all'))
print("\n--- Count of unique values in 'Gender' ---")
print(df['Gender'].value_counts())
print("\n--- Mean Blood Pressure by Gender ---")
print(df.groupby('Gender')['Blood_Pressure'].mean())

# Scatter plot: Age vs Blood Pressure colored by Gender
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Blood_Pressure', hue='Gender', data=df)
plt.title('Age vs. Blood Pressure by Gender')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.grid(True)
plt.show()

# Histogram: Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=5, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Box plot: Cholesterol by Diagnosis
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diagnosis', y='Cholesterol', data=df)
plt.title('Cholesterol Levels by Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Cholesterol')
plt.grid(True)
plt.show()

# Correlation matrix for numerical columns
numeric_df = df.select_dtypes(include=['number'])
if not numeric_df.empty:
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()
else:
    print("\nNo numerical columns found for correlation matrix.")