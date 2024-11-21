
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Inspect the dataset
print("Dataset Overview:")
print(df.head())
print("\nClass Distribution:")
print(df['Class'].value_counts())  # Class 1 = Fraud, Class 0 = Non-fraud

# Step 1: Check for Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 2: Normalize the 'Amount' column
scaler = StandardScaler()
df['Normalized_Amount'] = scaler.fit_transform(df[['Amount']])
df.drop(['Amount'], axis=1, inplace=True)  # Drop original 'Amount' column

# Step 3: Handle Imbalance using SMOTE
X = df.drop(['Class'], axis=1)
y = df['Class']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Display preprocessed data information
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Save the preprocessed data
X_train.to_csv('X_train_preprocessed.csv', index=False)
X_test.to_csv('X_test_preprocessed.csv', index=False)
y_train.to_csv('y_train_preprocessed.csv', index=False)
y_test.to_csv('y_test_preprocessed.csv', index=False)

print("Preprocessed data saved as CSV files.")
