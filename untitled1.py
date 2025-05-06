# -*- coding: utf-8 -*-
"""Untitled1.ipynb



Original file is located at
    https://colab.research.google.com/drive/1liwywFiNe2YMtb4Sj0MitpK4Sg1aR1x7
"""

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Display first few rows
print(df.head())

# Step 1: Check missing values
print("\nMissing values:\n", df.isnull().sum())

# Step 2: Handle missing values
# Example strategy: Fill 'Age' with median, 'Embarked' with mode
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' because it has too many missing values
df.drop(columns=['Cabin'], inplace=True)

# Step 3: Encode categorical variables
# Example: 'Sex' and 'Embarked'
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Step 4: Drop unnecessary columns
# 'Name', 'Ticket' might not be useful directly
df.drop(columns=['Name', 'Ticket'], inplace=True)

# Step 5: Feature Scaling (optional, mostly for ML models)
scaler = StandardScaler()
# Let's scale 'Age' and 'Fare'
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Final cleaned data
print("\nCleaned and Preprocessed Data:\n", df.head())