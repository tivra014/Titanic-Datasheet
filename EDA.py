# -*- coding: utf-8 -*-
"""Untitled3.ipynb



Original file is located at
    https://colab.research.google.com/drive/1JSQqlA5eNOFYLB5hIBoPi4mzivXI0t-n
"""

import pandas as pd

# Data loading
df = pd.read_csv('Titanic-Dataset 2.csv')
# display(df.head())  # Optional: display the first 5 rows

# Data exploration
print("DataFrame shape:", df.shape)
print("\nDataFrame info:")
df.info()
print("\nMissing values per column:")
print(df.isnull().sum())
categorical_cols = ['Survived', 'Pclass', 'Sex', 'Embarked']
for col in categorical_cols:
    print(f"\nUnique values in '{col}' column:")
    print(df[col].unique())

# Data analysis
numerical_features = ['Age', 'Fare']
print("Descriptive Statistics:")
print(df[numerical_features].describe())
df['Age'].fillna(df['Age'].median(), inplace=True)
categorical_features = ['Pclass', 'Sex', 'Embarked']
for feature in categorical_features:
    print(f"\nSurvival Rate by {feature}:")
    print(df.groupby(feature)['Survived'].mean())
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 60, 100], labels=['Child', 'Adult', 'Senior'])
print("\nSurvival Rate by Age Group:")
print(df.groupby('Age_Group')['Survived'].mean())
df['Fare_Group'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very_High'])
print("\nSurvival Rate by Fare Group:")
print(df.groupby('Fare_Group')['Survived'].mean())
print("\nCorrelation Matrix:")
correlation_matrix = df[['Survived', 'Age', 'Fare']].corr()
print(correlation_matrix)