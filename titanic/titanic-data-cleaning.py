# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# load dataset
df = pd.read_csv("Titanic-Dataset.csv")


# show first 5 rows
print("First 5 rows:")
print(df.head())


# check rows and columns
print("\nShape of dataset:")
print(df.shape)


# check information about dataset
print("\nDataset information:")
df.info()


# statistical summary
print("\nStatistical summary:")
print(df.describe())


# check missing values
print("\nMissing values:")
print(df.isnull().sum())


# check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())


# fill missing values in Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())


# fill missing values in Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


# remove Cabin column
df = df.drop("Cabin", axis=1)


# check missing values after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())


# convert Sex into numbers
# male = 0 and female = 1
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})


# convert Embarked into numbers
# S = 0, C = 1, Q = 2
df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})


# show data after preprocessing
print("\nData after preprocessing:")
print(df.head())


# survival count graph
plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()


# survival by gender
plt.figure(figsize=(6, 4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.xlabel("Sex (0 = Male, 1 = Female)")
plt.ylabel("Number of Passengers")
plt.show()


# age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()


# fare distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.show()


# correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# final dataset information
print("\nFinal dataset information:")
df.info()


# save cleaned dataset
df.to_csv("Titanic_Cleaned.csv", index=False)

print("\nData cleaning and preprocessing completed!")
print("Cleaned dataset saved as Titanic_Cleaned.csv")