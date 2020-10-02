# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 20:11:40 2020

@author: seelc
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import accuracy_score

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns




'''STEP1: First loading and cleaning data'''

df = pd.read_csv("C:\\Users\\seelc\\OneDrive\\Desktop\\Projects\\Titanic Kaggle\\train.csv")

#First cleaning up the age column, if their age is empty set equal to the median age
df['Age'] = df['Age'].fillna(df['Age'].median())



#Cleaning up the cabin column. If there is no cabin listed set equal to zero
df['Cabin'] = df['Cabin'].fillna("None")

#Creating cabin level and cabin number in order to generate easy to analyse factors
df['Cabin_Level'] = df['Cabin'].copy()
df['Cabin_Number'] = df['Cabin'].copy()

for index, value in df['Cabin_Level'].items():
    if value == "None":
        pass
    else:
        split_cabin = value.split(" ")
        level = split_cabin[0][0:1]
        number = split_cabin[0][1:]
        df['Cabin_Level'][index] = level
        df['Cabin_Number'][index] = number
        
ticket_int = df['Ticket'].copy().rename("ticket_int")
ticket_str = df['Ticket'].copy().rename("ticket_str")

#Splitting ticket into two columns one with the integer and one with the string
for index, value in ticket_int.items():
    value = value.split(" ")
    if len(value) == 1:
        ticket_str[index] = 0
        ticket_int[index] = value[0]
    elif len(value) == 2:
        ticket_int[index] = value[1]
        ticket_str[index] = value[0]
        
df = pd.concat([df, ticket_int, ticket_str], axis = 1, sort = False)
df["Cabin_Number"] = df["Cabin_Number"].replace('',0)
df["Cabin_Number"] = df["Cabin_Number"].replace("None",0)
df["Cabin_Number"] = df["Cabin_Number"].astype(int)
print(df.info())
#Dropping rows without boarding information
df = df.dropna()

'''STEP2: Transforming data in preparation for machine learning'''

first_df = df[["Sex", "Pclass", "Age", "Fare", "Survived", "Cabin_Level", "Cabin_Number", "Embarked", "Parch"]].copy()


#First encoding categorical variables
label_encoder = LabelEncoder()
first_df["Sex"] = label_encoder.fit_transform(first_df["Sex"])
first_df["Cabin_Level"] = label_encoder.fit_transform(first_df["Cabin_Level"])
first_df["Embarked"] = label_encoder.fit_transform(first_df["Embarked"])



#testing out a couple new pandas functions
print(first_df.info())
print(first_df.describe())

#Now generating summary table
print(pd.crosstab(df["Survived"], df["Sex"]))


'''Now doing some intense data visualization'''

plt.scatter(df["Pclass"], df["Age"], c = df["Survived"])
plt.xlabel("Class")
plt.ylabel("Age")
plt.title("Age and class versus survival rate")
plt.show()



#correlation matrix
corrmat = first_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
plt.title("Correlation heatmap")
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
print(corrmat["Survived"])

#What this shows, Fare, Pclass, Sex, and Cabin Number have largest impact on survival

#No scaling required in decision tree, splitting data into train and test
x_train, x_test, y_train, y_test = train_test_split(first_df[["Sex", "Pclass", "Age", "Fare", "Cabin_Level", "Cabin_Number", "Embarked", "Parch"]],
                                                    first_df["Survived"], test_size = 0.2, random_state = 42)
      


'''STEP3: Then applying a bagging algorithm'''

bag_classifier = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500,
                                                          max_samples = 100, bootstrap = True,
                                                          n_jobs = -1)
bag_classifier.fit(x_train, y_train)
train_prediction = bag_classifier.predict(x_train)
test_prediction = bag_classifier.predict(x_test)


'''STEP4: Making prediction and determining model accuracy'''
#Determining the model accuracy
test_score = accuracy_score(y_test, test_prediction)
train_score = accuracy_score(y_train, train_prediction)
print("Test Accuracy: ", test_score)
print("Train Accuracy: ", train_score)







