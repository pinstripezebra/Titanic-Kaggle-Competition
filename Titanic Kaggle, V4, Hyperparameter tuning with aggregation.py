# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 14:33:23 2020

@author: seelc
"""
'''
In this version will sampe both a bagging and pasting method to improve accuracy

'''


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import accuracy_score

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

'''STEP1: First loading and cleaning data'''

df = pd.read_csv("C:\\Users\\seelc\\OneDrive\\Desktop\\Projects\\Titanic Kaggle\\train.csv")
test_df = pd.read_csv("C:\\Users\\seelc\\OneDrive\\Desktop\\Projects\\Titanic Kaggle\\test (1).csv")


#First cleaning up the age column, if their age is empty set equal to the median age
df['Age'] = df['Age'].fillna(df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

#Cleaning up the cabin column. If there is no cabin listed set equal to zero
df['Cabin'] = df['Cabin'].fillna("None")
test_df['Cabin'] = test_df['Cabin'].fillna("None")

#Creating cabin level and cabin number in order to generate easy to analyse factors
df['Cabin_Level'] = df['Cabin'].copy()
df['Cabin_Number'] = df['Cabin'].copy()

test_df['Cabin_Level'] = test_df['Cabin'].copy()
test_df['Cabin_Number'] = test_df['Cabin'].copy()



#applying manual cleaning
df["Cabin"].loc[75] = "G73"
df["Cabin"].loc[128] = "E69"
df["Cabin"].loc[292] = "None"
df["Cabin"].loc[327] = "None"
df["Cabin"].loc[339] = "None"
df["Cabin"].loc[473] = "None"
df["Cabin"].loc[699] = "G63"

test_df["Cabin"].loc[75] = "G73"
test_df["Cabin"].loc[128] = "E69"
test_df["Cabin"].loc[292] = "None"
test_df["Cabin"].loc[327] = "None"
test_df["Cabin"].loc[339] = "None"
test_df["Cabin"].loc[473] = "None"
test_df["Cabin"].loc[699] = "G63"


for index, value in df['Cabin_Level'].items():
    if value == "None":
        pass
    else:
        split_cabin = value.split(" ")
        level = split_cabin[0][0:1]
        number = split_cabin[0][1:]
        df['Cabin_Level'][index] = level
        df['Cabin_Number'][index] = number
        
#Now doing the same thing for the test_df
for index, value in test_df['Cabin_Level'].items():
    if value == "None":
        pass
    else:
        split_cabin = value.split(" ")
        level = split_cabin[0][0:1]
        number = split_cabin[0][1:]
        test_df['Cabin_Level'][index] = level
        test_df['Cabin_Number'][index] = number        
        
ticket_int = df['Ticket'].copy().rename("ticket_int")
ticket_str = df['Ticket'].copy().rename("ticket_str")

#Splitting ticket into two columns one with the integer and one with the string
for index, value in ticket_int.items():
    value = value.split(" ")
    print(value)
    print(len(value))
    if len(value) == 1:
        ticket_str[index] = 0
        ticket_int[index] = value[0]
    elif len(value) == 2:
        ticket_int[index] = value[1]
        ticket_str[index] = value[0]
        
#Now doing the same thing for test_Df
ticket_int_test = test_df['Ticket'].copy().rename("ticket_int")
ticket_str_test = test_df['Ticket'].copy().rename("ticket_str")

#Splitting ticket into two columns one with the integer and one with the string
for index, value in ticket_int_test.items():
    value = value.split(" ")
    print(value)
    print(len(value))
    if len(value) == 1:
        ticket_str_test[index] = 0
        ticket_int_test[index] = value[0]
    elif len(value) == 2:
        ticket_int_test[index] = value[1]
        ticket_str_test[index] = value[0]
        
df = pd.concat([df, ticket_int, ticket_str], axis = 1, sort = False)

test_df = pd.concat([test_df, ticket_int_test, ticket_str_test], axis = 1, sort = False)

#feature engineering on cabin number
df['Cabin_Number_Grouped'] = df["Cabin_Number"].copy()
for index, value in df['Cabin_Number_Grouped'].items():
    if value == "None" or type(value) == str:
        value = 0
    else:
        print("Value :", value)
        print("index :", index)
        value = int(value)
        
    if 0<value<20:
        value = 1
    elif 20<value<40:
        value = 2
    elif 40<value<60:
        value = 3
    elif 60<value<80:
        value = 4
    elif 80<value<100:
        value = 5
    else:
        value = 6
    df['Cabin_Number_Grouped'][index] = value

'''STEP2: Transforming data in preparation for machine learning'''

first_df = df[["Sex", "Pclass", "Age", "Fare", "Survived", "Cabin_Level",'Cabin_Number_Grouped']].copy()

#First encoding categorical variables
label_encoder = LabelEncoder()
first_df["Sex"] = label_encoder.fit_transform(first_df["Sex"])
first_df["Cabin_Level"] = label_encoder.fit_transform(first_df["Cabin_Level"])

#Now encoding categorical for test
test_df["Sex"] = label_encoder.fit_transform(test_df["Sex"])
test_df["Cabin_Level"] = label_encoder.fit_transform(test_df["Cabin_Level"])


#No scaling required in decision tree, splitting data into train and test
x_train, x_test, y_train, y_test = train_test_split(first_df[["Sex", "Pclass", "Age", "Fare", "Cabin_Level"]],
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


'''STEP5: Making prediction on test dataset and exporting to excel for grading'''
print(test_df[["Sex", "Pclass", "Age", "Fare", "Cabin_Level"]].max())
test_df[["Sex", "Pclass", "Age", "Fare", "Cabin_Level"]] = test_df[["Sex", "Pclass", "Age", "Fare", "Cabin_Level"]].fillna(0)
prediction_for_submission = bag_classifier.predict(test_df[["Sex", "Pclass", "Age", "Fare", "Cabin_Level"]])

#Now writing results to excel
pd.DataFrame([test_df['PassengerId'],prediction_for_submission]).to_excel("For Submission.xlsx")