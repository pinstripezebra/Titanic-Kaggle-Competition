# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 08:08:58 2020

@author: seelc
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import accuracy_score


df = pd.read_csv("C:\\Users\\seelc\\OneDrive\\Desktop\\Projects\\Titanic Kaggle\\train.csv")

#First cleaning up the age column, if their age is empty set equal to the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

#Cleaning up the cabin column. If there is no cabin listed set equal to zero
df['Cabin'] = df['Cabin'].fillna("None")

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
        
df = pd.concat([df, ticket_int, ticket_str], axis = 1, sort = False)

'''Visualization'''
fig, axes = plt.subplots(2,2)

axes[0,0].scatter(df['Pclass'], df['Fare'],c = df['Survived'])
axes[0,0].set_title('Ticket cost and Class')
axes[0,0].set_xlabel('PClass')
axes[0,0].set_ylabel('Ticket Cost')


axes[0,1].scatter(df['Age'], df['Sex'], c = df['Survived'])
axes[0,1].set_title('Gender and Age')
axes[0,1].set_xlabel('Age')
axes[0,1].set_ylabel('Gender')

axes[1,0].scatter(df['PassengerId'], df['Parch'], c = df['Survived'])
axes[1,0].set_title('Children and Survival Rate')
axes[1,0].set_xlabel('Passeger ID')
axes[1,0].set_ylabel('Number of Children')


fig.tight_layout()
plt.show()

'''First decision tree with minimum sized dataframe'''
first_df = df[["Sex", "Pclass", "Age", "Fare", "Survived"]].copy()

#First encoding categorical variables
label_encoder = LabelEncoder()
first_df["Sex"] = label_encoder.fit_transform(first_df["Sex"])


#No scaling required in decision tree, splitting data into train and test
x_train, x_test, y_train, y_test = train_test_split(first_df[["Sex", "Pclass", "Age", "Fare"]],
                                                    first_df["Survived"], test_size = 0.2, random_state = 42)
      
first_tree = DecisionTreeClassifier(max_depth = 3)
first_predictor= first_tree.fit(x_train, y_train)
  
#Predicting outcomes on test and train sets      
test_prediction = first_predictor.predict(x_test)
train_prediction = first_predictor.predict(x_train)

#Visualizing results
fig = plt.figure(figsize=(25,20))
tree.plot_tree(first_predictor)

#Determining the model accuracy
test_score = accuracy_score(y_test, test_prediction)
train_score = accuracy_score(y_train, train_prediction)
print("Test Accuracy: ", test_score)
print("Train Accuracy: ", train_score)
