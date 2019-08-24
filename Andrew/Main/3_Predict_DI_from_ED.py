"""
Author: Andrew Garvey
Partner: Sargon Morad
Date: Aug 24, 2019
Client: Hospital for Sick Children

Title: Predict_DI_from_ED

Purpose:
-   Turn cleaned data into usable ml data
"""
# clear user created variables
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

del name

#Basic Imports
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import datetime as dt
import os

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report

# set seed
Random_State = 42

# ----------------------------------------------------------------------------------------------------------------------
## Prep data splits
# Import data
ML_Clean = pd.read_csv('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Data/ML_Clean.csv')

# Split data for modeling
Modalities = ['X-Ray', 'US', 'MRI', 'CT']

X = ML_Clean.drop(Modalities, axis=1)
y = ML_Clean[Modalities]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=Random_State)

# ----------------------------------------------------------------------------------------------------------------------
## Basic Random Forest

# Set initial directory

# Set all the hyper-variables


# some loop that does random forest but for each of the aimed predictions


# Check out results, in particular confusion matrix, I think what we aim for is a good ROC curve stats,

# ----------------------------------------------------------------------------------------------------------------------
## Logistic Regression

# Set initial directory

# Set all the hyper-variables

# some loop that does random forest but for each of the aimed predictions

# Check out results, in particular confusion matrix, I think what we aim for is a good ROC curve stats,
