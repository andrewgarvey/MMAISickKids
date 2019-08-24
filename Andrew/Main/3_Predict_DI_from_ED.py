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

# ML based imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report

# ----------------------------------------------------------------------------------------------------------------------
## Prep data splits
# Import data
ML_Clean = pd.read_csv('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Data/ML_Clean.csv')

# Split data for modeling
test = a

# Split train and test, randomly 80/20 and also split each of the y away from x

# ----------------------------------------------------------------------------------------------------------------------
## Basic Random forest

# Set initial directory

# Set all the hyper-variables


# some loop that does random forest but for each of the aimed predictions


# Check out results, in particular confusion matrix, I think what we aim for is a good ROC curve stats,

# ----------------------------------------------------------------------------------------------------------------------
## Logistic Regression

# Set initial directory
os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Model/LogisticRegression')

# Set all the hyper-variables


# some loop that does random forest but for each of the aimed predictions


# Check out results, in particular confusion matrix, I think what we aim for is a good ROC curve stats,
